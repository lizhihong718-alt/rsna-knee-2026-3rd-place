"""Grouped single-fold CoAtNet cohorts with globally isolated expert studies."""
from pathlib import Path

import numpy as np
import pandas as pd

from kneemri.raptor_training import load_cohort as load_historical_cohort, sha256


def load_cohort(cfg):
    """Return the historical tuple while adding explicit selection/monitor roles.

    The existing five-fold assignments are consumed unchanged. Only the selected
    fold is evaluated, and expert-labeled studies and every report-group peer are
    removed before either weak training or weak validation is constructed.
    """
    all_ids, eligible, gold_ids, labels, audit = load_historical_cohort(cfg)
    strategy = cfg.get('validation_strategy', 'gold')
    outer = cfg.get('outer_holdout_fold')
    if outer is not None and (strategy != 'grouped_cv' or isinstance(outer, bool)
            or not isinstance(outer, int) or outer not in range(5)
            or outer == cfg.get('val_fold', 0)):
        raise ValueError('outer_holdout_fold requires a distinct grouped_cv fold in 0..4')
    if strategy == 'gold':
        audit.update(validation_strategy='gold', selection_label_source='official_gold',
                     gold_study_uids=gold_ids, gold_monitor_study_uids=[],
                     protected_validation_study_uids=gold_ids + audit['excluded_report_group_peers'],
                     selection_metric='macro_roc_auc', metric_target_threshold=.5)
        return all_ids, eligible, gold_ids, labels, audit
    if strategy not in {'grouped_cv', 'full_refit'}:
        raise ValueError('validation_strategy must be gold, grouped_cv, or full_refit')
    num_folds, val_fold = cfg.get('num_folds', 5), cfg.get('val_fold', 0)
    if (isinstance(num_folds, bool) or num_folds != 5 or isinstance(val_fold, bool)
            or not isinstance(val_fold, int) or not 0 <= val_fold < num_folds):
        raise ValueError('grouped_cv requires exactly five folds and one val_fold in 0..4')
    folds_path = Path(cfg.get('folds_csv', ''))
    if not folds_path.is_absolute() or not folds_path.is_file():
        raise ValueError('folds_csv must name the absolute existing five-fold assignment file')
    table = pd.read_csv(folds_path)
    required = ['StudyInstanceUID', 'group', 'fold']
    if not set(required).issubset(table.columns) or table[required].isna().any().any():
        raise ValueError('fold assignment requires finite StudyInstanceUID, group and fold fields')
    if table.StudyInstanceUID.duplicated().any() or not set(all_ids) <= set(table.StudyInstanceUID):
        raise ValueError('incomplete or duplicate fold assignments')
    fold_values = pd.to_numeric(table.fold, errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(fold_values).all() or not np.equal(fold_values, np.floor(fold_values)).all()
            or set(fold_values) != set(range(num_folds))):
        raise ValueError('fold assignments must contain exactly the integers 0..4')
    table['fold'] = fold_values.astype(int)
    if (table.groupby('group').fold.nunique() > 1).any():
        raise ValueError('a related study group crosses validation folds')
    table = table.set_index('StudyInstanceUID')
    groups = pd.read_csv(cfg['groups_csv']).set_index('StudyInstanceUID')['group']
    if groups.loc[all_ids].isna().any() or not table.loc[all_ids, 'group'].equals(groups.loc[all_ids]):
        raise ValueError('fold group identities differ from the protected cohort grouping table')
    # An inner teacher must never train on, select on, or monitor the outer fold.
    # Keep the original fold table intact so report-group isolation is auditable.
    active_eligible = [uid for uid in eligible if table.at[uid, 'fold'] != outer]
    outer_ids = [uid for uid in all_ids if table.at[uid, 'fold'] == outer]
    outer_groups = set(table.loc[outer_ids, 'group'])
    selection_train_ids = [uid for uid in active_eligible if table.at[uid, 'fold'] != val_fold]
    selection_val_ids = [uid for uid in active_eligible if table.at[uid, 'fold'] == val_fold]
    if not selection_train_ids or not selection_val_ids:
        raise ValueError('the selected fold requires nonempty weak training and validation cohorts')
    selection_train_groups = set(table.loc[selection_train_ids, 'group'])
    val_groups = set(table.loc[selection_val_ids, 'group'])
    gold_groups = set(table.loc[gold_ids, 'group'])
    if (selection_train_groups & val_groups
            or (selection_train_groups | val_groups) & gold_groups):
        raise ValueError('study groups leak between training, selection or gold monitoring')
    if strategy == 'full_refit':
        train_ids, val_ids = list(eligible), []
        train_groups = selection_train_groups | val_groups
        protected_groups = gold_groups
    else:
        train_ids, val_ids = selection_train_ids, selection_val_ids
        train_groups = selection_train_groups
        protected_groups = val_groups | gold_groups | outer_groups
    protected = [uid for uid in all_ids if table.at[uid, 'group'] in protected_groups]
    audit.update(
        validation_strategy=strategy, num_folds=num_folds, val_fold=val_fold,
        trained_study_uids=train_ids, validation_study_uids=val_ids,
        gold_study_uids=gold_ids,
        gold_monitor_study_uids=gold_ids if cfg.get('monitor_gold', True) else [],
        trained_groups=sorted(train_groups),
        validation_groups=sorted(val_groups) if strategy == 'grouped_cv' else [],
        parent_selection_train_study_uids=selection_train_ids,
        parent_selection_validation_study_uids=selection_val_ids,
        parent_selection_validation_groups=sorted(val_groups),
        excluded_gold_groups=sorted(gold_groups),
        protected_validation_study_uids=protected,
        fold_assignments={uid: int(table.at[uid, 'fold']) for uid in all_ids},
        folds_csv=str(folds_path), folds_sha256=sha256(folds_path),
        eligible_fold_counts={str(fold): int(sum(table.at[uid, 'fold'] == fold for uid in eligible))
                              for fold in range(num_folds)},
        selection_label_source=('report_weak' if strategy == 'grouped_cv'
                                else 'parent_grouped_cv_checkpoint'),
        selection_metric=('macro_roc_auc_weak_target_gt_0.5' if strategy == 'grouped_cv'
                          else 'none_fixed_parent_selected_epoch'),
        metric_target_threshold=.5,
        gold_monitor_role=('gold58_monitor_after_fold_selection_only' if strategy == 'grouped_cv'
                           else 'gold58_monitor_after_fixed_full_corpus_refit_only'),
    )
    if outer is not None:
        audit.update(outer_holdout_fold=outer, outer_holdout_study_uids=outer_ids,
                     outer_holdout_groups=sorted(outer_groups),
                     outer_holdout_role='excluded from inner teacher training and checkpoint selection')
        if ((set(train_ids) | set(val_ids)) & set(outer_ids)
                or (train_groups | val_groups) & outer_groups):
            raise ValueError('outer holdout leaks into an inner teacher cohort')
    return all_ids, train_ids, val_ids, labels, audit


def assert_checkpoint_is_held_out(checkpoint, cohort):
    """Reject previously trained selection studies, including related groups."""
    source = checkpoint.get('cohort', {})
    trained = set(source.get('trained_study_uids', []))
    if cohort.get('validation_strategy') in {'grouped_cv', 'full_refit'} and not trained:
        raise ValueError('grouped_cv/full_refit checkpoint evaluation requires its source training cohort')
    protected = set(cohort.get('protected_validation_study_uids', []))
    protected.update(cohort['validation_study_uids'])
    protected.update(cohort.get('gold_study_uids', []))
    protected.update(cohort.get('excluded_report_group_peers', []))
    if trained & protected:
        raise ValueError('checkpoint training cohort overlaps protected validation studies or peers')
    if set(source.get('trained_groups', [])) & set(cohort.get('validation_groups', [])):
        raise ValueError('checkpoint training groups overlap the selected validation fold')
