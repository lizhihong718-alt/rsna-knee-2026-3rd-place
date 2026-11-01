#!/usr/bin/env python3
"""Train CoAtNet with a fixed grouped validation fold and isolated gold monitoring."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kneemri import raptor_reference as raptor
from kneemri.coatnet_model import CoAtNetAblationModel, build_model
from kneemri.labels import LABEL_COLS, read_labels
from kneemri.metrics import per_target_auc
from kneemri.coatnet_cohort import load_cohort, assert_checkpoint_is_held_out


DEFAULTS = dict(
    train_windows=12, eval_windows=42, eval_budgets=[12, 24, 42, 'all'],
    eval_sweep_every=4, window_mode='same_series', train_sampling='random',
    eval_sampling='uniform', deduplicate_windows=False, batch_size=2,
    gradient_accumulation=4, drop_last_train=False, epochs=16, workers=2, eval_workers=0,
    eval_encode_chunk=32, backbone_lr=3e-5, spatial_lr=None, head_lr=1e-3,
    weight_decay=.02, warmup_fraction=.15, gradient_clip=3.,
    gradient_checkpointing=True, intensity_gain=.10, rotation_deg=0.,
    scale_jitter=0., gamma_jitter=0., positive_weight='legacy',
    positive_weight_max=10., loss='bce', focal_gamma=2.,
    loss_normalization='global', rank_weight=0., rank_positive_threshold=.6,
    rank_negative_threshold=.4, freeze_bn=False, amp='auto', eval_amp='fp16', seed=2026,
    scheduler='onecycle', scheduler_total_epochs=None, log_every=25, checkpoint_every=0,
    normalization='imagenet', resolution=384, num_classes=12,
    label_mixing=False, label_mixing_cap=2.0,
    arch='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
    validation_role='gold58_checkpoint_selection', num_threads=2,
    validation_strategy='gold', num_folds=5, val_fold=0, monitor_gold=True,
    gold_monitor_min_selection_auc=None,
    refit_parent_checkpoint=None, refit_targets_receipt=None,
    # Throughput only; none of these changes the data, sampling or objective.
    # channels_last changes kernel choice (FP rounding), the others are exact.
    channels_last=True, checkpoint_scope='bn_stages', uint8_transfer=True, eval_cache_gb=32.,
    parallel_world_size=1,
    lr_tail_start_epoch=None, lr_tail_end_epoch=None, lr_tail_final_ratio=.02,
    fixed_positive_weights=None, checkpoint_selection='best', evaluation_every=1,
    # Per-update weight EMAs, one shadow per decay; evaluated and saved beside the raw
    # model, never fed back into training. Empty: the trainer is unchanged.
    ema_decays=[],
)
INFRA_KEYS = {'channels_last', 'checkpoint_scope', 'uint8_transfer', 'eval_cache_gb',
              'parallel_world_size'}
MODEL_KEYS = {'aggregate', 'attention_hidden', 'head_dropout', 'dropout',
              'attention_temperature', 'temperature', 'value_norm', 'group_embedding',
              'position_embedding', 'hierarchical', 'hierarchy_axis', 'spatial_query',
              'spatial_query_gap', 'spatial_residual_gated', 'spatial_hidden_dim',
              'spatial_logit_cap', 'spatial_residual_mix', 'spatial_slot_conditioned',
              'num_groups', 'position_dim', 'position_scale_mm', 'spatial_query_dim',
              'slot_embedding_dim', 'feature_projection_dim', 'pool_mix_attention_init',
              'label_mixing', 'label_mixing_cap', 'probability_pooling', 'intermediate_readout'}
DATA_KEYS = {'corpus', 'corpus_format', 'provenance', 'labels_csv', 'gold_csv', 'groups_csv', 'initial_weights',
             'confidence_csv', 'valid_mask_csv', 'legacy_sampling', 'include_provenance',
             'train_label_source', 'eval_label_source', 'folds_csv', 'outer_holdout_fold',
             'backbone_init_checkpoint'}


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
    temporary.replace(path)


def save_atomic(value, path):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(_):
    seed_all(torch.initial_seed() % 2**32)
    torch.set_num_threads(1)


def valid_budget(value):
    return value == 'all' or (isinstance(value, int) and not isinstance(value, bool) and value > 0)


def smoke_coverage_subset(ids, labels, minimum=8):
    """Choose a small deterministic subset with both metric states for every label."""
    ids = list(ids)
    states = {}
    for uid in ids:
        values = np.asarray(labels[uid])
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError('smoke coverage requires twelve finite labels per study')
        states[uid] = {(index, bool(value > .5)) for index, value in enumerate(values)}
    required = {(index, state) for index in range(12) for state in (False, True)}
    available = set().union(*(states.values())) if states else set()
    if not required <= available:
        missing = sorted(required - available)
        raise ValueError(f'smoke cohort cannot cover both metric states: {missing}')
    selected, remaining, uncovered = [], list(ids), set(required)
    while uncovered:
        gains = [len(states[uid] & uncovered) for uid in remaining]
        best = int(np.argmax(gains))
        if gains[best] == 0:
            raise RuntimeError('smoke coverage selection made no progress')
        uid = remaining.pop(best)
        selected.append(uid)
        uncovered -= states[uid]
    for uid in remaining:
        if len(selected) >= minimum:
            break
        selected.append(uid)
    return selected


def normalize_config(config):
    config = dict(config)
    aliases = {'eval_window_budgets': 'eval_budgets', 'pos_weight_mode': 'positive_weight',
               'torch_threads': 'num_threads', 'provenance_dir': 'provenance'}
    for alias, canonical in aliases.items():
        if alias in config:
            if canonical in config and config[canonical] != config[alias]:
                raise ValueError(f'contradictory {alias} and {canonical}')
            config[canonical] = config.pop(alias)
    if 'bn_strategy' in config:
        strategy = config.pop('bn_strategy')
        if strategy not in {'train', 'freeze', 'frozen', 'update'}:
            raise ValueError('bn_strategy must be train/update or freeze/frozen')
        freeze = strategy in {'freeze', 'frozen'}
        if 'freeze_bn' in config and config['freeze_bn'] != freeze:
            raise ValueError('contradictory bn_strategy and freeze_bn')
        config['freeze_bn'] = freeze
    unknown = set(config) - set(DEFAULTS) - MODEL_KEYS - DATA_KEYS
    if unknown:
        raise ValueError(f'unsupported configuration fields: {sorted(unknown)}')
    cfg = {**DEFAULTS, **config}
    if cfg['checkpoint_selection'] not in {'best', 'fixed_final'}:
        raise ValueError('checkpoint_selection must be best or fixed_final')
    if (isinstance(cfg['evaluation_every'], bool) or not isinstance(cfg['evaluation_every'], int)
            or cfg['evaluation_every'] < 1):
        raise ValueError('evaluation_every must be a positive integer')
    if cfg['checkpoint_selection'] == 'best' and cfg['evaluation_every'] != 1:
        raise ValueError('sparse evaluation requires fixed_final checkpoint selection')
    if cfg['fixed_positive_weights'] is not None:
        values = np.asarray(cfg['fixed_positive_weights'], dtype=np.float64)
        if values.shape != (12,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('fixed_positive_weights requires twelve finite positive values')
        cfg['fixed_positive_weights'] = values.tolist()
    decays = cfg['ema_decays']
    if (not isinstance(decays, list) or len(set(decays)) != len(decays)
            or any(isinstance(d, bool) or not isinstance(d, float) or not 0 < d < 1 for d in decays)):
        raise ValueError('ema_decays must be a list of distinct floats in (0, 1)')
    if decays and cfg['checkpoint_selection'] != 'fixed_final':
        raise ValueError('weight EMA is only defined for fixed_final runs')
    if cfg['lr_tail_start_epoch'] is not None:
        start = cfg['lr_tail_start_epoch']
        end = cfg['lr_tail_end_epoch'] if cfg['lr_tail_end_epoch'] is not None else cfg['epochs']
        if (cfg['scheduler'] != 'onecycle' or isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int) or not 0 <= start < end
                or end > (cfg['scheduler_total_epochs'] or cfg['epochs'])
                or not 0 < cfg['lr_tail_final_ratio'] <= 1):
            raise ValueError('invalid OneCycle cosine-tail schedule')
        cfg['lr_tail_end_epoch'] = end
    if cfg['validation_strategy'] not in {'gold', 'grouped_cv', 'full_refit'}:
        raise ValueError('validation_strategy must be gold, grouped_cv, or full_refit')
    outer = cfg.get('outer_holdout_fold')
    if outer is not None and (cfg['validation_strategy'] != 'grouped_cv'
            or isinstance(outer, bool) or not isinstance(outer, int)
            or outer not in range(5) or outer == cfg['val_fold']):
        raise ValueError('outer_holdout_fold requires a distinct grouped_cv fold in 0..4')
    if cfg['validation_strategy'] in {'grouped_cv', 'full_refit'}:
        if (cfg['num_folds'] != 5 or isinstance(cfg['val_fold'], bool)
                or not isinstance(cfg['val_fold'], int) or not 0 <= cfg['val_fold'] < 5):
            raise ValueError('grouped_cv/full_refit requires five folds and one val_fold in 0..4')
        if cfg['validation_strategy'] == 'grouped_cv' and (
                'validation_role' not in config or cfg['validation_role'].startswith('gold')):
            cfg['validation_role'] = f'grouped_cv_fold{cfg["val_fold"]}_weak_checkpoint_selection'
        if cfg['validation_strategy'] == 'full_refit' and (
                'validation_role' not in config or 'selection' in cfg['validation_role']):
            cfg['validation_role'] = 'fixed_full_corpus_refit_no_validation_selection'
    cfg['eval_label_source'] = ('report_weak' if cfg['validation_strategy'] == 'grouped_cv'
                                else 'official_gold')
    budgets = cfg['train_windows'] if isinstance(cfg['train_windows'], list) else [cfg['train_windows']]
    if not budgets or not all(valid_budget(k) for k in budgets):
        raise ValueError('train_windows requires positive integers, all, or a nonempty list of those')
    if not valid_budget(cfg['eval_windows']) or not all(valid_budget(k) for k in cfg['eval_budgets']):
        raise ValueError('invalid evaluation budget')
    if min(cfg[k] for k in ['batch_size', 'gradient_accumulation', 'epochs', 'eval_encode_chunk',
                           'eval_sweep_every', 'num_threads', 'log_every']) < 1:
        raise ValueError('batch, accumulation, epoch, chunk, sweep and logging counts must be positive')
    if isinstance(cfg['parallel_world_size'], bool) or not isinstance(cfg['parallel_world_size'], int) \
            or cfg['parallel_world_size'] < 1:
        raise ValueError('parallel_world_size must be a positive integer')
    if not isinstance(cfg['drop_last_train'], bool):
        raise ValueError('drop_last_train must be boolean')
    gold_threshold = cfg['gold_monitor_min_selection_auc']
    if (gold_threshold is not None
            and (isinstance(gold_threshold, bool) or not isinstance(gold_threshold, (int, float))
                 or not math.isfinite(gold_threshold) or not 0 <= gold_threshold <= 1)):
        raise ValueError('gold_monitor_min_selection_auc must be null or a finite AUC in [0,1]')
    if cfg['positive_weight'] not in {'legacy', 'none', 'ratio', 'sqrt'}:
        raise ValueError('positive_weight must be legacy, none, ratio, or sqrt')
    if cfg['positive_weight_max'] < 1 or not math.isfinite(cfg['positive_weight_max']):
        raise ValueError('positive_weight_max must be finite and at least 1')
    if cfg['loss'] not in {'bce', 'focal'} or cfg['focal_gamma'] < 0:
        raise ValueError('loss requires bce or focal with nonnegative gamma')
    if cfg['loss_normalization'] not in {'global', 'per_class'}:
        raise ValueError('loss_normalization must be global or per_class')
    if (not math.isfinite(cfg['rank_weight']) or cfg['rank_weight'] < 0
            or not 0 <= cfg['rank_negative_threshold'] < cfg['rank_positive_threshold'] <= 1):
        raise ValueError('rank loss requires a nonnegative weight and ordered thresholds in [0,1]')
    if cfg['rank_weight'] and not cfg.get('spatial_residual_gated', False):
        raise ValueError('detached-backbone rank replay requires spatial_residual_gated')
    if cfg['spatial_lr'] is not None and (cfg['spatial_lr'] <= 0 or not math.isfinite(cfg['spatial_lr'])):
        raise ValueError('spatial_lr must be finite and positive when configured')
    if cfg['label_mixing'] and (cfg['label_mixing_cap'] <= 0 or not math.isfinite(cfg['label_mixing_cap'])):
        raise ValueError('label_mixing_cap must be finite and positive')
    if cfg['amp'] not in {'auto', 'bf16', 'fp16', 'off'} or cfg['eval_amp'] not in {'bf16', 'fp16', 'off'}:
        raise ValueError('amp/eval_amp must use the supported auto, bf16, fp16, or off modes')
    if cfg['scheduler'] not in {'onecycle', 'cosine', 'constant'}:
        raise ValueError('scheduler must be onecycle, cosine, or constant')
    scheduler_epochs = cfg['scheduler_total_epochs']
    if scheduler_epochs is not None and (isinstance(scheduler_epochs, bool)
                                         or not isinstance(scheduler_epochs, int)
                                         or scheduler_epochs < cfg['epochs']):
        raise ValueError('scheduler_total_epochs must be an integer at least as large as epochs')
    if cfg['validation_strategy'] == 'full_refit':
        parent = cfg['refit_parent_checkpoint']
        if not isinstance(parent, str) or not parent:
            raise ValueError('full_refit requires refit_parent_checkpoint')
        if cfg['scheduler_total_epochs'] is None:
            raise ValueError('full_refit requires the parent scheduler_total_epochs')
        if cfg['checkpoint_every']:
            raise ValueError('full_refit saves only its fixed final checkpoint')
        if not cfg['monitor_gold']:
            raise ValueError('full_refit requires one post-training gold monitor evaluation')
        if cfg['gold_monitor_min_selection_auc'] is not None:
            raise ValueError('full_refit has no fold-selection score for a Gold monitor threshold')
    if not 0 < cfg['warmup_fraction'] < 1:
        raise ValueError('warmup_fraction must lie in (0,1)')
    if (cfg['rotation_deg'] < 0 or not 0 <= cfg['scale_jitter'] < 1
            or not 0 <= cfg['gamma_jitter'] < 1):
        raise ValueError('augmentation magnitudes must be nonnegative and jitter values below 1')
    if cfg['normalization'] not in {'unit', 'imagenet', 'timm_half'} or cfg['num_classes'] != 12:
        raise ValueError('this cache experiment requires an explicit supported normalization and 12 classes')
    if list(raptor.LAB) != list(LABEL_COLS):
        raise ValueError('reference/model/metric label order mismatch')
    if not isinstance(cfg['channels_last'], bool) or not isinstance(cfg['uint8_transfer'], bool):
        raise ValueError('channels_last and uint8_transfer must be boolean')
    if cfg['checkpoint_scope'] not in {'all', 'bn_stages'}:
        raise ValueError('checkpoint_scope must be all or bn_stages')
    cache = cfg['eval_cache_gb']
    if isinstance(cache, bool) or not isinstance(cache, (int, float)) or not math.isfinite(cache) or cache < 0:
        raise ValueError('eval_cache_gb must be a finite nonnegative size')
    return cfg


def validate_refit_parent(cfg, cohort):
    """Bind a full refit to a completed selected or prespecified-fold checkpoint."""
    path = Path(cfg['refit_parent_checkpoint'])
    if not path.is_absolute() or not path.is_file() or path.name not in {'best.pt', 'final.pt'}:
        raise ValueError('refit_parent_checkpoint must be an absolute existing best.pt or final.pt')
    fixed = path.name == 'final.pt'
    run = path.parent
    required = ['complete.json', 'history.json', 'config.json', 'cohort.json',
                'final_evaluation_matrix.json' if fixed else 'best_evaluation_matrix.json']
    if any(not (run / name).is_file() for name in required):
        raise ValueError('refit parent must be a completed grouped-fold run with full receipts')
    checkpoint_sha256 = hash_file(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    parent_cfg = normalize_config(checkpoint.get('config', {}))
    parent_cohort = checkpoint.get('cohort', {})
    if (checkpoint.get('lab') != LABEL_COLS or checkpoint.get('arch') != cfg['arch']
            or checkpoint.get('res') != cfg['resolution']):
        raise ValueError('refit parent model contract differs from the requested model')
    complete = json.loads((run / 'complete.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    recorded_cfg = json.loads((run / 'config.json').read_text())
    recorded_cohort = json.loads((run / 'cohort.json').read_text())
    if checkpoint.get('config') != recorded_cfg or parent_cohort != recorded_cohort:
        raise ValueError('refit parent checkpoint differs from its run receipts')
    if (parent_cfg['validation_strategy'] != 'grouped_cv'
            or parent_cohort.get('validation_strategy') != 'grouped_cv'
            or parent_cfg['val_fold'] != cfg['val_fold']
            or parent_cohort.get('val_fold') != cfg['val_fold']):
        raise ValueError('refit parent must use the configured grouped validation fold')
    if (complete.get('validation_strategy') != 'grouped_cv'
            or complete.get('val_fold') != cfg['val_fold']
            or complete.get('epochs_completed') != parent_cfg['epochs']
            or complete.get('smoke') is not False or complete.get('synthetic') is not False
            or complete.get('fixed_final_checkpoint_sha256' if fixed else 'best_checkpoint_sha256') != checkpoint_sha256):
        raise ValueError('refit parent completion receipt is invalid')
    selected_epoch = checkpoint.get('epoch', -1) + 1
    selection_auc = checkpoint.get('selection_auc')
    if (isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int)
            or selected_epoch < 1 or selected_epoch != cfg['epochs']
            or (not fixed and (not isinstance(selection_auc, (int, float)) or not math.isfinite(selection_auc)))):
        raise ValueError('refit epochs must equal the parent checkpoint selected epoch')
    rows = history.get('history', [])
    if fixed:
        if (parent_cfg['checkpoint_selection'] != 'fixed_final'
                or cfg['checkpoint_selection'] != 'fixed_final'
                or complete.get('checkpoint_selection') != 'fixed_final'
                or parent_cohort.get('selection_metric') != 'none_prespecified_final_epoch'
                or parent_cohort.get('fixed_epoch') != selected_epoch
                or selected_epoch != parent_cfg['epochs'] or selection_auc is not None
                or complete.get('best_selection_auc') is not None
                or history.get('best_selection_auc') is not None
                or history.get('epochs_completed') != selected_epoch
                or [row['epoch'] for row in rows] != list(range(1, selected_epoch + 1))
                or any(row.get('selection_auc') is not None for row in rows)):
            raise ValueError('refit parent does not prove the prespecified final epoch')
        evaluation = json.loads((run / 'final_evaluation_matrix.json').read_text())[str(parent_cfg['eval_windows'])]
        if (evaluation['epoch'] != selected_epoch or evaluation.get('selection_auc') is not None
                or evaluation['macro_auc'] != rows[-1]['diagnostic_auc']):
            raise ValueError('refit parent final diagnostic receipt differs')
    elif (history.get('epochs_completed') != parent_cfg['epochs'] or not rows
            or history.get('best_selection_auc') != selection_auc
            or max(row['selection_auc'] for row in rows) != selection_auc
            or not any(row['epoch'] == selected_epoch
                       and row['selection_auc'] == selection_auc for row in rows)):
        raise ValueError('refit parent history does not prove the selected epoch')
    parent_scheduler_epochs = parent_cfg['scheduler_total_epochs'] or parent_cfg['epochs']
    if cfg['scheduler_total_epochs'] != parent_scheduler_epochs:
        raise ValueError('refit scheduler_total_epochs must preserve the parent schedule horizon')
    ignored = {'validation_strategy', 'validation_role', 'eval_label_source', 'epochs',
               'scheduler_total_epochs', 'refit_parent_checkpoint', 'refit_targets_receipt', 'gold_monitor_min_selection_auc',
               *INFRA_KEYS}
    target_metadata = None
    if cfg.get('refit_targets_receipt'):
        if not fixed:
            raise ValueError('OOF target rebuilding requires a prespecified-final parent')
        target_metadata = validate_refit_targets(cfg, cohort, parent_cohort, checkpoint_sha256)
        ignored.add('labels_csv')
    differences = {key for key in set(cfg) | set(parent_cfg)
                   if key not in ignored and cfg.get(key) != parent_cfg.get(key)}
    if cfg.get('parallel_world_size', 1) > 1:
        same_effective_batch = (cfg['batch_size'] * cfg['gradient_accumulation']
                                == parent_cfg['batch_size'] * parent_cfg['gradient_accumulation'])
        if same_effective_batch:
            differences -= {'batch_size', 'gradient_accumulation'}
    if differences:
        raise ValueError(f'refit configuration differs from parent training fields: {sorted(differences)}')
    parent_train = set(parent_cohort.get('trained_study_uids', []))
    parent_validation = set(parent_cohort.get('validation_study_uids', []))
    refit_train = set(cohort['trained_study_uids'])
    if (not parent_train or not parent_validation or parent_train & parent_validation
            or parent_train | parent_validation != refit_train
            or parent_train != set(cohort['parent_selection_train_study_uids'])
            or parent_validation != set(cohort['parent_selection_validation_study_uids'])):
        raise ValueError('refit parent cohort does not partition the eligible weak-label studies')
    protected = set(cohort.get('gold_study_uids', []))
    protected.update(cohort.get('excluded_report_group_peers', []))
    if refit_train & protected:
        raise ValueError('full refit training cohort overlaps gold studies or report peers')
    for key in ['labels_sha256', 'gold_sha256', 'groups_sha256', 'folds_sha256']:
        if key == 'labels_sha256' and target_metadata:
            continue
        if parent_cohort.get(key) != cohort.get(key):
            raise ValueError(f'refit parent {key} differs from the full-refit cohort')
    metadata = {
        'parent_checkpoint': str(path),
        'parent_checkpoint_sha256': checkpoint_sha256,
        'parent_selected_epoch': selected_epoch,
        'parent_selection_auc': None if fixed else float(selection_auc),
        'parent_validation_role': parent_cfg['validation_role'],
        'parent_validation_strategy': 'grouped_cv',
        'parent_val_fold': parent_cfg['val_fold'],
        'parent_scheduler_total_epochs': parent_scheduler_epochs,
        'fresh_initialization_required': True,
        'parent_weights_used_for_initialization': False,
    }
    if fixed:
        metadata.update(parent_checkpoint_selection='fixed_final',
                        parent_diagnostic_auc=rows[-1]['diagnostic_auc'])
    if target_metadata:
        metadata['rebuilt_oof_targets'] = target_metadata
    del checkpoint
    return metadata


def validate_refit_targets(cfg, cohort, parent_cohort, checkpoint_sha256):
    """Allow a rebuilt label table only with complete, disjoint five-fold provenance."""
    path = Path(cfg['refit_targets_receipt'])
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != 'full-oof-refit-targets-v1'
            or receipt.get('teacher_fraction') != .5
            or receipt.get('teacher_predictions_per_study') != 1):
        raise ValueError('invalid full OOF target receipt')
    for source, digest in receipt['files'].items():
        if hash_file(source) != digest:
            raise ValueError('full OOF target provenance changed: ' + source)
    matches = [v for v in receipt['students'].values() if v['path'] == cfg['labels_csv']]
    if len(matches) != 1:
        raise ValueError('refit target table absent from receipt')
    entry = matches[0]
    if (entry['parent_checkpoint_sha256'] != checkpoint_sha256
            or entry['parent_labels_sha256'] != parent_cohort['labels_sha256']
            or entry['sha256'] != cohort['labels_sha256']
            or hash_file(cfg['labels_csv']) != entry['sha256']):
        raise ValueError('full OOF targets do not bind this parent and label table')
    eligible = set(cohort['trained_study_uids'])
    if set(receipt['study_uids']) != eligible or len(receipt['study_uids']) != len(eligible):
        raise ValueError('full OOF coverage differs from full refit cohort')
    protected = set(cohort['gold_study_uids']) | set(cohort['excluded_report_group_peers'])
    seen = set()
    teachers = receipt['teachers']
    if sorted(t['fold'] for t in teachers) != list(range(5)):
        raise ValueError('full OOF requires exactly five teachers')
    for teacher in teachers:
        if receipt['files'].get(teacher['cohort_path']) != hash_file(teacher['cohort_path']):
            raise ValueError('teacher cohort missing from pinned target sources')
        source = json.loads(Path(teacher['cohort_path']).read_text())
        train, held = set(source['trained_study_uids']), set(source['validation_study_uids'])
        expected = {uid for uid in eligible if cohort['fold_assignments'][uid] == teacher['fold']}
        if (source['folds_sha256'] != cohort['folds_sha256']
                or held != expected or train != eligible - expected
                or (train | held) & protected or seen & held
                or set(source['trained_groups']) & set(source['validation_groups'])):
            raise ValueError('full OOF teacher leakage or incomplete held-out coverage')
        seen.update(held)
    if seen != eligible:
        raise ValueError('incomplete full OOF coverage')
    return dict(receipt=str(path), receipt_sha256=hash_file(path),
                labels_sha256=entry['sha256'], variant=entry['variant'], studies=len(eligible))


def supervision_weights(cfg, train_ids, cohort):
    weights = {uid: np.ones(12, np.float32) for uid in train_ids}
    for key in ['confidence_csv', 'valid_mask_csv']:
        if not cfg.get(key):
            continue
        frame = read_labels(cfg[key])
        if not set(train_ids).issubset(frame.index):
            raise ValueError(f'{key} misses training studies')
        values = frame.loc[train_ids].to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'{key} must explicitly contain finite weights for every training label')
        if key == 'valid_mask_csv' and not np.isin(values, [0, 1]).all():
            raise ValueError('valid_mask_csv must be binary')
        for uid, row in zip(train_ids, values):
            weights[uid] *= row
        cohort[key + '_sha256'] = hash_file(cfg[key])
    return weights


def positive_weights(targets, weights, cfg):
    if cfg.get('fixed_positive_weights') is not None:
        return np.asarray(cfg['fixed_positive_weights'], dtype=np.float32)
    targets, weights = np.asarray(targets, np.float64), np.asarray(weights, np.float64)
    valid = np.isfinite(targets) & np.isfinite(weights) & (weights > 0)
    safe_y = np.where(valid, targets, 0)
    safe_w = np.where(valid, weights, 0)
    denominator = safe_w.sum(0)
    prevalence = np.divide((safe_y * safe_w).sum(0), denominator,
                           out=np.zeros(12, dtype=np.float64), where=denominator > 0)
    if cfg['positive_weight'] == 'none':
        return np.ones(12, np.float32)
    if cfg['positive_weight'] == 'legacy':
        prevalence = np.clip(prevalence, .03, .7)
    ratio = np.divide(1 - prevalence, prevalence, out=np.full(12, np.inf), where=prevalence > 0)
    if cfg['positive_weight'] == 'sqrt':
        ratio = np.sqrt(ratio)
    result = np.clip(ratio, 1, cfg['positive_weight_max'])
    result[denominator == 0] = 1
    return result.astype(np.float32)


def weighted_loss(logits, targets, weights, pos_weight, cfg, group_denominators=None):
    logits, targets, weights = logits.float(), targets.float(), weights.float()
    valid = torch.isfinite(targets) & torch.isfinite(weights) & (weights > 0)
    safe_y = torch.where(valid, targets, torch.zeros_like(targets))
    safe_w = torch.where(valid, weights, torch.zeros_like(weights))
    if cfg['loss'] == 'bce':
        values = F.binary_cross_entropy_with_logits(logits, safe_y, pos_weight=pos_weight, reduction='none')
    else:
        probability = logits.sigmoid()
        gamma = cfg['focal_gamma']
        values = (-pos_weight * safe_y * (1 - probability).pow(gamma) * F.logsigmoid(logits)
                  - (1 - safe_y) * probability.pow(gamma) * F.logsigmoid(-logits))
    values = torch.where(valid, values, torch.zeros_like(values)) * safe_w
    denominators = safe_w.sum(0) if group_denominators is None else group_denominators
    if cfg['loss_normalization'] == 'global':
        return values.sum() / denominators.sum().clamp_min(torch.finfo(torch.float32).eps)
    present = denominators > 0
    class_loss = values.sum(0) / denominators.clamp_min(torch.finfo(torch.float32).eps)
    return class_loss.sum() / present.sum().clamp_min(1)


def pairwise_rank_loss(logits, targets, positive_threshold=.6, negative_threshold=.4):
    """Macro softplus rank loss over confident within-finding study pairs."""
    scores, target = logits.float(), targets.float()
    if scores.ndim != 2 or scores.shape != target.shape or scores.shape[1] != 12:
        raise ValueError('rank logits and targets must align as [B,12]')
    finite = torch.isfinite(target)
    differences = scores.unsqueeze(1) - scores.unsqueeze(0)
    pairs = ((finite & (target > positive_threshold)).unsqueeze(1)
             & (finite & (target < negative_threshold)).unsqueeze(0))
    counts = pairs.sum(dim=(0, 1))
    per_class = (F.softplus(-differences) * pairs).sum(dim=(0, 1)) / counts.clamp_min(1)
    active = counts > 0
    loss = per_class[active].mean() if active.any() else scores.sum() * 0.
    return loss, counts.sum()


def _pad_studies(tensor, windows, value=0):
    if tensor.shape[1] == windows:
        return tensor
    output = tensor.new_full((tensor.shape[0], windows, *tensor.shape[2:]), value)
    output[:, :tensor.shape[1]] = tensor
    return output


def replay_rank_group(model, records):
    """Replay spatial/head modules for one effective batch from detached maps."""
    if not records:
        raise ValueError('rank replay requires at least one microbatch')
    windows = max(record['metadata']['window_mask'].shape[1] for record in records)
    features, targets = [], []
    metadata = {key: [] for key in records[0]['metadata']}
    fill = {'window_mask': False, 'group_ids': -1, 'series_ids': -1,
            'plane_ids': -1, 'positions': 0, 'position_mask': False}
    for record in records:
        current = record['metadata']
        features.append(_pad_studies(model.decode_backbone_features(
            record['feature_maps'], current['window_mask'], current['group_ids']), windows))
        targets.append(record['targets'])
        for key in metadata:
            metadata[key].append(_pad_studies(current[key], windows, fill[key]))
    combined_metadata = {key: torch.cat(values) for key, values in metadata.items()}
    logits = model.head(torch.cat(features), **combined_metadata)
    return logits, torch.cat(targets)


def accumulation_groups(loader, steps, accumulation):
    iterator = iter(loader)
    for start in range(0, steps, accumulation):
        batches = list(itertools.islice(iterator, min(accumulation, steps - start)))
        if not batches:
            raise RuntimeError('training loader ended before its declared number of steps')
        yield start, batches


def group_supervision(batches, weights):
    """Compute exact effective-batch loss denominators before microbatch forwards."""
    denominators = torch.zeros(12)
    for batch in batches:
        label_weight = torch.from_numpy(np.stack([weights[uid] for uid in batch['study_ids']]))
        valid = torch.isfinite(batch['targets']) & batch['window_mask'].any(1, keepdim=True)
        label_weight = torch.where(valid, label_weight, torch.zeros_like(label_weight))
        batch['label_weights'] = label_weight
        denominators += label_weight.sum(0)
    return denominators


class OneCycleWithCosineTail(torch.optim.lr_scheduler.OneCycleLR):
    """Preserve the v152 OneCycle momentum and its independently fixed LR tail."""

    def __init__(self, optimizer, *, tail_start_step, tail_end_step, tail_final_ratio, **kwargs):
        self.tail_start_step = int(tail_start_step)
        self.tail_end_step = int(tail_end_step)
        self.tail_final_ratio = float(tail_final_ratio)
        super().__init__(optimizer, **kwargs)

    def _original_lr_at(self, group, step):
        start_step = 0.
        for index, phase in enumerate(self._schedule_phases):
            end_step = phase['end_step']
            if step <= end_step or index == len(self._schedule_phases) - 1:
                progress = (step - start_step) / (end_step - start_step)
                return self._anneal_func(group[phase['start_lr']], group[phase['end_lr']], progress)
            start_step = end_step
        raise RuntimeError('OneCycle phase missing')

    def get_lr(self):
        original = super().get_lr()
        if self.last_epoch <= self.tail_start_step:
            return original
        progress = min(1., (self.last_epoch - self.tail_start_step)
                       / (self.tail_end_step - self.tail_start_step))
        cosine = .5 * (1 + math.cos(math.pi * progress))
        return [group['max_lr'] * self.tail_final_ratio
                + (self._original_lr_at(group, self.tail_start_step)
                   - group['max_lr'] * self.tail_final_ratio) * cosine
                for group in self.optimizer.param_groups]


def create_scheduler(optimizer, cfg, updates_per_epoch):
    total_steps = (cfg['scheduler_total_epochs'] or cfg['epochs']) * updates_per_epoch
    if cfg['scheduler'] == 'onecycle':
        # The scheduler advances once per completed optimizer update, including a
        # shorter final accumulation group, never once per microbatch.
        scheduler_type = torch.optim.lr_scheduler.OneCycleLR
        extra = {}
        if cfg.get('lr_tail_start_epoch') is not None:
            scheduler_type = OneCycleWithCosineTail
            extra = dict(tail_start_step=cfg['lr_tail_start_epoch'] * updates_per_epoch,
                         tail_end_step=cfg['lr_tail_end_epoch'] * updates_per_epoch,
                         tail_final_ratio=cfg['lr_tail_final_ratio'])
        return scheduler_type(
            optimizer, max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=total_steps, pct_start=cfg['warmup_fraction'], **extra)
    if cfg['scheduler'] == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    warmup = max(1, int(total_steps * cfg['warmup_fraction']))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, total_steps - warmup))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


METADATA_KEYS = ['window_mask', 'group_ids', 'series_ids', 'plane_ids', 'positions', 'position_mask']


def model_inputs(batch, device):
    return {key: batch[key].to(device, non_blocking=True) for key in METADATA_KEYS}


def device_images(batch, device, windows=slice(None)):
    """FP32 model input for a window slice; compact uint8 bags expand on the device."""
    images = batch['images'][:, windows].to(device, non_blocking=True)
    if 'intensity_lut' in batch:
        # Imported here so runtimes shipping an older data module still load this trainer.
        from kneemri.coatnet_original_data import expand_intensity
        images = expand_intensity(images, batch['intensity_lut'].to(device, non_blocking=True))
    return images


def configure_checkpointing(model, cfg):
    """Keep activation recomputation only where it is load-bearing.

    Recomputing a stage re-runs its forward, and a training-mode BatchNorm then
    updates its running statistics a second time. Existing runs checkpoint every
    stage, so the BN stages keep that exact trajectory. Stages without BN are
    stateless (dropout replays its RNG state), and at 1/16 and 1/32 resolution
    their stored activations are small, so recomputing them only costs time.
    """
    stages = getattr(model.backbone, 'stages', None)
    if not cfg['gradient_checkpointing'] or cfg['checkpoint_scope'] == 'all' or stages is None:
        return []
    released = []
    for index, stage in enumerate(stages):
        if not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in stage.modules()):
            stage.grad_checkpointing = False
            released.append(index)
    return released


def cpu_state(model):
    """Contiguous CPU copy, so checkpoints keep the same layout without channels_last."""
    return {key: value.detach().to('cpu', memory_format=torch.contiguous_format, copy=True)
            for key, value in model.state_dict().items()}


class WeightEma:
    """Exponential moving averages of the model state, one FP32 shadow per decay.

    Updated after every completed optimizer update n (1-based) with the warm-up decay
    min(decay, (1 + n) / (10 + n)); non-floating buffers are copied. The shadows live
    on the training device and are only read for evaluation and checkpoints.
    """

    def __init__(self, model, decays):
        state = model.state_dict()
        self.decays = {self.tag(d): d for d in decays}
        self.floating = [k for k, v in state.items() if v.is_floating_point()]
        self.dtypes = {k: v.dtype for k, v in state.items()}
        self.shadows = {tag: {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
                              for k, v in state.items()} for tag in self.decays}
        self.updates = 0

    @staticmethod
    def tag(decay):
        return 'ema' + f'{decay:g}'.replace('.', 'p')

    @torch.no_grad()
    def update(self, model, updates):
        state = model.state_dict()
        source = [state[k].float() for k in self.floating]
        for tag, decay in self.decays.items():
            shadow = self.shadows[tag]
            torch._foreach_lerp_([shadow[k] for k in self.floating], source,
                                 1. - min(decay, (1. + updates) / (10. + updates)))
            for k, v in state.items():
                if not v.is_floating_point():
                    shadow[k].copy_(v)
        self.updates = updates

    def cpu_state(self, tag):
        """Same dtypes and contiguous layout as cpu_state(model)."""
        return {k: v.detach().to('cpu', dtype=self.dtypes[k], memory_format=torch.contiguous_format, copy=True)
                for k, v in self.shadows[tag].items()}

    def load_into(self, net, tag):
        net.load_state_dict(self.shadows[tag], strict=True)

    def describe(self, tag):
        return dict(decay=self.decays[tag], optimizer_updates=self.updates,
                    warmup='min(decay, (1 + n) / (10 + n))', update='after every completed optimizer update')

    def state_dict(self):
        return dict(updates=self.updates, shadows={tag: self.cpu_state(tag) for tag in self.shadows})

    def load_state_dict(self, state):
        if set(state['shadows']) != set(self.shadows):
            raise ValueError('resume EMA decays differ')
        for tag, shadow in self.shadows.items():
            for k, v in shadow.items():
                v.copy_(state['shadows'][tag][k])
        self.updates = state['updates']


def amp_context(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()


def select_amp(cfg, device):
    if cfg['amp'] == 'off' or device.type == 'cpu' and cfg['amp'] == 'auto':
        return None
    if cfg['amp'] == 'auto':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if cfg['amp'] == 'fp16' and device.type != 'cuda':
        raise ValueError('fp16 AMP is only enabled for CUDA')
    return torch.bfloat16 if cfg['amp'] == 'bf16' else torch.float16


def select_eval_amp(cfg, device):
    if device.type != 'cuda' or cfg['eval_amp'] == 'off':
        return None
    return torch.bfloat16 if cfg['eval_amp'] == 'bf16' else torch.float16


def _encode_chunks(model, batch, metadata, device, chunk_size):
    features = []
    for start in range(0, batch['images'].shape[1], chunk_size):
        window = slice(start, start + chunk_size)
        features.append(model.encode(device_images(batch, device, window),
                                     metadata['window_mask'][:, window], metadata['group_ids'][:, window]))
    return torch.cat(features, dim=1)


def inference_logits(model, batch, device, dtype, chunk_size):
    metadata = model_inputs(batch, device)
    with torch.no_grad(), amp_context(device, dtype):
        logits = model.head(_encode_chunks(model, batch, metadata, device, chunk_size), **metadata).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f'nonfinite validation logits for {batch["study_ids"]}')
    return logits.cpu()


def budget_logits(model, batch, device, dtype, chunk_size, budgets):
    """Encode one study's window union once, then pool each budget's subset.

    Window features are per-image (BatchNorm is in eval mode), so a budget's
    features are exactly the union features at its positions. When the union is
    the 'all' selection, that budget's encode batches and logits are unchanged.
    """
    if len(batch['study_ids']) != 1:
        raise ValueError('multi-budget evaluation scores one study per batch')
    metadata = model_inputs(batch, device)
    results = {}
    with torch.no_grad(), amp_context(device, dtype):
        features = _encode_chunks(model, batch, metadata, device, chunk_size)
        for budget in budgets:
            index = torch.as_tensor(batch['budget_positions'][0][str(budget)], device=device)
            selected = {key: value[:, index] for key, value in metadata.items()}
            logits = model.head(features[:, index], **selected).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'nonfinite validation logits for {batch["study_ids"]}')
            results[str(budget)] = logits.cpu()
    return results


class CachedBatches:
    """Replays a deterministic evaluation loader from memory after its first pass.

    Worker batches arrive in /dev/shm segments; tensors are cloned into private
    process memory so a long-lived cache never fills the host's shared memory.
    """

    def __init__(self, loader):
        self.loader, self.batches = loader, None

    @staticmethod
    def _private(batch):
        return {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()}

    def __iter__(self):
        if self.batches is not None:
            yield from self.batches
            return
        batches = []
        for batch in self.loader:
            batch = self._private(batch)
            batches.append(batch)
            yield batch
        self.batches, self.loader = batches, None


def prediction_directory(cfg):
    return f'predictions/fold{cfg["val_fold"]}' if cfg['validation_strategy'] == 'grouped_cv' else 'predictions'


def evaluate(model, loader, cfg, device, dtype, output, epoch, budget, role=None, subdirectory=None):
    model.eval()
    uid_list, logits_list, target_list, provenance, label_sources = [], [], [], [], []
    for batch in loader:
        logits_list.append(inference_logits(model, batch, device, dtype, cfg['eval_encode_chunk']))
        target_list.append(batch['targets'])
        uid_list.extend(batch['study_ids'])
        provenance.extend(batch['provenance'])
        label_sources.extend(batch['label_source'])
    return write_evaluation(cfg, output, epoch, budget, role, subdirectory, uid_list,
                            torch.cat(logits_list).numpy(), torch.cat(target_list).numpy(),
                            provenance, label_sources)


def evaluate_budgets(model, loader, cfg, device, dtype, output, epoch, budgets, role=None, subdirectory=None):
    """One pass over a MultiBudgetEvaluationBags loader; one report per budget."""
    model.eval()
    uid_list, target_list, label_sources = [], [], []
    logits_lists = {str(budget): [] for budget in budgets}
    provenance = {str(budget): [] for budget in budgets}
    for batch in loader:
        scored = budget_logits(model, batch, device, dtype, cfg['eval_encode_chunk'], budgets)
        for budget in budgets:
            logits_lists[str(budget)].append(scored[str(budget)])
            provenance[str(budget)].extend(trace[str(budget)] for trace in batch['provenance'])
        target_list.append(batch['targets'])
        uid_list.extend(batch['study_ids'])
        label_sources.extend(batch['label_source'])
    targets = torch.cat(target_list).numpy()
    return {str(budget): write_evaluation(cfg, output, epoch, budget, role, subdirectory, uid_list,
                                          torch.cat(logits_lists[str(budget)]).numpy(), targets,
                                          provenance[str(budget)], label_sources)
            for budget in budgets}


def write_evaluation(cfg, output, epoch, budget, role, subdirectory, uid_list, logits, targets,
                     provenance, label_sources):
    probabilities = torch.from_numpy(logits).sigmoid().numpy()
    aucs = per_target_auc(logits, targets)
    defined = [value for value in aucs.values() if math.isfinite(value)]
    strict_macro = float(np.mean(defined)) if len(defined) == 12 else None
    valid_targets = np.isfinite(targets)
    metric_targets = np.where(valid_targets, (targets > .5).astype(np.float32), np.nan)
    counts = {name: {'positive': int((valid_targets[:, i] & (targets[:, i] > .5)).sum()),
                     'negative': int((valid_targets[:, i] & (targets[:, i] <= .5)).sum()),
                     'valid': int(np.isfinite(targets[:, i]).sum())}
              for i, name in enumerate(LABEL_COLS)}
    report = dict(epoch=epoch, eval_windows=budget, macro_auc=strict_macro,
                  selection_auc=(strict_macro if role is None
                                 and cfg.get('checkpoint_selection', 'best') == 'best' else None),
                  macro_auc_defined_classes=float(np.mean(defined)) if defined else None,
                  defined_classes=len(defined), per_class_auc=aucs, target_counts=counts,
                  n_studies=len(uid_list), validation_role=role or cfg['validation_role'],
                  validation_strategy=cfg['validation_strategy'],
                  val_fold=cfg['val_fold'] if cfg['validation_strategy'] == 'grouped_cv' and role is None else None,
                  metric_target_rule='target_probability > 0.5', label_sources=sorted(set(label_sources)))
    path = output / (subdirectory or prediction_directory(cfg))
    path.mkdir(parents=True, exist_ok=True)
    stem = f'ep{epoch:03d}_k{budget}'
    np.savez_compressed(path / f'{stem}.npz', study_uids=np.asarray(uid_list),
                        logits=logits, probabilities=probabilities, targets=targets, metric_targets=metric_targets,
                        label_names=np.asarray(LABEL_COLS),
                        label_source=np.asarray(label_sources))
    frame = pd.DataFrame({'StudyInstanceUID': uid_list, 'label_source': label_sources})
    for i, name in enumerate(LABEL_COLS):
        frame[f'logit::{name}'] = logits[:, i]
        frame[f'probability::{name}'] = probabilities[:, i]
        frame[f'target::{name}'] = targets[:, i]
        frame[f'metric_target::{name}'] = metric_targets[:, i]
    frame.to_csv(path / f'{stem}.csv', index=False)
    write_json(path / f'{stem}.json', report)
    # Deterministic validation selections do not change across epochs. Bind each
    # prediction to this single immutable selection receipt by budget.
    provenance_path = path / f'windows_k{budget}.json'
    if not provenance_path.exists():
        write_json(provenance_path, {'study_uids': uid_list, 'windows': provenance})
    print(json.dumps(json_safe({'event': 'evaluation', **report})), flush=True)
    return report


def source_receipt(output, cfg, cohort, synthetic=False):
    source_paths = ['scripts/train_coatnet_ablation.py', 'src/kneemri/coatnet_model.py',
                    'src/kneemri/coatnet_data.py', 'src/kneemri/coatnet_training_data.py',
                    'src/kneemri/coatnet_original_data.py', 'src/kneemri/coatnet_public224_data.py',
                    'src/kneemri/coatnet_published_raptor_data.py',
                    'src/kneemri/raptor_training.py', 'src/kneemri/coatnet_cohort.py',
                    'src/kneemri/raptor_reference.py', 'src/kneemri/labels.py',
                    'src/kneemri/metrics.py', 'src/kneemri/cache.py']
    sources = {}
    for name in source_paths:
        path = ROOT / name
        sources[name] = hash_file(path)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
    dependencies = {}
    for name in ['torch', 'timm', 'numpy', 'pandas', 'PyYAML', 'safetensors', 'pydicom']:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    provenance = Path(cfg.get('provenance', cfg.get('corpus', '') + '_provenance'))
    receipt = dict(schema='coatnet-ablation-run-v3', sources=sources, dependencies=dependencies,
                   python=sys.version, git_head=git.stdout.strip() if git.returncode == 0 else None,
                   label_order=LABEL_COLS, normalization=cfg['normalization'], synthetic=synthetic,
                   cuda_version=torch.version.cuda,
                   initial_weights_sha256=cohort.get('initial_weights_sha256'),
                   validation_strategy=cfg['validation_strategy'],
                   folds_sha256=cohort.get('folds_sha256'), val_fold=cohort.get('val_fold'),
                   refit_parent_checkpoint_sha256=(cohort.get('refit_parent') or {}).get(
                       'parent_checkpoint_sha256'))
    if cfg['ema_decays']:
        receipt['weight_ema'] = dict(decays=cfg['ema_decays'], warmup='min(decay, (1 + n) / (10 + n))',
                                     update='after every completed optimizer update', precision='fp32',
                                     checkpoints='final_<tag>.pt; final.pt stays the raw model')
    if (provenance / 'manifest.json').exists():
        receipt['provenance_manifest_sha256'] = hash_file(provenance / 'manifest.json')
        receipt['provenance_arrays_sha256'] = {p.name: hash_file(p) for p in sorted(provenance.glob('*.npy'))}
    corpus_manifest = Path(cfg.get('corpus', '')) / 'manifest.json'
    if corpus_manifest.exists():
        receipt['corpus_manifest_sha256'] = hash_file(corpus_manifest)
    return receipt


class SyntheticBags(Dataset):
    def __init__(self, ids, labels, cfg, train):
        self.ids, self.labels, self.cfg, self.train = ids, labels, cfg, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        k = self.cfg['train_windows'] if self.train else self.cfg['eval_windows']
        if isinstance(k, list):
            k = k[index % len(k)]
        k = 6 if k == 'all' else min(k, 6)
        count = max(1, k - index % 2)
        generator = torch.Generator().manual_seed(17 + int(uid.split('_')[-1]))
        return dict(images=torch.rand(count, 3, 16, 16, generator=generator),
                    window_mask=torch.ones(count, dtype=torch.bool),
                    group_ids=torch.arange(count) % 5, series_ids=torch.arange(count) // 2,
                    plane_ids=torch.arange(count) % 3,
                    positions=torch.arange(count).float() * 3 - 6,
                    position_mask=torch.ones(count, dtype=torch.bool),
                    targets=torch.from_numpy(self.labels[uid]), study_id=uid,
                    label_source='synthetic',
                    provenance={'study_uid': uid, 'synthetic': True, 'window_count': count})


class SyntheticBackbone(torch.nn.Module):
    num_features = 16

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1),
                                           torch.nn.BatchNorm2d(16), torch.nn.GELU())

    def forward_features(self, x):
        return self.layers(x)

    def forward_head(self, x):
        return x.mean((-1, -2))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def make_loader(dataset, cfg, device, train=False, generator=None, pin_memory=True):
    from kneemri.coatnet_training_data import coatnet_collate
    workers = cfg['workers'] if train else cfg['eval_workers']
    kwargs = dict(batch_size=cfg['batch_size'] if train else 1, shuffle=train,
                  num_workers=workers, pin_memory=pin_memory and device.type == 'cuda',
                  worker_init_fn=seed_worker, collate_fn=coatnet_collate, generator=generator,
                  drop_last=cfg['drop_last_train'] if train else False)
    if workers:
        # Batches go to workers round-robin in sampler order regardless of depth,
        # so a deeper training queue absorbs storage jitter without changing draws.
        kwargs['prefetch_factor'] = 2 if train else 1
    return DataLoader(dataset, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--devices',
                        help='comma-separated CUDA device IDs for optional DataParallel training')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--evaluate-checkpoint', type=Path,
                        help='evaluate one fixed checkpoint at all budgets without training')
    parser.add_argument('--smoke', action='store_true', help='two optimizer updates on a real subset')
    parser.add_argument('--synthetic-smoke', action='store_true', help='tiny CPU-capable model and synthetic bags')
    args = parser.parse_args()
    cfg = normalize_config(yaml.safe_load(args.config.read_text()))
    declared_cfg = dict(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((args.out / 'config.json').exists() or list(args.out.glob('*.pt'))):
        raise ValueError('output already contains a run; use --resume or a new output directory')
    smoke = args.smoke or args.synthetic_smoke
    if args.evaluate_checkpoint and (args.resume or smoke):
        raise ValueError('--evaluate-checkpoint cannot be combined with resume or smoke')
    if smoke:
        cfg.update(epochs=1, workers=0, eval_workers=0, eval_sweep_every=1)
    requested_devices = None
    if args.devices:
        if not torch.cuda.is_available():
            raise RuntimeError('--devices requires CUDA')
        requested_devices = [int(value.strip()) for value in args.devices.split(',') if value.strip()]
        if not requested_devices or len(set(requested_devices)) != len(requested_devices):
            raise ValueError('--devices must contain distinct CUDA device IDs')
        if min(requested_devices) < 0 or max(requested_devices) >= torch.cuda.device_count():
            raise ValueError('--devices contains an unavailable CUDA device')
        device = torch.device(f'cuda:{requested_devices[0]}')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but unavailable')
    torch.set_num_threads(cfg['num_threads'])
    seed_all(cfg['seed'])
    dtype = select_amp(cfg, device)
    eval_dtype = select_eval_amp(cfg, device)
    if args.synthetic_smoke:
        train_count = max(16, 2 * cfg['batch_size'] * cfg['gradient_accumulation'])
        monitor_count = (8 if cfg['validation_strategy'] in {'grouped_cv', 'full_refit'}
                         and cfg['monitor_gold'] else 0)
        all_ids = [f'synthetic_{i}' for i in range(train_count + 8 + monitor_count)]
        train_ids, val_ids = all_ids[:train_count], all_ids[train_count:train_count + 8]
        monitor_ids = all_ids[train_count + 8:] if cfg['monitor_gold'] else []
        labels = {uid: ((np.arange(12) + i) % 2).astype(np.float32) for i, uid in enumerate(all_ids)}
        if cfg['validation_strategy'] == 'grouped_cv':
            for uid in train_ids + val_ids:
                labels[uid] = labels[uid] * .8 + .1
        elif cfg['validation_strategy'] == 'full_refit':
            train_ids, val_ids = train_ids + val_ids, []
        cohort = dict(trained_study_uids=train_ids, validation_study_uids=val_ids,
                      excluded_report_group_peers=[], synthetic=True,
                      validation_strategy=cfg['validation_strategy'], val_fold=cfg['val_fold'],
                      gold_monitor_study_uids=monitor_ids, gold_study_uids=monitor_ids,
                      selection_label_source='synthetic', trained_groups=[], validation_groups=[],
                      protected_validation_study_uids=monitor_ids,
                      gold_monitor_role='synthetic_gold_monitor_after_fold_selection_only')
        if cfg['validation_strategy'] == 'full_refit':
            cohort['refit_parent'] = dict(
                parent_checkpoint='synthetic-smoke', parent_checkpoint_sha256=None,
                parent_selected_epoch=declared_cfg['epochs'], parent_selection_auc=None,
                parent_validation_role='synthetic_grouped_cv_selection',
                parent_validation_strategy='grouped_cv', parent_val_fold=cfg['val_fold'],
                parent_scheduler_total_epochs=cfg['scheduler_total_epochs'],
                fresh_initialization_required=True, parent_weights_used_for_initialization=False)
    else:
        all_ids, train_ids, val_ids, labels, cohort = load_cohort(cfg)
        if cfg['validation_strategy'] == 'full_refit':
            cohort['refit_parent'] = validate_refit_parent(declared_cfg, cohort)
        if smoke:
            train_ids = train_ids[:max(16, 2 * cfg['batch_size'] * cfg['gradient_accumulation'])]
            if val_ids:
                val_ids = smoke_coverage_subset(val_ids, labels)
            monitor_ids = (smoke_coverage_subset(cohort['gold_monitor_study_uids'], labels)
                           if cohort['gold_monitor_study_uids'] else [])
            cohort.update(trained_study_uids=train_ids, validation_study_uids=val_ids, smoke=True)
            cohort['gold_monitor_study_uids'] = monitor_ids
        if args.evaluate_checkpoint:
            cohort['evaluation_checkpoint_sha256'] = hash_file(args.evaluate_checkpoint)
        else:
            cohort['initial_weights_sha256'] = hash_file(cfg['initial_weights'])
            if cfg.get('backbone_init_checkpoint'):
                cohort['backbone_init_checkpoint_sha256'] = hash_file(cfg['backbone_init_checkpoint'])
    cohort['model_origin'] = ('evaluation-checkpoint' if args.evaluate_checkpoint else
                              'synthetic-smoke' if args.synthetic_smoke else
                              'fresh-external-backbone-full-refit' if cfg.get('backbone_init_checkpoint')
                              else 'fresh-generic-local-ImageNet-only-full-refit'
                              if cfg['validation_strategy'] == 'full_refit' else
                              'external-backbone' if cfg.get('backbone_init_checkpoint')
                              else 'generic-local-ImageNet-only')
    fixed_final = cfg['checkpoint_selection'] == 'fixed_final'
    if fixed_final:
        cohort.update(selection_metric='none_prespecified_final_epoch',
                      checkpoint_selection='fixed_final', fixed_epoch=declared_cfg['epochs'],
                      gold_monitor_role='gold58_monitor_after_prespecified_final_epoch_only')
    weights = supervision_weights(cfg, train_ids, cohort)
    receipt = source_receipt(args.out, cfg, cohort, args.synthetic_smoke)
    if args.resume:
        previous = json.loads((args.out / 'receipt.json').read_text())
        if previous != receipt:
            raise ValueError('source, dependency or provenance changed since the saved run')
        if (json.loads((args.out / 'config.json').read_text()) != cfg
                or json.loads((args.out / 'cohort.json').read_text()) != cohort):
            raise ValueError('resume config, labels or cohort mismatch')
    write_json(args.out / 'config.json', cfg)
    write_json(args.out / 'cohort.json', cohort)
    write_json(args.out / 'receipt.json', receipt)
    print(json.dumps({'event': 'run_start', 'train': len(train_ids), 'validation': len(val_ids),
                      'validation_strategy': cfg['validation_strategy'],
                      'val_fold': cohort.get('val_fold'), 'gold_monitor': len(cohort['gold_monitor_study_uids']),
                      'excluded_peers': len(cohort['excluded_report_group_peers']),
                      'config': cfg, 'smoke': smoke, 'device': str(device),
                      'training_devices': requested_devices or
                      [device.index if device.type == 'cuda' else 'cpu']}), flush=True)
    multi_budget_dataset = None
    if args.synthetic_smoke:
        train_ds = SyntheticBags(train_ids, labels, cfg, True)
        def validation_dataset(budget, ids=None, label_source=None):
            return SyntheticBags(val_ids if ids is None else ids, labels, dict(cfg, eval_windows=budget), False)
        model = CoAtNetAblationModel(SyntheticBackbone(), cfg)
    else:
        corpus_format = cfg.get('corpus_format', 'raptor64')
        if corpus_format == 'original_neighbors':
            from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags as StudyDataset
        elif corpus_format == 'public224_triplets':
            from kneemri.coatnet_public224_data import CoAtNetPublic224Bags as StudyDataset
        elif corpus_format == 'published_raptor44':
            from kneemri.coatnet_published_raptor_data import CoAtNetPublishedRaptorBags as StudyDataset
        else:
            from kneemri.coatnet_training_data import CoAtNetStudyBags as StudyDataset
        compact = corpus_format == 'original_neighbors' and cfg['uint8_transfer']
        extra = dict(uint8_output=True) if compact else {}
        TrainDataset = StudyDataset
        if cfg.get('train_evidence_windows'):
            if corpus_format != 'original_neighbors':
                raise ValueError('evidence sampling requires the original-neighbour corpus')
            from kneemri.resnet_evidence_data import EvidenceNeighborBags as TrainDataset
        train_ds = TrainDataset(cfg, all_ids, train_ids, labels, train=True, **extra)
        def validation_dataset(budget, ids=None, label_source=None):
            evaluation_cfg = dict(cfg, eval_windows=budget,
                                  eval_label_source=label_source or cfg['eval_label_source'])
            return StudyDataset(evaluation_cfg, all_ids, val_ids if ids is None else ids, labels,
                                train=False, **extra)
        if corpus_format == 'original_neighbors':
            from kneemri.coatnet_original_data import MultiBudgetEvaluationBags
            def multi_budget_dataset(budgets, ids=None, label_source=None):
                evaluation_cfg = dict(cfg, eval_label_source=label_source or cfg['eval_label_source'])
                return MultiBudgetEvaluationBags(evaluation_cfg, all_ids, val_ids if ids is None else ids,
                                                 labels, budgets, uint8_output=compact)
        model = build_model(cfg, pretrained=not bool(args.evaluate_checkpoint))
    released_stages = configure_checkpointing(model, cfg)
    model = model.to(device)
    if cfg['channels_last'] and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    squeeze_native_warmup(model, cfg, device)
    train_model = model
    if requested_devices and len(requested_devices) > 1:
        train_model = torch.nn.DataParallel(model, device_ids=requested_devices,
                                            output_device=requested_devices[0])

    def evaluate_many(budgets, ids=None, label_source=None, cache=None, epoch=0,
                      dataset_budgets=None, net=None, **report):
        """Score several budgets, sharing one pixel read and backbone pass when possible.

        ``dataset_budgets`` (a superset of ``budgets``) fixes which windows each
        item holds, so one cached dataset serves both plain and sweep epochs.
        ``net`` scores another network (an EMA shadow) instead of the trained model.
        """
        net = model if net is None else net
        budgets = list(dict.fromkeys(budgets))
        if multi_budget_dataset is None:
            return {str(budget): evaluate(net, make_loader(validation_dataset(budget, ids, label_source),
                                                             cfg, device), cfg, device, eval_dtype,
                                          args.out, epoch, budget, **report) for budget in budgets}
        dataset_budgets = list(dict.fromkeys(dataset_budgets or budgets))
        if not {str(b) for b in budgets} <= {str(b) for b in dataset_budgets}:
            raise ValueError('scored budgets must be held by the evaluation dataset')
        key = (tuple(str(b) for b in dataset_budgets), None if ids is None else tuple(ids), label_source)
        loader = cache.get(key) if cache is not None else None
        if loader is None:
            dataset = multi_budget_dataset(dataset_budgets, ids, label_source)
            size = dataset.pixel_bytes()
            cacheable = cache is not None and size <= cfg['eval_cache_gb'] * 2**30
            # Cached batches stay pageable: pinning tens of GB would lock host RAM for the run.
            loader = make_loader(dataset, cfg, device, pin_memory=not cacheable)
            if cacheable:
                loader = cache[key] = CachedBatches(loader)
            print(json.dumps({'event': 'evaluation_loader', 'budgets': list(key[0]), 'cached': cacheable,
                              'pixel_gb': size / 2**30}), flush=True)
        return evaluate_budgets(net, loader, cfg, device, eval_dtype, args.out, epoch, budgets, **report)
    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=True)
        if (checkpoint.get('lab') != LABEL_COLS or checkpoint.get('arch') != cfg['arch']
                or checkpoint.get('res') != cfg['resolution']):
            raise ValueError('evaluation checkpoint label order, architecture or resolution differs')
        previously_trained = set(checkpoint.get('cohort', {}).get('trained_study_uids', []))
        assert_checkpoint_is_held_out(checkpoint, cohort)
        model.load_state_dict(checkpoint['model'], strict=True)
        checkpoint_epoch = checkpoint.get('epoch', -1) + 1
        evaluation_ids = (cohort['gold_monitor_study_uids']
                          if cfg['validation_strategy'] == 'full_refit' else None)
        evaluation_source = 'official_gold' if cfg['validation_strategy'] == 'full_refit' else None
        evaluation_role = ('gold58_independent_fixed_refit_checkpoint_evaluation'
                           if cfg['validation_strategy'] == 'full_refit' else None)
        evaluation_subdirectory = ('gold_monitor/independent_predictions'
                                   if cfg['validation_strategy'] == 'full_refit' else None)
        results = evaluate_many([cfg['eval_windows']] + cfg['eval_budgets'], evaluation_ids,
                                evaluation_source, epoch=checkpoint_epoch, role=evaluation_role,
                                subdirectory=evaluation_subdirectory)
        write_json(args.out / 'evaluation_matrix.json', results)
        write_json(args.out / 'complete.json', {'evaluation_only': True,
                   'checkpoint_sha256': cohort['evaluation_checkpoint_sha256'],
                   'source_training_cohort_available': bool(previously_trained),
                   'epochs_completed': 0, 'checkpoint_epoch': checkpoint_epoch})
        print('EVALUATION COMPLETE', flush=True)
        return
    loader_generator = torch.Generator()
    loader = make_loader(train_ds, cfg, device, train=True, generator=loader_generator)
    train_steps = min(len(loader), 2 * cfg['gradient_accumulation']) if smoke else len(loader)
    updates_per_epoch = math.ceil(train_steps / cfg['gradient_accumulation'])
    backbone = list(model.backbone.parameters())
    spatial = ([p for name, p in model.named_parameters() if name.startswith('spatial_pool.')]
               if cfg['spatial_lr'] is not None else [])
    spatial_ids = {id(p) for p in spatial}
    head = [p for name, p in model.named_parameters()
            if not name.startswith('backbone.') and id(p) not in spatial_ids]
    parameter_groups = [{'params': backbone, 'lr': cfg['backbone_lr']}]
    if spatial:
        parameter_groups.append({'params': spatial, 'lr': cfg['spatial_lr']})
    parameter_groups.append({'params': head, 'lr': cfg['head_lr']})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=cfg['weight_decay'])
    scheduler = create_scheduler(optimizer, cfg, updates_per_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
    ema = WeightEma(model, cfg['ema_decays']) if cfg['ema_decays'] else None
    ema_model = copy.deepcopy(model).requires_grad_(False) if ema else None
    pw = positive_weights(np.stack([labels[uid] for uid in train_ids]),
                          np.stack([weights[uid] for uid in train_ids]), cfg)
    pos_weight = torch.from_numpy(pw).to(device)
    write_json(args.out / 'supervision.json', {'label_order': LABEL_COLS, 'positive_weights': pw,
               'weighted_valid_totals': np.stack(list(weights.values())).sum(0),
               'confidence_csv': cfg.get('confidence_csv'), 'valid_mask_csv': cfg.get('valid_mask_csv')})
    full_refit = cfg['validation_strategy'] == 'full_refit'
    start, best, history, updates = 0, None if full_refit or fixed_final else -math.inf, [], 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint['config'] != cfg or checkpoint['cohort'] != cohort or checkpoint['receipt'] != receipt:
            raise ValueError('resume config, labels, cohort or source mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start = checkpoint['epoch'] + 1
        best, history, updates = checkpoint['best_selection_auc'], checkpoint['history'], checkpoint['optimizer_updates']
        if ema:
            ema.load_state_dict(checkpoint['ema'])
        del checkpoint
    print(json.dumps({'event': 'model_ready', 'backbone_parameters': sum(p.numel() for p in backbone),
                      'spatial_parameters': sum(p.numel() for p in spatial),
                      'head_parameters': sum(p.numel() for p in head), 'amp': str(dtype),
                      'eval_amp': str(eval_dtype),
                      'steps_per_epoch': train_steps, 'updates_per_epoch': updates_per_epoch,
                      'channels_last': cfg['channels_last'] and device.type == 'cuda',
                      'uint8_transfer': bool(getattr(train_ds, 'uint8_output', False)),
                      'recompute_free_stages': released_stages}), flush=True)
    evaluation_cache = {}
    wall = time.time()
    for epoch in range(start, cfg['epochs']):
        seed_all(cfg['seed'] + epoch)
        loader_generator.manual_seed(cfg['seed'] + epoch)
        train_model.train()
        if cfg['freeze_bn']:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen, epoch_provenance = 0., 0, []
        rank_loss_sum, rank_pair_sum, rank_groups = 0., 0, 0
        for group_start, batches in accumulation_groups(loader, train_steps, cfg['gradient_accumulation']):
            denominators = group_supervision(batches, weights).to(device)
            rank_records = []
            for local_step, batch in enumerate(batches):
                step = group_start + local_step
                x = device_images(batch, device)
                target = batch['targets'].to(device, non_blocking=True)
                label_weight = batch['label_weights'].to(device, non_blocking=True)
                metadata = model_inputs(batch, device)
                with amp_context(device, dtype):
                    if cfg['rank_weight']:
                        features, feature_maps = model.encode(
                            x, metadata['window_mask'], metadata['group_ids'],
                            return_backbone_features=True)
                        logits = model.head(features, **metadata)
                    else:
                        logits = train_model(x, **metadata)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f'nonfinite logits for {batch["study_ids"]}')
                    loss = weighted_loss(logits, target, label_weight, pos_weight, cfg, denominators)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss for {batch["study_ids"]}')
                if cfg['rank_weight']:
                    rank_records.append(dict(feature_maps=feature_maps.detach(),
                                             metadata={key: value.detach() for key, value in metadata.items()},
                                             targets=target.detach()))
                scaler.scale(loss).backward()
                local_loss = weighted_loss(logits.detach(), target, label_weight, pos_weight, cfg)
                total_loss += float(local_loss) * len(target)
                seen += len(target)
                epoch_provenance.extend(batch['provenance'])
                boundary = local_step + 1 == len(batches)
                norm = None
                if boundary:
                    if cfg['rank_weight']:
                        with amp_context(device, dtype):
                            rank_logits, rank_targets = replay_rank_group(model, rank_records)
                            rank_loss, rank_pairs = pairwise_rank_loss(
                                rank_logits, rank_targets, cfg['rank_positive_threshold'],
                                cfg['rank_negative_threshold'])
                        if not torch.isfinite(rank_loss):
                            raise FloatingPointError('nonfinite pairwise rank loss')
                        scaler.scale(cfg['rank_weight'] * rank_loss).backward()
                        rank_loss_sum += float(rank_loss.detach())
                        rank_pair_sum += int(rank_pairs.detach())
                        rank_groups += 1
                    scaler.unscale_(optimizer)
                    norm = squeeze_clip_gradients(model, cfg['gradient_clip'] or math.inf)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('nonfinite gradient; optimizer update aborted')
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                        updates += 1
                        if ema:
                            ema.update(model, updates)
                    optimizer.zero_grad(set_to_none=True)
                if step == 0 or boundary and ((step + 1) % cfg['log_every'] == 0 or step + 1 == train_steps):
                    print(json.dumps({'event': 'train_step', 'epoch': epoch + 1, 'step': step + 1,
                                      'steps': train_steps, 'train_loss': total_loss / seen,
                                      'optimizer_updates': updates, 'lr': [g['lr'] for g in optimizer.param_groups],
                                      'valid_windows': int(metadata['window_mask'].sum()),
                                      'rank_loss': (rank_loss_sum / rank_groups if rank_groups else None),
                                      'rank_pairs': rank_pair_sum,
                                      'gradient_norm': float(norm) if norm is not None else None,
                                      'elapsed_s': time.time() - wall,
                                      'peak_cuda_memory_mb': torch.cuda.max_memory_allocated(device) / 2**20
                                          if device.type == 'cuda' else 0}), flush=True)
        write_json(args.out / f'train_windows_ep{epoch + 1:03d}.json', epoch_provenance)
        results, ema_results = {}, {}
        if full_refit or ((epoch + 1) % cfg['evaluation_every'] != 0
                          and epoch + 1 != cfg['epochs']):
            score, primary = None, None
        else:
            sweep = list(dict.fromkeys([cfg['eval_windows']] + cfg['eval_budgets']))
            budgets = [cfg['eval_windows']]
            if (epoch + 1) % cfg['eval_sweep_every'] == 0 or epoch + 1 == cfg['epochs']:
                budgets = sweep
            results = evaluate_many(budgets, cache=evaluation_cache, epoch=epoch + 1, dataset_budgets=sweep)
            primary = results[str(cfg['eval_windows'])]
            score = None if fixed_final else primary['macro_auc']
            for tag in ema.decays if ema else []:
                ema.load_into(ema_model, tag)
                ema_results[tag] = evaluate_many(
                    [cfg['eval_windows']], cache=evaluation_cache, epoch=epoch + 1, dataset_budgets=sweep,
                    net=ema_model, role=f'{cfg["validation_role"]}__{tag}',
                    subdirectory=f'{prediction_directory(cfg)}/{tag}')[str(cfg['eval_windows'])]
        row = dict(epoch=epoch + 1, train_loss=total_loss / seen, selection_auc=score,
                   diagnostic_auc=primary['macro_auc'] if primary else None,
                   train_rank_loss=(rank_loss_sum / rank_groups if rank_groups else None),
                   train_rank_pairs=rank_pair_sum, evaluations=results,
                   elapsed_s=time.time() - wall, optimizer_updates=updates)
        if ema:
            row['ema_diagnostic_auc'] = {tag: report['macro_auc'] for tag, report in ema_results.items()}
        history.append(row)
        common = dict(model=cpu_state(model),
                      arch=cfg['arch'], res=cfg['resolution'], lab=LABEL_COLS, epoch=epoch,
                      src=cohort['model_origin'], selection_auc=score,
                      aucs=json_safe(primary['per_class_auc']) if primary else None,
                      config=cfg, cohort=cohort, receipt=receipt)
        if (full_refit or fixed_final) and epoch + 1 == cfg['epochs']:
            save_atomic(common, args.out / 'final.pt')
            if not full_refit:
                shutil.copy2(args.out / prediction_directory(cfg) / f'ep{epoch + 1:03d}_k{cfg["eval_windows"]}.npz',
                             args.out / 'final_validation.npz')
                write_json(args.out / 'final_evaluation_matrix.json', results)
            for tag in ema.decays if ema else []:
                report = ema_results.get(tag)
                save_atomic(dict(common, model=ema.cpu_state(tag), ema=ema.describe(tag),
                                 aucs=json_safe(report['per_class_auc']) if report else None),
                            args.out / f'final_{tag}.pt')
                if report:
                    shutil.copy2(args.out / prediction_directory(cfg) / tag
                                 / f'ep{epoch + 1:03d}_k{cfg["eval_windows"]}.npz',
                                 args.out / f'final_validation_{tag}.npz')
        elif score is not None and score > best:
            best = score
            save_atomic(common, args.out / 'best.pt')
            shutil.copy2(args.out / prediction_directory(cfg) / f'ep{epoch + 1:03d}_k{cfg["eval_windows"]}.npz',
                         args.out / 'best_validation.npz')
        if cfg['checkpoint_every'] and (epoch + 1) % cfg['checkpoint_every'] == 0:
            save_atomic(common, args.out / f'ep{epoch + 1:03d}.pt')
        save_atomic(dict(common, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), best_selection_auc=best, history=history,
                         optimizer_updates=updates, **({'ema': ema.state_dict()} if ema else {})),
                    args.out / 'last.pt')
        write_json(args.out / 'history.json', dict(validation_role=cfg['validation_role'],
                   validation_strategy=cfg['validation_strategy'], val_fold=cohort.get('val_fold'),
                   best_selection_auc=best, epochs_completed=epoch + 1, epochs_planned=cfg['epochs'], history=history))
        print(json.dumps(json_safe({'event': 'epoch_complete', 'epoch': epoch + 1,
                                    'epochs': cfg['epochs'], 'selection_auc': score,
                                    'best_selection_auc': best, 'elapsed_s': time.time() - wall})), flush=True)
    if full_refit:
        final_path = args.out / 'final.pt'
        if not final_path.exists():
            raise RuntimeError('full refit ended without its prespecified final checkpoint')
        gold_report = evaluate_many(
            [cfg['eval_windows']], cohort['gold_monitor_study_uids'], 'official_gold', epoch=cfg['epochs'],
            role=cohort['gold_monitor_role'], subdirectory='gold_monitor/predictions')[str(cfg['eval_windows'])]
        write_json(args.out / 'gold_monitor' / 'report.json', dict(
            report=gold_report, selected_on=cohort['refit_parent']['parent_validation_role'],
            parent_checkpoint_sha256=cohort['refit_parent']['parent_checkpoint_sha256'],
            parent_selected_epoch=cohort['refit_parent']['parent_selected_epoch'],
            checkpoint_sha256=hash_file(final_path), used_for_checkpoint_selection=False,
            refit_weights_initialized_from_parent=False))
    elif fixed_final:
        final_path = args.out / 'final.pt'
        if not final_path.exists():
            raise RuntimeError('fixed-epoch training ended without its final checkpoint')
        evaluation_cache.clear()
        if cohort['gold_monitor_study_uids']:
            gold_report = evaluate_many(
                [cfg['eval_windows']], cohort['gold_monitor_study_uids'], 'official_gold',
                epoch=cfg['epochs'], role=cohort['gold_monitor_role'],
                subdirectory='gold_monitor/predictions')[str(cfg['eval_windows'])]
            write_json(args.out / 'gold_monitor' / 'report.json', dict(
                report=gold_report, selected_on='prespecified_final_epoch',
                checkpoint_sha256=hash_file(final_path), used_for_checkpoint_selection=False))
    elif (args.out / 'best.pt').exists():
        checkpoint = torch.load(args.out / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        best_epoch = checkpoint['epoch'] + 1
        del checkpoint
        sweep = list(dict.fromkeys([cfg['eval_windows']] + cfg['eval_budgets']))
        best_evaluations = {}
        missing = []
        for budget in sweep:
            report_path = args.out / prediction_directory(cfg) / f'ep{best_epoch:03d}_k{budget}.json'
            if report_path.exists():
                best_evaluations[str(budget)] = json.loads(report_path.read_text())
            else:
                missing.append(budget)
        if missing:
            best_evaluations.update(evaluate_many(missing, cache=evaluation_cache, epoch=best_epoch,
                                                  dataset_budgets=sweep))
        best_evaluations = {str(budget): best_evaluations[str(budget)] for budget in sweep}
        evaluation_cache.clear()
        write_json(args.out / 'best_evaluation_matrix.json', best_evaluations)
        gold_threshold = cfg['gold_monitor_min_selection_auc']
        monitor_gold = (bool(cohort['gold_monitor_study_uids'])
                        and (gold_threshold is None or best > gold_threshold))
        if monitor_gold:
            gold_report = evaluate_many(
                [cfg['eval_windows']], cohort['gold_monitor_study_uids'], 'official_gold', epoch=best_epoch,
                role=cohort['gold_monitor_role'],
                subdirectory='gold_monitor/predictions')[str(cfg['eval_windows'])]
            write_json(args.out / 'gold_monitor' / 'report.json', dict(
                report=gold_report, selected_on=cfg['validation_role'],
                best_selection_auc=best, checkpoint_sha256=hash_file(args.out / 'best.pt'),
                used_for_checkpoint_selection=False))
        elif cohort['gold_monitor_study_uids']:
            (args.out / 'gold_monitor').mkdir(parents=True, exist_ok=True)
            write_json(args.out / 'gold_monitor' / 'skipped.json', dict(
                reason='fold_selection_auc_did_not_exceed_threshold',
                best_selection_auc=best, required_strictly_greater_than=gold_threshold,
                checkpoint_sha256=hash_file(args.out / 'best.pt'),
                used_for_checkpoint_selection=False))
    write_json(args.out / 'complete.json', dict(epochs_completed=cfg['epochs'], best_selection_auc=best,
               checkpoint_selection=cfg['checkpoint_selection'],
               validation_strategy=cfg['validation_strategy'], val_fold=cohort.get('val_fold'),
               elapsed_s=time.time() - wall, synthetic=args.synthetic_smoke, smoke=smoke,
               last_checkpoint_sha256=hash_file(args.out / 'last.pt'),
               best_checkpoint_sha256=hash_file(args.out / 'best.pt') if (args.out / 'best.pt').exists() else None,
               fixed_final_checkpoint_sha256=(hash_file(args.out / 'final.pt')
                                              if (args.out / 'final.pt').exists() else None),
               refit_parent=cohort.get('refit_parent')))
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
