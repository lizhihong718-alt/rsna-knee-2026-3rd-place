"""Optionally train on the 58 expert-labeled Gold studies as well (final-model variant).

Gold58 has never predicted the leaderboard at the +-.004 level, so for a final
leaderboard model its expert labels are more useful as supervision than as a
selection ruler. With ``train_include_gold`` the Gold studies (official binary
labels, already in the cohort label map) join the training set; the report-group
peers stay excluded. Gold58 is still scored every epoch, but in-sample only.
"""

GOLD_KEYS = {'train_include_gold'}


def install(trainer):
    """Must run before the full-strength install so its constants check sees Gold."""
    if getattr(trainer, '_gold_cohort_installed', False):
        return
    trainer._gold_cohort_installed = True
    trainer.DATA_KEYS.update(GOLD_KEYS)
    original = trainer.load_cohort

    def load_cohort(cfg):
        all_ids, train_ids, val_ids, labels, cohort = original(cfg)
        if not cfg.get('train_include_gold'):
            return all_ids, train_ids, val_ids, labels, cohort
        if cfg.get('validation_strategy') != 'gold' or cfg.get('checkpoint_selection') != 'fixed_final':
            raise ValueError('Gold-in-training requires the fixed-final Gold-monitor protocol')
        gold = list(cohort['gold_study_uids'])
        if set(gold) & set(train_ids) or set(cohort['excluded_report_group_peers']) & set(train_ids):
            raise ValueError('Gold or report peers already in training')
        if any(sorted(set(labels[u].tolist())) not in ([0.], [1.], [0., 1.]) for u in gold):
            raise ValueError('Gold supervision must be the official binary labels')
        train_ids = list(train_ids) + gold
        cohort = dict(cohort, trained_study_uids=train_ids, gold_in_training=True,
                      gold_monitor_role='in_sample_only_not_a_validation',
                      protected_validation_study_uids=list(cohort['excluded_report_group_peers']))
        return all_ids, train_ids, val_ids, labels, cohort

    trainer.load_cohort = load_cohort
