"""Astra v9 full-strength mass-preserved BCE, after lizhihong's C0 study.

The original soft labels and their positive weights are unchanged. Statistics
are fitted exclusively on the actual training cohort, never on Gold58.
"""
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F


CONFIG_KEYS = {'soft_supervision', 'mass_strength', 'mass_normalizer', 'soft_manifest'}


def mass_statistics(targets, trainer, cfg):
    y = np.asarray(targets, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != 12 or not np.isfinite(y).all():
        raise ValueError('mass statistics require complete twelve-class training probabilities')
    if np.any((y < 0) | (y > 1)):
        raise ValueError('targets outside [0, 1]')
    pos = trainer.positive_weights(y, np.ones_like(y), cfg)
    normalizer = 1. / (1. + (pos - 1.) * y).astype(np.float64).mean(0)
    return pos, normalizer


def weighted_loss(logits, targets, weights, pos_weight, cfg, group_denominators=None):
    x, y, w = logits.float(), targets.float(), weights.float()
    valid = torch.isfinite(y) & torch.isfinite(w) & (w > 0)
    y = torch.where(valid, y, torch.zeros_like(y))
    w = torch.where(valid, w, torch.zeros_like(w))
    normalizer = torch.as_tensor(cfg['mass_normalizer'], device=x.device, dtype=torch.float32)
    multiplier = (1. + (pos_weight - 1.) * y).reciprocal() / normalizer
    values = F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight, reduction='none')
    values = torch.where(valid, values, torch.zeros_like(values)) * w * multiplier
    denominator = w.sum(0) if group_denominators is None else group_denominators
    return values.sum() / denominator.sum().clamp_min(torch.finfo(torch.float32).eps)


def install(trainer):
    from kneemri.gold_cohort import install as install_gold
    install_gold(trainer)
    if getattr(trainer, '_resnet_fullstrength_installed', False):
        return
    trainer._resnet_fullstrength_installed = True
    trainer.DATA_KEYS.update(CONFIG_KEYS)
    original_normalize = trainer.normalize_config
    original_cohort = trainer.load_cohort
    original_receipt = trainer.source_receipt

    def normalize(config):
        cfg = original_normalize(config)
        if (cfg.get('soft_supervision') != 'mass_preserved' or cfg.get('mass_strength') != 1.
                or cfg['loss'] != 'bce' or cfg['loss_normalization'] != 'global'
                or cfg.get('confidence_csv') or cfg.get('valid_mask_csv') or cfg['label_mixing']):
            raise ValueError('requires full-strength mass-preserved BCE with unchanged Astra labels')
        norm = np.asarray(cfg['mass_normalizer'], dtype=float)
        if norm.shape != (12,) or not np.isfinite(norm).all() or (norm <= 0).any():
            raise ValueError('requires twelve finite positive training mass normalizers')
        return cfg

    def load_cohort(cfg):
        result = original_cohort(cfg)
        _, train_ids, _, labels, _ = result
        _, expected = mass_statistics(np.stack([labels[u] for u in train_ids]), trainer, cfg)
        np.testing.assert_allclose(cfg['mass_normalizer'], expected, rtol=0, atol=1e-12)
        return result

    def receipt(output, cfg, cohort, synthetic=False):
        result = original_receipt(output, cfg, cohort, synthetic)
        for name in ('resnet_fullstrength.py', 'train_resnet_fullstrength.py'):
            source = Path(__file__).with_name(name)
            target = output / 'source/scripts' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            result['sources']['scripts/' + name] = trainer.hash_file(source)
        result['soft_supervision'] = dict(
            mechanism='mass_preserved', strength=1.,
            manifest_sha256=trainer.hash_file(cfg['soft_manifest']),
            source='lizhihong/soft_weight_c0_followup_20261003/soft_objective.py',
            fitted_only_on='actual 4400 training cohort including official Gold58, excluding report peers',
            original_astra_probabilities_unchanged=True)
        return result

    trainer.normalize_config = normalize
    trainer.load_cohort = load_cohort
    trainer.weighted_loss = weighted_loss
    trainer.source_receipt = receipt
