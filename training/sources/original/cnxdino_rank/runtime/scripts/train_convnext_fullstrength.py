#!/usr/bin/env python3
"""Train one fresh ConvNeXt with PF0.959 full-strength supervision.

Same trainer, objective and data hooks as the ResNet PF route (wave2 runtime_e):
mass-preserved BCE (strength 1), optional laterality canonicalization and optional
Gold58-in-training. Only the backbone differs: the shared CoAtNet/ConvNeXt builder
in kneemri.coatnet_model replaces the ResNet factory.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_coatnet_ablation as trainer
from kneemri.gold_cohort import install as install_gold
from resnet_fullstrength import install as install_fullstrength

# Training-only window transform; evaluation/inference apply the same flags.
CANONICAL_KEYS = {'canonical_orientation'}
trainer.DATA_KEYS.update(CANONICAL_KEYS)

_shared_receipt = trainer.source_receipt


def source_receipt(output, cfg, cohort, synthetic=False):
    if not str(cfg['arch']).startswith('convnext'):
        raise ValueError('this entrypoint trains ConvNeXt backbones only')
    if not synthetic:
        cohort['model_origin'] = 'generic-local-pretraining:' + cfg['arch']
    receipt = _shared_receipt(output, cfg, cohort, synthetic)
    for name in ('scripts/train_convnext_fullstrength.py', 'src/kneemri/coatnet_model.py',
                 'src/kneemri/gold_cohort.py'):
        source = ROOT / name
        receipt['sources'][name] = trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt.update(schema='convnext-study-mil-v1', backbone_arch=cfg['arch'],
                   prediction_contract={'models': 1, 'seeds': [cfg['seed']],
                                        'checkpoint_averaging': False,
                                        'test_time_augmentation': False})
    return receipt


def main():
    trainer.source_receipt = source_receipt
    install_gold(trainer)  # before the objective, whose constants check must see Gold
    install_fullstrength(trainer)
    trainer.main()


if __name__ == '__main__':
    main()
