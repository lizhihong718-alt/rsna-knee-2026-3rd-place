#!/usr/bin/env python3
"""Train one native-224 study model (BiT ResNet or CoAtNet), single seed, no ensemble.

Shared trainer, PF0.959 mass-preserved full-strength objective and Gold58 cohort
hooks exactly as train_resnet_depth_e2e.py / train_coatnet_fullstrength.py. The
model factory and bag class come from kneemri.native224, whose options all
default to those source recipes.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch

import train_coatnet_ablation as trainer
import kneemri.coatnet_original_data as original_data
from kneemri import native224
from kneemri.gold_cohort import install as install_gold
from kneemri.resnet_model import register_config
from resnet_fullstrength import install as install_fullstrength

register_config(trainer)
trainer.MODEL_KEYS.update(native224.MODEL_KEYS)
trainer.DATA_KEYS.update(native224.DATA_KEYS)
# Infrastructure only (kernel choice, not the math). raptor_reference, imported by
# the trainer, turns cuDNN autotuning on globally; with whole-series bags every step
# has a new window count, and CoAtNet's depthwise convs then re-autotune each step
# (measured: ~9 s forward / ~37 s backward per step instead of ~1 s).
trainer.DATA_KEYS.add('cudnn_benchmark')
trainer.INFRA_KEYS.add('cudnn_benchmark')
_shared_normalize = trainer.normalize_config


def normalize_config(config):
    cfg = _shared_normalize(config)
    value = cfg.get('cudnn_benchmark', True)
    if not isinstance(value, bool):
        raise ValueError('cudnn_benchmark must be boolean')
    torch.backends.cudnn.benchmark = value
    return cfg

_shared_receipt = trainer.source_receipt

SOURCES = ('scripts/train_native224.py', 'src/kneemri/native224.py', 'src/kneemri/resnet_model.py',
           'src/kneemri/coatnet_model.py', 'src/kneemri/resnet_depth.py', 'src/kneemri/resnet_depth_e2e.py',
           'src/kneemri/gold_cohort.py', 'src/kneemri/native_crop.py')


def source_receipt(output, cfg, cohort, synthetic=False):
    if not synthetic:
        cohort['model_origin'] = 'generic-local-pretraining:' + cfg['arch']
    receipt = _shared_receipt(output, cfg, cohort, synthetic)
    for name in SOURCES:
        source = ROOT / name
        receipt['sources'][name] = trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    family = 'coatnet' if str(cfg['arch']).startswith('coatnet') else 'resnet'
    receipt.update(schema='native224-study-mil-v1', backbone_family=family, backbone_arch=cfg['arch'],
                   prediction_contract={'models': 1, 'seeds': [cfg['seed']],
                                        'checkpoint_averaging': False,
                                        'test_time_augmentation': False})
    receipt['training_bags'] = dict(
        sampling='whole_series' if cfg.get('series_block_cap') is not None else 'random_windows',
        cap=cfg.get('series_block_cap'), blur_prob=cfg.get('train_blur_prob'),
        blur_sigma=cfg.get('train_blur_sigma'))
    if cfg.get('depth_e2e'):
        receipt['depth_e2e'] = cfg['depth_e2e']
    return receipt


def main():
    trainer.normalize_config = normalize_config
    trainer.build_model = native224.build_model
    trainer.source_receipt = source_receipt
    install_gold(trainer)  # before the objective, whose constants check must see Gold
    install_fullstrength(trainer)
    # Training, validation and Gold58 datasets all resolve this name at call time.
    original_data.CoAtNetOriginalNeighborBags = native224.NativeBags
    trainer.main()


if __name__ == '__main__':
    main()
