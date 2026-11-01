#!/usr/bin/env python3
"""EfficientNet E2E_D study model at a fixed 224 input from the native384 corpus.

train_resnet_n384.py (R2/E1 trainer: whole-series bags, end-to-end depth,
native_aug warp/degradation keys, weight EMA) with two substitutions:
  * the encoder factory is kneemri.effnet224.build_model (timm EfficientNet);
  * bag pixels are reduced 384 -> 224 once in uint8 (effnet224.with_downscale)
    for training, Gold58 evaluation and inference alike.
``cudnn_benchmark`` is kernel choice only (whole-series bags vary N per step).
"""
import json
from pathlib import Path
import shutil

import torch

import train_resnet
from resnet_fullstrength import install
import kneemri.coatnet_original_data as original_data
from kneemri import effnet224
from kneemri import native_aug
from kneemri import resnet_depth_e2e as e2e

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('src/kneemri/resnet_depth.py', 'src/kneemri/resnet_depth_e2e.py', 'src/kneemri/native_aug.py',
           'src/kneemri/effnet224.py', 'scripts/train_effnet224.py', 'scripts/train_resnet_n384.py')


def source_receipt(output, cfg, cohort, synthetic=False):
    receipt = train_resnet.source_receipt(output, cfg, cohort, synthetic)
    for name in SOURCES:
        source = ROOT / name
        receipt['sources'][name] = train_resnet.trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt.update(schema='effnet224-study-mil-v1', backbone_family='efficientnet')
    receipt['training_bags'] = dict(sampling='whole_series', cap=cfg.get('series_block_cap', 32))
    receipt['training_degradation'] = {k: cfg.get(k) for k in sorted(native_aug.DEGRADE_KEYS)}
    receipt['input_pixels'] = dict(corpus=cfg['corpus'], resolution=cfg['resolution'],
                                   downscale=cfg.get('input_downscale', 'area'),
                                   stage='uint8_after_canonical_before_augmentation')
    if cfg.get('depth_e2e'):
        receipt['depth_e2e'] = cfg['depth_e2e']
    return receipt


def main():
    trainer = train_resnet.trainer
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS | effnet224.MODEL_KEYS)
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS | native_aug.DEGRADE_KEYS | effnet224.DATA_KEYS)
    trainer.INFRA_KEYS.add('cudnn_benchmark')
    shared_normalize = trainer.normalize_config

    def normalize_config(config):
        cfg = shared_normalize(config)
        value = cfg.get('cudnn_benchmark', True)
        if not isinstance(value, bool):
            raise ValueError('cudnn_benchmark must be boolean')
        torch.backends.cudnn.benchmark = value
        return cfg

    trainer.normalize_config = normalize_config
    built = {}

    def build_model(cfg, pretrained=False):
        model = effnet224.build_model(cfg, pretrained)
        built.update(model=model, cfg=cfg)
        return model

    # Layer-wise LR decay (``layer_decay`` in (0,1]; absent = unchanged): the trainer builds
    # AdamW([backbone, (spatial), head]); the backbone group is split per EfficientNet depth
    # (stem, blocks 0..6, conv_head) with lr * decay**(depth - stage). OneCycle then uses each
    # group's lr as its peak, so the top stage keeps backbone_lr and earlier stages scale down.
    shared_adamw = torch.optim.AdamW

    def adamw(groups, **kwargs):
        groups = list(groups)
        decay = built.get('cfg', {}).get('layer_decay')
        if decay is not None and len(groups) >= 2 and 'model' in built:
            if not 0.0 < float(decay) <= 1.0:
                raise ValueError('layer_decay must lie in (0, 1]')
            backbone = built['model'].backbone
            if {id(p) for p in groups[0]['params']} != {id(p) for p in backbone.parameters()}:
                raise ValueError('first optimizer group is not the backbone')
            groups = effnet224.split_layer_decay(backbone, groups[0], float(decay)) + groups[1:]
            print(json.dumps({'event': 'layer_decay', 'decay': float(decay),
                              'lrs': [round(g['lr'], 8) for g in groups]}), flush=True)
        return shared_adamw(groups, **kwargs)

    trainer.torch.optim.AdamW = adamw
    trainer.build_model = build_model
    trainer.source_receipt = source_receipt
    install(trainer)
    # Training, validation and Gold58 datasets all resolve this name at call time.
    original_data.CoAtNetOriginalNeighborBags = effnet224.with_downscale(
        native_aug.with_degradation(e2e.DepthSeriesBags))
    trainer.main()


if __name__ == '__main__':
    main()
