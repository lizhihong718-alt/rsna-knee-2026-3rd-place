#!/usr/bin/env python3
"""EfficientNet E2E_D study model at a fixed 384 input on the native384 corpus.

train_resnet_n384.py (R2/E1 trainer: whole-series bags, end-to-end depth,
native_aug warp/degradation keys, weight EMA) with the encoder factory replaced by
kneemri.effnet224.build_model (timm EfficientNet; the module is resolution-agnostic
and is reused byte-for-byte from the 224 route). Pixels are the native384 windows
as stored: resolution must equal the corpus resolution (384), so no resampling of
any kind happens between the corpus and the network (no ``input_downscale``, no
loader interpolation, no in-model resize).
``cudnn_benchmark`` is kernel choice only (whole-series bags vary N per step).
``warp_device: gpu`` runs the training Lanczos-4 warp on the GPU with kneemri.gpu_warp, a bit-exact
re-implementation of cv2.warpAffine (same random draws, same pixels; analysis/check_gpu_warp.py), so it
is infrastructure only: at 384 the CPU warp costs ~0.9 core-s per study and starved the GPU.
"""
import json
from pathlib import Path
import shutil

import torch

import train_resnet
from resnet_fullstrength import install
import kneemri.coatnet_original_data as original_data
from kneemri import effnet224
from kneemri import gpu_warp
import kneemri.coatnet_training_data as training_data
from kneemri import native_aug
from kneemri import resnet_depth_e2e as e2e

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('src/kneemri/resnet_depth.py', 'src/kneemri/resnet_depth_e2e.py', 'src/kneemri/native_aug.py',
           'src/kneemri/effnet224.py', 'src/kneemri/gpu_warp.py', 'scripts/train_effnet384.py',
           'scripts/train_resnet_n384.py')
RESOLUTION = 384


def source_receipt(output, cfg, cohort, synthetic=False):
    receipt = train_resnet.source_receipt(output, cfg, cohort, synthetic)
    for name in SOURCES:
        source = ROOT / name
        receipt['sources'][name] = train_resnet.trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt.update(schema='effnet384-study-mil-v1', backbone_family='efficientnet')
    receipt['training_bags'] = dict(sampling='whole_series', cap=cfg.get('series_block_cap', 32))
    receipt['training_degradation'] = {k: cfg.get(k) for k in sorted(native_aug.DEGRADE_KEYS)}
    receipt['input_pixels'] = dict(corpus=cfg['corpus'], resolution=cfg['resolution'], resample='none')
    receipt['training_warp'] = dict(interpolation=cfg.get('warp_interpolation', 'linear'),
                                    device=cfg.get('warp_device', 'cpu'), bit_exact_vs='cv2.warpAffine')
    if cfg.get('depth_e2e'):
        receipt['depth_e2e'] = cfg['depth_e2e']
    return receipt


def main():
    trainer = train_resnet.trainer
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS | effnet224.MODEL_KEYS)
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS | native_aug.DEGRADE_KEYS | effnet224.DATA_KEYS | gpu_warp.WARP_KEYS)
    trainer.INFRA_KEYS.update({'cudnn_benchmark'} | gpu_warp.WARP_KEYS)
    shared_normalize = trainer.normalize_config

    def normalize_config(config):
        cfg = shared_normalize(config)
        value = cfg.get('cudnn_benchmark', False)
        if not isinstance(value, bool):
            raise ValueError('cudnn_benchmark must be boolean')
        torch.backends.cudnn.benchmark = value
        if int(cfg['resolution']) != RESOLUTION or 'native384_crop140' not in str(cfg['corpus']):
            raise ValueError('EffNet@384 route: resolution 384 on the native384_crop140 corpus only')
        if cfg.get('warp_device', 'cpu') not in ('cpu', 'gpu'):
            raise ValueError('warp_device must be cpu or gpu')
        if cfg.get('input_downscale') is not None or cfg.get('model_input_size') not in (None, RESOLUTION):
            raise ValueError('EffNet@384 route never resamples the corpus windows')
        return cfg

    trainer.normalize_config = normalize_config
    built = {}

    def build_model(cfg, pretrained=False):
        model = effnet224.build_model(cfg, pretrained)
        built.update(model=model, cfg=cfg)
        return model

    # Layer-wise LR decay (``layer_decay`` in (0,1]; absent = unchanged): the backbone group of
    # AdamW([backbone, (spatial), head]) is split per EfficientNet depth (stem, blocks, conv_head)
    # with lr * decay**(depth - stage); OneCycle uses each group's lr as its peak.
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
    # Training, validation and Gold58 datasets all resolve this name at call time; the
    # degradation mixin only acts when train=True. No downscale mixin: windows stay at 384.
    original_data.CoAtNetOriginalNeighborBags = gpu_warp.with_gpu_warp(
        native_aug.with_degradation(e2e.DepthSeriesBags))
    # make_loader imports the collate at call time; device_images is resolved at call time.
    training_data.coatnet_collate = gpu_warp.collate_with_warp(training_data.coatnet_collate)
    trainer.device_images = gpu_warp.device_images_with_warp(trainer.device_images)
    trainer.main()


if __name__ == '__main__':
    main()
