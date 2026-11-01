#!/usr/bin/env python3
"""ResNet whole-series (+optional end-to-end depth) trainer for native single-resample corpora.

train_resnet_depth_e2e.py unchanged, plus kneemri.native_aug: an optional training-only
resolution degradation (``degrade_*`` keys). With degrade_prob 0 (default) this is
exactly the E2E trainer.

runtime_up (idle_wave_20261010): + ``model_input_size`` from native224 (multiple of 32 above
``resolution``): windows are bilinearly upsampled inside the backbone's forward_features, so
training, evaluation and DICOM inference see the same op. Absent -> unchanged.
"""
from pathlib import Path
import shutil

import torch.nn.functional as F

import train_resnet
from resnet_fullstrength import install
import kneemri.coatnet_original_data as original_data
from kneemri import native_aug
from kneemri import resnet_depth_e2e as e2e

ROOT = Path(__file__).resolve().parents[1]


_UPSAMPLED = {}


def _upsampled_class(base):
    """Subclass of the timm backbone class whose forward_features upsamples first (native224.py)."""
    if base not in _UPSAMPLED:
        def forward_features(self, x):
            size = self.model_input_size
            if x.shape[-1] != size or x.shape[-2] != size:
                x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
            return base.forward_features(self, x)
        _UPSAMPLED[base] = type(f'Upsampled{base.__name__}', (base,), {'forward_features': forward_features})
    return _UPSAMPLED[base]


def build_model(cfg, pretrained=False):
    size = cfg.get('model_input_size')
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size % 32
                             or size <= int(cfg['resolution'])):
        raise ValueError('model_input_size must be a multiple of 32 above resolution')
    model = e2e.build_model(cfg, pretrained) if cfg.get('depth_e2e') else train_resnet.build_model(cfg, pretrained)
    if size:
        model.backbone.__class__ = _upsampled_class(type(model.backbone))
        model.backbone.model_input_size = int(size)
    return model


def source_receipt(output, cfg, cohort, synthetic=False):
    receipt = train_resnet.source_receipt(output, cfg, cohort, synthetic)
    for name in ('src/kneemri/resnet_depth.py', 'src/kneemri/resnet_depth_e2e.py',
                 'src/kneemri/native_aug.py', 'scripts/train_resnet_n384.py'):
        source = ROOT / name
        receipt['sources'][name] = train_resnet.trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt['training_bags'] = dict(sampling='whole_series', cap=cfg.get('series_block_cap', 32))
    receipt['training_degradation'] = {k: cfg.get(k) for k in sorted(native_aug.DEGRADE_KEYS)}
    if cfg.get('depth_e2e'):
        receipt['depth_e2e'] = cfg['depth_e2e']
    return receipt


def main():
    trainer = train_resnet.trainer
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS | {'model_input_size'})
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS | native_aug.DEGRADE_KEYS)
    trainer.build_model = build_model
    trainer.source_receipt = source_receipt
    install(trainer)
    # Training, validation and Gold58 datasets all resolve this name at call time;
    # the mixin only acts when train=True.
    original_data.CoAtNetOriginalNeighborBags = native_aug.with_degradation(e2e.DepthSeriesBags)
    trainer.main()


if __name__ == '__main__':
    main()
