#!/usr/bin/env python3
"""Stage-1 ResNet with whole-series training bags and (optionally) an end-to-end depth module.

Identical to train_resnet_fullstrength.py except: training bags are whole series
(``DepthSeriesBags``), sagittal positions may be canonical lateral mm, and a config
with ``depth_e2e`` builds ``DepthStudyModel`` (depth context before the spatial queries).
"""
from pathlib import Path
import shutil

import train_resnet
from resnet_fullstrength import install
import kneemri.coatnet_original_data as original_data
from kneemri import resnet_depth_e2e as e2e

ROOT = Path(__file__).resolve().parents[1]


def build_model(cfg, pretrained=False):
    if cfg.get('depth_e2e'):
        return e2e.build_model(cfg, pretrained)
    return train_resnet.build_model(cfg, pretrained)


def source_receipt(output, cfg, cohort, synthetic=False):
    receipt = train_resnet.source_receipt(output, cfg, cohort, synthetic)
    for name in ('src/kneemri/resnet_depth.py', 'src/kneemri/resnet_depth_e2e.py',
                 'scripts/train_resnet_depth_e2e.py'):
        source = ROOT / name
        receipt['sources'][name] = train_resnet.trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt['training_bags'] = dict(sampling='whole_series', cap=cfg.get('series_block_cap', 32))
    if cfg.get('depth_e2e'):
        receipt['depth_e2e'] = cfg['depth_e2e']
    return receipt


def main():
    trainer = train_resnet.trainer
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS)
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS)
    trainer.build_model = build_model
    trainer.source_receipt = source_receipt
    install(trainer)
    # Training, validation and Gold58 datasets all resolve this name at call time.
    original_data.CoAtNetOriginalNeighborBags = e2e.DepthSeriesBags
    trainer.main()


if __name__ == '__main__':
    main()
