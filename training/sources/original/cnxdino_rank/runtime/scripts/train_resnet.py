#!/usr/bin/env python3
"""Train one ResNet on the existing audited Rayan/grouped-CV data contract.

Only the model factory and provenance hook are replaced in this process. The
CoAtNet source and running CoAtNet jobs are not modified. All optimizer, resume,
validation, full-refit, masking, and deterministic sampling code is shared.
"""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_coatnet_ablation as trainer
from kneemri.resnet_model import build_model, register_config

register_config(trainer)

_shared_receipt = trainer.source_receipt


def source_receipt(output, cfg, cohort, synthetic=False):
    if not synthetic:
        cohort['model_origin'] = 'generic-local-pretraining:' + cfg['arch']
    receipt = _shared_receipt(output, cfg, cohort, synthetic)
    for name in ('scripts/train_resnet.py', 'src/kneemri/resnet_model.py',
                 'src/kneemri/resnet_evidence_data.py', 'src/kneemri/gold_cohort.py'):
        source = ROOT / name
        receipt['sources'][name] = trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt.update(schema='resnet-study-mil-v1', backbone_arch=cfg['arch'],
                   prediction_contract={'models': 1, 'seeds': [cfg['seed']],
                                        'checkpoint_averaging': False,
                                        'test_time_augmentation': False})
    if cfg.get('resnet_pyramid', False):
        receipt['feature_decoder'] = dict(
            name='residual_c3_c4_c5_gated_gem', reductions=[8, 16, 32],
            output_stride=8, width=cfg.get('pyramid_width', 256),
            semantic_path='original_c5_spatial_query_plus_gap',
            residual_initialization='zero', backbone_count=1)
    metadata = Path(cfg.get('initial_weights', '')).parent / 'pretraining.json'
    if not synthetic and metadata.is_file():
        details = json.loads(metadata.read_text())
        if details['weights_sha256'] != trainer.hash_file(cfg['initial_weights']):
            raise ValueError('pretraining provenance differs from the supplied weights')
        receipt['pretraining'] = details
    return receipt


def main():
    trainer.build_model = build_model
    trainer.source_receipt = source_receipt
    trainer.main()


if __name__ == '__main__':
    main()
