#!/usr/bin/env python3
"""Train one ConvNeXt or DINOv3-ViT on the native384 corpus (PF0.959 full strength).

Copy of train_convnext_fullstrength.py (the ConvNeXt LB .958 path) that also admits
DINOv3 ViTs and four training-only knobs, all defaulting to the legacy behaviour:
  warp_interpolation (linear) / geometric_prob (1.0)  -> data loader warp
  layer_decay (1.0) / layer_decay_coarse (False)      -> optimizer groups
  drop_path_rate (0.0)                                -> timm backbone
None of them changes evaluation or inference pixels or the model state contract.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_coatnet_ablation as trainer
from kneemri.coatnet_model import is_dino_vit
from kneemri.gold_cohort import install as install_gold
from resnet_fullstrength import install as install_fullstrength

CANONICAL_KEYS = {'canonical_orientation'}
TRAINING_KNOBS = {'warp_interpolation', 'geometric_prob', 'layer_decay', 'layer_decay_coarse',
                  'blur_prob', 'blur_sigma', 'resample_prob', 'resample_scale'}
trainer.DATA_KEYS.update(CANONICAL_KEYS | TRAINING_KNOBS)
trainer.MODEL_KEYS.update({'drop_path_rate', 'output_stride', 'dino_gap', 'dino_layers'})
# runtime_b: throughput-only keys (kneemri/fast512.py); training forward compiled, eval/inference eager.
trainer.DEFAULTS.update(compile_backbone=False, compile_bucket=16)
trainer.INFRA_KEYS.update({'compile_backbone', 'compile_bucket'})

_shared_receipt = trainer.source_receipt


def check_knobs(cfg):
    if cfg.get('warp_interpolation', 'linear') not in {'linear', 'cubic', 'area', 'lanczos4'}:
        raise ValueError('warp_interpolation must be linear, cubic, area or lanczos4')
    if not 0.0 <= float(cfg.get('geometric_prob', 1.0)) <= 1.0:
        raise ValueError('geometric_prob must lie in [0,1]')
    if not 0.0 <= float(cfg.get('blur_prob', 0.0)) <= 1.0:
        raise ValueError('blur_prob must lie in [0,1]')
    if not 0.0 <= float(cfg.get('resample_prob', 0.0)) <= 1.0:
        raise ValueError('resample_prob must lie in [0,1]')
    lo, hi = cfg.get('resample_scale', [0.5, 0.9])
    if not 0.25 <= float(lo) <= float(hi) <= 1.0:
        raise ValueError('resample_scale must satisfy 0.25 <= lo <= hi <= 1')
    if not 0.0 < float(cfg.get('layer_decay', 1.0)) <= 1.0:
        raise ValueError('layer_decay must lie in (0,1]')
    if not 0.0 <= float(cfg.get('drop_path_rate', 0.0)) < 0.5:
        raise ValueError('drop_path_rate must lie in [0,0.5)')
    return cfg


def source_receipt(output, cfg, cohort, synthetic=False):
    arch = str(cfg['arch'])
    if not (arch.startswith('convnext') or is_dino_vit(arch)):
        raise ValueError('this entrypoint trains ConvNeXt or DINOv3 ViT backbones only')
    if not synthetic:
        cohort['model_origin'] = 'generic-local-pretraining:' + arch
    receipt = _shared_receipt(output, cfg, cohort, synthetic)
    for name in ('scripts/train_cnx_dino384.py', 'src/kneemri/coatnet_model.py', 'src/kneemri/fast512.py',
                 'src/kneemri/gold_cohort.py', 'src/kneemri/coatnet_original_data.py', *DEPTH_SOURCES):
        source = ROOT / name
        receipt['sources'][name] = trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt.update(schema='cnx-dino-native512-study-mil-v1', backbone_arch=arch, runtime='runtime_b',
                   compiled_training_forward=bool(cfg.get('compile_backbone')),
                   prediction_contract={'models': 1, 'seeds': [cfg['seed']],
                                        'checkpoint_averaging': False,
                                        'test_time_augmentation': False})
    return receipt


DEPTH_SOURCES = []


def install_depth(config_path):
    """runtime_d: whole-series bags (series_block_cap) + end-to-end DepthContext (depth_e2e), as ResNet E2E_D."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not (cfg.get('series_block_cap') or cfg.get('depth_e2e')):
        return
    import kneemri.coatnet_original_data as original_data
    from kneemri import depth384
    trainer.MODEL_KEYS.update(depth384.DEPTH_MODEL_KEYS)
    trainer.DATA_KEYS.update(depth384.DEPTH_DATA_KEYS)
    trainer.build_model = depth384.build_model
    # Training, validation and Gold58 datasets all resolve this name at call time.
    original_data.CoAtNetOriginalNeighborBags = depth384.DepthSeriesBags
    DEPTH_SOURCES.extend(['src/kneemri/depth384.py', 'src/kneemri/resnet_depth.py'])


def install_compile():
    """runtime_b: wrap the (possibly depth-) model builder so the training forward is compiled."""
    build = trainer.build_model

    def build_model(cfg, pretrained=False):
        model = build(cfg, pretrained=pretrained)
        if cfg.get('compile_backbone'):
            from kneemri import fast512
            fast512.install(model, int(cfg.get('compile_bucket', 16)))
        return model
    trainer.build_model = build_model


def main():
    if '--config' in sys.argv:
        install_depth(sys.argv[sys.argv.index('--config') + 1])
    install_compile()
    trainer.source_receipt = source_receipt
    install_gold(trainer)
    install_fullstrength(trainer)
    inner = trainer.normalize_config

    def normalize(config):
        return check_knobs(inner(config))
    trainer.normalize_config = normalize
    trainer.main()


if __name__ == '__main__':
    main()
