#!/usr/bin/env python3
"""Train one ConvNeXt or DINO ViT on the native 224 corpus (cnx_dino_224_20261009).

Same trainer, PF0.959 full-strength objective, canonical orientation and Gold58
fixed-final protocol as the ConvNeXt PF route (convnext_pf_fs_20261006). Adds only:
  warp_interpolation  training warp flag (linear legacy | cubic | lanczos4)
  layer_decay         layer-wise backbone LR decay (ConvNeXt stages / ViT blocks)
  drop_path_rate      stochastic depth in the backbone
  output_stride       ConvNeXt dilated stride (16 -> 14x14 spatial-query map at 224)
  arch                dinov3_vitl16 | dinov2_vitb14 | dino_vitb16 | dino_vits16 (DINO ViTs)
  dino_gap            DINO global residual: cls (default) | mean of patch tokens
  series_block_cap    whole-series training bags (ResNet E2E_S); depth_e2e adds DepthContext (E2E_D)
  plane_positions     with depth_canonical_positions + position_embedding/position_dim 3: per-plane
                      window coordinates (sagittal canonical lateral mm) for the MIL head
  position_zero_init  zero-init the position MLP output layer (model starts at its base)
  map_norm            groupnorm_gelu: BiT-style GroupNorm(32)+GELU on the final maps before the head
  input_blur_sigma    fixed in-model Gaussian anti-alias of every window, train AND eval (px)
  zoom_view           {corpus, fov_mm}: second 224 view of every window, centred zoom (zoom224.py)
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_coatnet_ablation as trainer
from kneemri.coatnet_model import DINO_ARCHS
from kneemri.gold_cohort import install as install_gold
from resnet_fullstrength import install as install_fullstrength

NEW_KEYS = {'canonical_orientation', 'warp_interpolation', 'layer_decay', 'drop_path_rate',
            'output_stride', 'dino_gap', 'position_zero_init', 'map_norm', 'input_blur_sigma'}
trainer.DATA_KEYS.update(NEW_KEYS)

_shared_receipt = trainer.source_receipt


def source_receipt(output, cfg, cohort, synthetic=False):
    arch = str(cfg['arch'])
    if not (arch.startswith('convnext') or arch in DINO_ARCHS):
        raise ValueError('this entrypoint trains ConvNeXt or DINO ViT backbones only')
    if not synthetic:
        cohort['model_origin'] = 'generic-local-pretraining:' + arch
    receipt = _shared_receipt(output, cfg, cohort, synthetic)
    for name in ('scripts/train_224.py', 'src/kneemri/coatnet_model.py',
                 'src/kneemri/gold_cohort.py', 'scripts/resnet_fullstrength.py', *DEPTH_SOURCES):
        source = ROOT / name
        receipt['sources'][name] = trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt.update(schema='cnx-dino-224-study-mil-v1', backbone_arch=arch,
                   prediction_contract={'models': 1, 'seeds': [cfg['seed']],
                                        'checkpoint_averaging': False,
                                        'test_time_augmentation': False})
    return receipt


def install_depth(config_path):
    """Whole-series bags (series_block_cap) and/or end-to-end depth (depth_e2e), as ResNet E2E."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not (cfg.get('series_block_cap') or cfg.get('depth_e2e') or cfg.get('depth_canonical_positions')):
        return
    if cfg.get('plane_positions') and (cfg.get('depth_e2e') or cfg.get('position_dim') != 3):
        raise ValueError('plane_positions feeds a 3-dim position embedding, not the depth context')
    import kneemri.coatnet_original_data as original_data
    from kneemri import depth224
    trainer.MODEL_KEYS.update(depth224.DEPTH_MODEL_KEYS)
    trainer.DATA_KEYS.update(depth224.DEPTH_DATA_KEYS)
    trainer.build_model = depth224.build_model
    # Training, validation and Gold58 datasets all resolve this name at call time.
    original_data.CoAtNetOriginalNeighborBags = (depth224.PlanePositionBags if cfg.get('plane_positions')
                                                 else depth224.DepthSeriesBags)
    for name in ('src/kneemri/depth224.py', 'src/kneemri/resnet_depth.py'):
        DEPTH_SOURCES.append(name)


def install_zoom(config_path):
    """Two-view (140 mm + centred zoom) bags for training, Gold58 and multi-budget evaluation."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not cfg.get('zoom_view'):
        return
    if cfg.get('series_block_cap') or cfg.get('depth_e2e'):
        raise ValueError('zoom views duplicate depth positions; not combined with depth bags')
    import kneemri.coatnet_original_data as original_data
    from kneemri import zoom224
    trainer.DATA_KEYS.update(zoom224.ZOOM_KEYS)
    original_data.CoAtNetOriginalNeighborBags = zoom224.ZoomViewBags
    DEPTH_SOURCES.append('src/kneemri/zoom224.py')


DEPTH_SOURCES = []


def main():
    if '--config' in sys.argv:
        install_depth(sys.argv[sys.argv.index('--config') + 1])
        install_zoom(sys.argv[sys.argv.index('--config') + 1])
    trainer.source_receipt = source_receipt
    install_gold(trainer)
    install_fullstrength(trainer)
    trainer.main()


if __name__ == '__main__':
    main()
