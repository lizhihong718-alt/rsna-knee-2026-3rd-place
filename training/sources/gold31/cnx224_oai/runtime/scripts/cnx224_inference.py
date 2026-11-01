#!/usr/bin/env python3
"""Single-checkpoint DICOM inference for native-224 ConvNeXt / DINO checkpoints (cnx_dino_224_20261009).

Same structure as native224_tuning_20261009/runtime/scripts/native224_inference.py (ResNet/CoAtNet):
windows are rebuilt from the raw DICOM exactly as the native corpus was built
(resnet_native512_inference.install_native: legacy window selection/metadata, pixels resampled
once by kneemri.native_crop at the checkpoint resolution, then the canonical laterality flags and
canonical lateral sagittal positions). The model comes from kneemri.depth224.build_model, the
factory used in training (ConvNeXt/DINO backbone, optional whole-series E2E DepthStudyModel), and
is scored by the audited streaming engine through its own encode and head.
One model, one seed, no TTA. Training-only keys (warp interpolation, LR schedule, bag sampling)
never reach this path.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import inference_coatnet_original_dicom as runtime
import resnet_native512_inference as rni
import train_coatnet_ablation as trainer
from kneemri import depth224
from kneemri.gold_cohort import GOLD_KEYS
from resnet_fullstrength import CONFIG_KEYS

# Same key registry as runtime/scripts/train_224.py; none of the training-only fields alters inference.
ROUTE_KEYS = {'canonical_orientation', 'warp_interpolation', 'layer_decay', 'drop_path_rate',
              'output_stride', 'dino_gap', 'position_zero_init', 'map_norm', 'input_blur_sigma'}
trainer.MODEL_KEYS.update(depth224.DEPTH_MODEL_KEYS)
trainer.DATA_KEYS.update(CONFIG_KEYS | GOLD_KEYS | ROUTE_KEYS | depth224.DEPTH_DATA_KEYS)
UNSUPPORTED = ('zoom_view', 'plane_positions')

_checkpoint_config = runtime.checkpoint_config


def checkpoint_config(checkpoint, config=None):
    if checkpoint['receipt'].get('schema') != 'cnx-dino-224-study-mil-v1':
        raise ValueError('this entrypoint requires a cnx_dino_224 single-model checkpoint')
    return _checkpoint_config(checkpoint, config)


def main():
    config = torch.load(rni.rdi.checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    if any(config.get(key) for key in UNSUPPORTED):
        raise ValueError(f'{UNSUPPORTED} need their own DICOM decode; not deployable through this entry')
    if config.get('depth_e2e') and not config.get('depth_canonical_positions'):
        raise ValueError('depth models are deployed only with canonical lateral positions')
    rni.install_native(bool(config.get('canonical_orientation')), int(config['resolution']))
    runtime.checkpoint_config = checkpoint_config
    runtime.build_model = depth224.build_model
    runtime.main()


if __name__ == '__main__':
    main()
