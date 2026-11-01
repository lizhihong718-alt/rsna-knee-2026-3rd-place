#!/usr/bin/env python3
"""Single-checkpoint DICOM inference for Swin checkpoints trained on native224 / native384 / native512 windows.

Pixels: resnet_native512_inference.install_native(·, R) rebuilds each study's windows from the raw
DICOM exactly as the nativeR corpus was built (legacy window selection/metadata, one native_crop
resample to R, canonical laterality flags, lateral sagittal positions; train-DICOM parity with pixel
difference 0 at 224 and 384 in the native224 / native384 routes). The windows go to the network as
they are (R = checkpoint resolution = corpus resolution); an optional in-model resize
(``model_input_size``) is part of the model. ``warp_device`` is a training-only infrastructure key; so are
``blur_*`` / ``degrade_before_warp`` (training augmentation) and ``compile_*`` (training forward only; eval is eager).
The model is built by the training factory kneemri.swin.build_model and scored by the audited
streaming engine. One model, one seed, no TTA.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import coatnet_fast_runtime as cfr
import inference_coatnet_original_dicom as runtime
import resnet_native512_inference as rni
import train_coatnet_ablation as trainer
from kneemri import gpu_warp, native_aug, swin
from kneemri.resnet_depth_e2e import E2E_DATA_KEYS, E2E_MODEL_KEYS
from kneemri.resnet_model import register_config
from resnet_fullstrength import CONFIG_KEYS

RESOLUTIONS = (224, 384, 512)
register_config(trainer)
trainer.MODEL_KEYS.update(E2E_MODEL_KEYS | swin.MODEL_KEYS)
# Training-objective/sampling/augmentation fields do not alter inference.
trainer.DATA_KEYS.update(CONFIG_KEYS | E2E_DATA_KEYS | native_aug.DEGRADE_KEYS | swin.DATA_KEYS
                         | gpu_warp.WARP_KEYS | gpu_warp.BLUR_KEYS | gpu_warp.ORDER_KEYS)
trainer.INFRA_KEYS.update(swin.INFRA | gpu_warp.WARP_KEYS)

_checkpoint_config = runtime.checkpoint_config


def checkpoint_config(checkpoint, config=None):
    if checkpoint['receipt'].get('schema') != 'swin-study-mil-v1':
        raise ValueError('this entrypoint requires a swin single-model checkpoint')
    return _checkpoint_config(checkpoint, config)


def install(config):
    """Patch the preprocessing entry (before the worker pool forks) to emit model-resolution windows."""
    resolution = int(config['resolution'])
    if resolution not in RESOLUTIONS or config.get('input_downscale') is not None:
        raise ValueError('Swin checkpoints read native224/native384/native512 windows without resampling')
    native = rni.install_native(bool(config.get('canonical_orientation')), resolution)
    cfr.prepare_study_fast = native
    runtime.prepare_study_fast = native
    return native


def main():
    config = torch.load(rni.rdi.checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    if config.get('depth_e2e') and not config.get('depth_canonical_positions'):
        raise ValueError('depth models are deployed only with canonical lateral positions')
    install(config)
    runtime.checkpoint_config = checkpoint_config
    runtime.build_model = swin.build_model
    runtime.main()


if __name__ == '__main__':
    main()
