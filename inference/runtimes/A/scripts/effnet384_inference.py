#!/usr/bin/env python3
"""Single-checkpoint DICOM inference for EffNet@384 checkpoints trained on native384 windows.

Pixels: resnet_native512_inference.install_native(·, 384) rebuilds each study's windows from
the raw DICOM exactly as the native384 corpus was built (legacy window selection/metadata,
one native_crop resample to 384, canonical laterality flags, lateral sagittal positions;
train-DICOM parity 31/31 + 58/58 with pixel difference 0, native384_tuning_20261009). The
windows go to the network as they are: the checkpoint resolution must be 384, so
kneemri.effnet224.downscale (called at the same point as the 224 route) is the identity.
``warp_device`` is a training-only infrastructure key (bit-exact GPU training warp). The model is built by the training
factory kneemri.effnet224.build_model and scored by the audited streaming engine. One model,
one seed, no TTA.
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
from kneemri import effnet224, gpu_warp, native_aug
from kneemri.resnet_depth_e2e import E2E_DATA_KEYS, E2E_MODEL_KEYS
from kneemri.resnet_model import register_config
from resnet_fullstrength import CONFIG_KEYS

CORPUS_RESOLUTION = 384
register_config(trainer)
trainer.MODEL_KEYS.update(E2E_MODEL_KEYS | effnet224.MODEL_KEYS)
# Training-objective/sampling/augmentation fields do not alter inference.
trainer.DATA_KEYS.update(CONFIG_KEYS | E2E_DATA_KEYS | native_aug.DEGRADE_KEYS | effnet224.DATA_KEYS
                         | gpu_warp.WARP_KEYS)
trainer.INFRA_KEYS.update({'cudnn_benchmark'} | gpu_warp.WARP_KEYS)

_checkpoint_config = runtime.checkpoint_config


def checkpoint_config(checkpoint, config=None):
    if checkpoint['receipt'].get('schema') != 'effnet384-study-mil-v1':
        raise ValueError('this entrypoint requires an effnet384 single-model checkpoint')
    return _checkpoint_config(checkpoint, config)


def install(config):
    """Patch the preprocessing entry (before the worker pool forks) to emit model-resolution windows."""
    resolution = int(config['resolution'])
    if resolution != CORPUS_RESOLUTION or config.get('input_downscale') is not None:
        raise ValueError('EffNet@384 checkpoints read native384 windows without resampling')
    mode = 'area'
    native = rni.install_native(bool(config.get('canonical_orientation')), CORPUS_RESOLUTION)

    def prepare(*args, **kwargs):
        out = native(*args, **kwargs)
        uid, arrays = out[0], out[1]
        arrays = (arrays[0], effnet224.downscale(arrays[1], resolution, mode), *arrays[2:])
        return (uid, arrays, *out[2:])

    cfr.prepare_study_fast = prepare
    runtime.prepare_study_fast = prepare
    return prepare


def main():
    config = torch.load(rni.rdi.checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    if config.get('depth_e2e') and not config.get('depth_canonical_positions'):
        raise ValueError('depth models are deployed only with canonical lateral positions')
    install(config)
    runtime.checkpoint_config = checkpoint_config
    runtime.build_model = effnet224.build_model
    runtime.main()


if __name__ == '__main__':
    main()
