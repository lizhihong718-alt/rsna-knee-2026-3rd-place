#!/usr/bin/env python3
"""Single-checkpoint DICOM inference for the PF0.959 full-strength ConvNeXt route.

Same as resnet_wave2_20261006/inference_canonical/inference_resnet_fullstrength_dicom.py
(the ResNet .959 path) except that the model comes from the shared CoAtNet/ConvNeXt
builder already used by inference_coatnet_original_dicom. A checkpoint trained with
``canonical_orientation`` gets the matching laterality canonicalization of its windows.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import inference_coatnet_original_dicom as runtime
import train_coatnet_ablation as trainer
from kneemri.gold_cohort import GOLD_KEYS
from resnet_fullstrength import CONFIG_KEYS

# These fields describe the training objective/cohort and do not alter inference;
# canonical_orientation is honoured below by installing the DICOM flag computation.
trainer.DATA_KEYS.update(CONFIG_KEYS | GOLD_KEYS | {'canonical_orientation'})

_checkpoint_config = runtime.checkpoint_config


def checkpoint_config(checkpoint, config=None):
    if checkpoint['receipt'].get('schema') != 'convnext-study-mil-v1':
        raise ValueError('this entrypoint requires an audited ConvNeXt checkpoint')
    return _checkpoint_config(checkpoint, config)


def checkpoint_path(argv):
    for index, value in enumerate(argv):
        if value == '--checkpoint':
            return argv[index + 1]
        if value.startswith('--checkpoint='):
            return value.split('=', 1)[1]
    raise ValueError('--checkpoint is required')


def main():
    config = torch.load(checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    if config.get('canonical_orientation'):
        import resnet_canonical
        resnet_canonical.install()
        print('laterality canonicalization: enabled', flush=True)
    runtime.checkpoint_config = checkpoint_config
    runtime.main()


if __name__ == '__main__':
    main()
