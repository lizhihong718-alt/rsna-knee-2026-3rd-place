#!/usr/bin/env python3
"""Stream raw competition DICOMs through exactly one ResNet checkpoint."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import inference_coatnet_original_dicom as runtime
from kneemri import resnet_model
from kneemri.resnet_model import build_model
import train_coatnet_ablation as trainer

# Older packaged ResNet checkpoints carry their original model source, which
# predates the optional pyramid configuration registration.
if hasattr(resnet_model, 'register_config'):
    resnet_model.register_config(trainer)

_checkpoint_config = runtime.checkpoint_config


def checkpoint_config(checkpoint, config=None):
    if checkpoint['receipt'].get('schema') != 'resnet-study-mil-v1':
        raise ValueError('this entrypoint requires an audited ResNet checkpoint')
    return _checkpoint_config(checkpoint, config)


def main():
    runtime.build_model = build_model
    runtime.checkpoint_config = checkpoint_config
    runtime.main()


if __name__ == '__main__':
    main()
