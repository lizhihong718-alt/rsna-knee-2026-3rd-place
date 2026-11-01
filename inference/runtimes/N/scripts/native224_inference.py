#!/usr/bin/env python3
"""Single-checkpoint DICOM inference for native-resolution native224 checkpoints (BiT ResNet or CoAtNet).

Windows are rebuilt from the raw DICOM exactly as the native corpus was built
(resnet_native512_inference.install_native: legacy window selection/metadata,
pixels resampled once by kneemri.native_crop at the checkpoint resolution, then
the canonical laterality flags and lateral sagittal positions). The model comes
from kneemri.native224.build_model, the factory used in training, so the
end-to-end depth model (DepthStudyModel) is scored by the audited streaming
engine through its own encode (final maps) and head (depth + queries + MIL).
One model, one seed, no TTA.
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
from kneemri import native224
from kneemri.gold_cohort import GOLD_KEYS
from kneemri.resnet_model import register_config
from resnet_fullstrength import CONFIG_KEYS

register_config(trainer)
trainer.MODEL_KEYS.update(native224.MODEL_KEYS)
# Training-objective/cohort/sampling fields do not alter inference.
trainer.DATA_KEYS.update(CONFIG_KEYS | GOLD_KEYS | native224.DATA_KEYS | {'cudnn_benchmark'})
trainer.INFRA_KEYS.add('cudnn_benchmark')

_checkpoint_config = runtime.checkpoint_config


def checkpoint_config(checkpoint, config=None):
    if checkpoint['receipt'].get('schema') != 'native224-study-mil-v1':
        raise ValueError('this entrypoint requires a native224 single-model checkpoint')
    return _checkpoint_config(checkpoint, config)


def main():
    config = torch.load(rni.rdi.checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    if config.get('depth_e2e') and not config.get('depth_canonical_positions'):
        raise ValueError('depth models are deployed only with canonical lateral positions')
    rni.install_native(bool(config.get('canonical_orientation')), int(config['resolution']))
    runtime.checkpoint_config = checkpoint_config
    runtime.build_model = native224.build_model
    runtime.main()


if __name__ == '__main__':
    main()
