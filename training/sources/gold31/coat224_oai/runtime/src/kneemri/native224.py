"""Native-224 study models and training bags (ResNet BiT or CoAtNet, one model).

Every option defaults to the source recipes, so a config without them is exactly
the ResNet E2E route (train_resnet_depth_e2e.py, LB .961) or the CoAtNet PF
full-strength route (train_coatnet_fullstrength.py, LB .959):

* ``depth_e2e`` (dict): DepthContext between the final maps and the spatial
  queries, for either backbone (resnet_depth_e2e.DepthStudyModel is backbone
  agnostic: it only needs ``forward_features`` NCHW maps and ``forward_head``).
* ``series_block_cap`` (int): training bags are whole series up to the cap
  (resnet_depth_e2e.DepthSeriesBags). Absent -> the original random windows.
* ``backbone_output_stride`` (ResNet only, 32 or 16): dilate the last BiT stage
  instead of striding it, so a 224 window gives a 14x14 final map (the 384
  recipe sees 12x12). Pretrained weights are unchanged.
* ``train_blur_prob`` / ``train_blur_sigma`` ([lo, hi] px): training-only
  Gaussian blur, one sigma per study bag. Evaluation and inference never blur.
* ``model_input_size`` (multiple of 32 > resolution): the native windows are
  bilinearly upsampled inside the model, right before the backbone, so every
  stage works on a finer grid of the same 224 px information. Identical in
  training, evaluation and DICOM inference (it is part of forward_features).
"""
import random

import numpy as np
import torch.nn.functional as F

from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags
from kneemri.resnet_depth_e2e import DepthSeriesBags, DepthStudyModel

MODEL_KEYS = {'depth_e2e', 'backbone_output_stride', 'model_input_size'}
DATA_KEYS = {'depth_canonical_positions', 'series_block_cap', 'train_blur_prob', 'train_blur_sigma',
             'canonical_orientation'}
_UPSAMPLED = {}


def _upsampled_class(base):
    """Subclass of the timm backbone class whose forward_features upsamples first."""
    if base not in _UPSAMPLED:
        def forward_features(self, x):
            size = self.model_input_size
            if x.shape[-1] != size or x.shape[-2] != size:
                x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
            return base.forward_features(self, x)
        _UPSAMPLED[base] = type(f'Upsampled{base.__name__}', (base,), {'forward_features': forward_features})
    return _UPSAMPLED[base]


def build_model(cfg, pretrained=False):
    arch = str(cfg['arch'])
    size = cfg.get('model_input_size')
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size % 32
                             or size <= int(cfg['resolution'])):
        raise ValueError('model_input_size must be a multiple of 32 above resolution')
    if arch.startswith('coatnet'):
        if cfg.get('backbone_output_stride', 32) != 32:
            raise ValueError('backbone_output_stride is implemented for BiT ResNets only')
        from kneemri import coatnet_model
        # CoAtNet's relative-position tables are sized from img_size = the backbone input.
        base = coatnet_model.build_model(dict(cfg, resolution=size) if size else cfg, pretrained=pretrained)
    else:
        from kneemri import resnet_model
        base = resnet_model.build_model(cfg, pretrained=pretrained)
    if size:
        base.backbone.__class__ = _upsampled_class(type(base.backbone))
        base.backbone.model_input_size = int(size)
    if not cfg.get('depth_e2e'):
        return base
    # Same construction as resnet_depth_e2e.build_model: the pretrained backbone
    # object is reused, the head is freshly initialised.
    return DepthStudyModel(base.backbone, cfg)


class NativeBags(DepthSeriesBags):
    """DepthSeriesBags whose whole-series sampling and blur are opt-in."""

    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
        super().__init__(cfg, all_ids, ids, labels, train, uint8_output)
        self.whole_series = cfg.get('series_block_cap') is not None
        self.blur_prob = float(cfg.get('train_blur_prob') or 0.)
        sigma = cfg.get('train_blur_sigma') or [0., 0.]
        self.blur_sigma = (float(sigma[0]), float(sigma[1]))
        if not 0 <= self.blur_prob <= 1 or not 0 <= self.blur_sigma[0] <= self.blur_sigma[1] <= 3:
            raise ValueError('train_blur_prob must be in [0,1] and train_blur_sigma 0 <= lo <= hi <= 3')
        if self.blur_prob and self.blur_sigma[1] <= 0:
            raise ValueError('blur enabled with zero sigma')

    def _selected(self, row, budget=None):
        if self.train and not self.whole_series:
            return CoAtNetOriginalNeighborBags._selected(self, row, budget)
        return super()._selected(row, budget)

    def _geometric(self, images):
        images = super()._geometric(images)
        if self.train and self.blur_prob and random.random() < self.blur_prob:
            import cv2
            cv2.setNumThreads(1)
            sigma = random.uniform(*self.blur_sigma)
            h = images.shape[-1]
            flat = images.reshape(-1, h, h)
            images = np.stack([cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                                borderType=cv2.BORDER_CONSTANT) for frame in flat])
            images = images.reshape(-1, 3, h, h)
        return images
