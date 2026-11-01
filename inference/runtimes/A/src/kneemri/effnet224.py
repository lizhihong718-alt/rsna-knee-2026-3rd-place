"""EfficientNet (NS B0-B7 / V2) study models at a fixed 224 input, read from native384.

Model: the BiT E2E_D study model (resnet_depth_e2e.DepthStudyModel: whole-series
bags, depth context on the final maps, disease spatial queries + GAP, study MIL)
with the encoder swapped for a timm EfficientNet. EfficientNet ends in
conv_head + BN + SiLU, so ``forward_features`` is a normalised [N,C,7,7] map at
224 exactly like BiT's final norm+ReLU map; nothing else in the head changes.

Pixels: the corpus is ``native384_crop140`` (one DICOM resample to 384 over the
140 mm crop). Every window is reduced 384 -> ``resolution`` ONCE, in uint8, right
after the laterality canonicalisation and before any augmentation, by
``input_downscale``:
  area        cv2.INTER_AREA (box-filter integral, anti-aliased; default);
  bicubic_aa  torch bicubic with antialias=True (sharper pass band, still anti-aliased);
  linear      cv2.INTER_LINEAR without prefilter (aliased; sharpness close to native224).
The same function runs for training, Gold58 evaluation and DICOM inference, so the
three see identical 224 windows. The legacy loader instead resized the augmented
float tensor with non-antialiased bilinear and lost the uint8/LUT transfer path.

Optional keys (defaults = the source recipe): ``drop_path_rate`` (stochastic depth,
training only), ``cudnn_benchmark`` (kernel choice only), and in runtime_b:
  bn_momentum             BatchNorm running-statistics momentum (timm/PyTorch default 0.1:
                          eval stats = EMA over the last ~10 forward batches = ~20 studies);
  backbone_output_stride  16 dilates the last stride-2 stage (same weights): 14x14 final map at 224;
  model_input_size        bilinear resize of the 224 windows inside the model before the encoder
                          (same 224 pixels, finer encoder grid; never used unless declared).
"""
from pathlib import Path
import re

import numpy as np
import torch

from kneemri.coatnet_data import NORMALIZATION_STATS

MODEL_KEYS = {'drop_path_rate', 'bn_momentum', 'backbone_output_stride', 'model_input_size'}
DATA_KEYS = {'input_downscale', 'cudnn_benchmark', 'layer_decay'}


def stage_of(name, depth=8):
    """EfficientNet parameter name -> depth index: stem 0, blocks.i -> i+1, conv_head/bn2 -> depth."""
    if name.startswith(('conv_stem', 'bn1')):
        return 0
    if name.startswith('blocks.'):
        return int(name.split('.')[1]) + 1
    if name.startswith(('conv_head', 'bn2')):
        return depth
    raise ValueError(f'unmapped EfficientNet parameter {name}')


def split_layer_decay(backbone, group, decay):
    """One backbone param group -> per-depth groups with lr * decay**(depth - stage)."""
    depth = len(backbone.blocks) + 1
    names = {id(p): n for n, p in backbone.named_parameters()}
    buckets = {}
    for p in group['params']:
        buckets.setdefault(stage_of(names[id(p)], depth), []).append(p)
    return [dict(group, params=ps, lr=group['lr'] * decay ** (depth - k)) for k, ps in sorted(buckets.items())]
DOWNSCALES = ('area', 'bicubic_aa', 'linear')
ARCH_PATTERN = r'^tf_efficientnet(?:v2)?_(?:b[0-7]|s|m|l|xl)\.'


def downscale(raw, size, mode):
    """uint8 [K,3,H,H] -> uint8 [K,3,size,size]; identity when already ``size``."""
    h = raw.shape[-1]
    if h == size:
        return raw
    if mode not in DOWNSCALES:
        raise ValueError(f'input_downscale must be one of {DOWNSCALES}')
    if size > h:
        raise ValueError('input_downscale only reduces resolution')
    flat = np.ascontiguousarray(raw).reshape(-1, h, h)
    if mode == 'bicubic_aa':
        x = torch.from_numpy(flat).float()[:, None]
        y = torch.nn.functional.interpolate(x, size=(size, size), mode='bicubic', antialias=True,
                                            align_corners=False)
        out = y[:, 0].round_().clamp_(0, 255).to(torch.uint8).numpy()
    else:
        import cv2
        cv2.setNumThreads(1)
        flag = cv2.INTER_AREA if mode == 'area' else cv2.INTER_LINEAR
        out = np.stack([cv2.resize(f, (size, size), interpolation=flag) for f in flat])
    return out.reshape(*raw.shape[:-2], size, size)


class DownscaleMixin:
    """Put in front of an original-neighbour bag class: 384 corpus -> uint8 224 windows."""

    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
        super().__init__(cfg, all_ids, ids, labels, train, uint8_output)
        self.corpus_resolution = self.source_resolution
        self.downscale_mode = cfg.get('input_downscale', 'area')
        if self.downscale_mode not in DOWNSCALES:
            raise ValueError(f'input_downscale must be one of {DOWNSCALES}')
        if self.resolution < self.corpus_resolution:
            # Windows leave _canonical already at the model resolution, so the
            # uint8/LUT transfer path is exact again.
            self.source_resolution = self.resolution
            self.uint8_output = bool(uint8_output)

    def _canonical(self, row, selected, raw):
        return downscale(super()._canonical(row, selected, raw), self.resolution, self.downscale_mode)


def with_downscale(base):
    return type(base.__name__ + 'Downscaled', (DownscaleMixin, base), {})


def build_backbone(cfg):
    import timm

    arch = str(cfg['arch'])
    if not re.match(ARCH_PATTERN, arch):
        raise ValueError(f'EfficientNet route requires a tf_efficientnet(v2) backbone, got {arch}')
    rate = float(cfg.get('drop_path_rate', 0.0))
    if not 0.0 <= rate < 0.6:
        raise ValueError('drop_path_rate must lie in [0, 0.6)')
    stride = cfg.get('backbone_output_stride', 32)
    if stride not in (16, 32) or isinstance(stride, bool):
        raise ValueError('backbone_output_stride must be 16 or 32')
    extra = {} if stride == 32 else dict(output_stride=stride)
    backbone = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg',
                                 drop_path_rate=rate, **extra)
    momentum = cfg.get('bn_momentum')
    if momentum is not None:
        if not 0.0 < float(momentum) <= 0.1:
            raise ValueError('bn_momentum must lie in (0, 0.1]')
        for module in backbone.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = float(momentum)
    stats = NORMALIZATION_STATS.get(cfg.get('normalization'))
    expected = backbone.pretrained_cfg
    if stats is None or not all(torch.allclose(torch.tensor(actual), torch.tensor(expected[key]))
                                for actual, key in zip(stats, ('mean', 'std'))):
        raise ValueError(f'normalization does not match pretrained {arch}')
    if cfg.get('gradient_checkpointing', False):
        raise ValueError('EfficientNet route does not recompute BatchNorm stages')
    return backbone


def load_pretrained(backbone, cfg):
    from safetensors.torch import load_file

    path = Path(cfg['initial_weights'])
    if path.suffix != '.safetensors' or not path.is_file():
        raise ValueError('initial_weights must be a local generic pretrained safetensors file')
    if path.parent.name != str(cfg['arch']):
        raise ValueError('initial_weights directory must be named after the timm arch tag')
    state = load_file(str(path))
    classifier = backbone.pretrained_cfg['classifier']
    allowed = {f'{classifier}.weight', f'{classifier}.bias'}
    extra = set(state) - set(backbone.state_dict())
    if not extra <= allowed:
        raise ValueError(f'unexpected pretrained keys: {sorted(extra)}')
    backbone.load_state_dict({k: v for k, v in state.items() if k not in allowed}, strict=True)


def _encoder_input(cfg):
    size = cfg.get('model_input_size')
    if size is None or size == int(cfg['resolution']):
        return None
    if isinstance(size, bool) or not isinstance(size, int) or size < int(cfg['resolution']) or size % 32:
        raise ValueError('model_input_size must be a multiple of 32 >= resolution')
    return size


def build_model(cfg, pretrained=False):
    from torch.nn import functional as F
    from kneemri.resnet_model import ResNetStudyModel
    from kneemri.resnet_depth_e2e import DepthStudyModel

    backbone = build_backbone(cfg)
    if pretrained:
        load_pretrained(backbone, cfg)
    if not cfg.get('depth_e2e'):
        if _encoder_input(cfg) is not None or cfg.get('backbone_output_stride', 32) != 32:
            raise ValueError('model_input_size / output stride are implemented for the E2E depth model')
        return ResNetStudyModel(backbone, cfg)

    class EffNetDepthStudyModel(DepthStudyModel):
        """DepthStudyModel.encode with an optional in-model resize and stride-aware empty maps."""
        input_size = _encoder_input(cfg)
        stride = int(cfg.get('backbone_output_stride', 32))

        def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
            if return_backbone_features:
                raise ValueError('feature replay is not supported with depth')
            if x.ndim != 5 or x.shape[2] != 3 or x.shape[1] == 0:
                raise ValueError('images must have shape [B,K,3,H,W] with K > 0')
            b, k = x.shape[:2]
            valid = self._valid((b, k), window_mask, x.device).flatten()
            side = -(-(self.input_size or x.shape[-1]) // self.stride)
            if not valid.any():
                return x.new_zeros((b, k, self.feature_dim, side, side))
            images = x.flatten(0, 1)[valid]
            if self.input_size is not None:
                images = F.interpolate(images, size=(self.input_size,) * 2, mode='bilinear', align_corners=False)
            maps = self.backbone.forward_features(images)
            out = maps.new_zeros((b * k, *maps.shape[1:]))
            out[valid] = maps
            return out.view(b, k, *maps.shape[1:])

    return EffNetDepthStudyModel(backbone, cfg)
