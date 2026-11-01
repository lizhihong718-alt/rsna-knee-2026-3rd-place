"""Single shared ResNet encoder with disease-specific spatial and study MIL.

The existing, backbone-independent MIL implementation is reused verbatim. There
is one encoder, one set of twelve heads, and one forward pass per MRI window;
the attention across a study's windows is part of the model, not an ensemble.
"""
from pathlib import Path
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from kneemri.coatnet_data import NORMALIZATION_STATS
from kneemri.coatnet_model import CoAtNetAblationModel


class ResNetStudyModel(CoAtNetAblationModel):
    """ResNet 2.5D features -> spatial queries + GAP -> masked study attention."""


PYRAMID_CONFIG_KEYS = {'resnet_pyramid', 'pyramid_width', 'pyramid_attention_dim',
                       'pyramid_gem_p'}


def register_config(trainer):
    """Register only in ResNet processes; leave the shared CoAtNet trainer alone."""
    trainer.MODEL_KEYS.update(PYRAMID_CONFIG_KEYS)


class ResidualPyramidPool(nn.Module):
    """C3/C4/C5 -> stride-8 gated spatial values + GeM -> zero-init residual.

    GroupNorm keeps intermediate pre-activation BiT maps on comparable scales
    without introducing running statistics dependent on MRI-window sampling.
    All scales belong to a single encoder and produce one feature residual.
    """

    def __init__(self, channels, out_dim, n_labels, width=256, hidden=128, gem_p=3.):
        super().__init__()
        if width < 32 or width % 32 or hidden < 1 or not 1 < gem_p < 4:
            raise ValueError('pyramid requires width divisible by 32, hidden > 0, 1 < GeM p < 4')
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, width, 1, bias=False),
                          nn.GroupNorm(32, width), nn.SiLU()) for c in channels])
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.GroupNorm(32, width), nn.ReLU())
        self.attn_value = nn.Conv2d(width, hidden, 1)
        self.attn_gate = nn.Conv2d(width, hidden, 1)
        self.attn_out = nn.Conv2d(hidden, n_labels, 1, bias=False)
        probability = (gem_p - 1.) / 3.
        self.gem_logit = nn.Parameter(torch.tensor(math.log(probability / (1 - probability))))
        self.residual_out = nn.Linear(2 * width, out_dim)
        # At initialization the model is exactly the existing C5 spatial+GAP
        # baseline in eval mode. The adapter opens gradually under supervision.
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def forward(self, maps):
        c3, c4, c5 = maps
        p5 = self.lateral[2](c5)
        p4 = (self.lateral[1](c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)) / math.sqrt(2.)
        p3 = (self.lateral[0](c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)) / math.sqrt(2.)
        features = self.refine(p3)
        gated = torch.tanh(self.attn_value(features)) * torch.sigmoid(self.attn_gate(features))
        raw = self.attn_out(gated).flatten(2)
        # Accumulate pooling in FP32; exponentiation can overflow FP16 even
        # when the final GeM value is moderate.
        with torch.autocast(device_type=features.device.type, enabled=False):
            attention = torch.softmax(4. * torch.tanh(raw.float() / 4.), dim=-1)
            tokens = features.float().flatten(2)
            local = torch.einsum('nlp,ndp->nld', attention, tokens)
            power = 1. + 3. * self.gem_logit.float().sigmoid()
            gem = tokens.clamp_min(1e-6).pow(power).mean(-1).pow(1. / power)
            combined = torch.cat([local, gem[:, None].expand(-1, local.shape[1], -1)], -1)
        return self.residual_out(combined.to(features.dtype))


class ResNetPyramidStudyModel(ResNetStudyModel):
    """Preserve C5 semantic head and add CNN-local multiscale evidence."""

    def __init__(self, backbone, cfg):
        if (not cfg.get('spatial_query', False) or cfg.get('spatial_residual_gated', False)
                or not cfg.get('spatial_query_gap', True) or cfg.get('rank_weight', 0)):
            raise ValueError('pyramid requires spatial_query+GAP and no feature-replay rank loss')
        super().__init__(backbone, cfg)
        info = backbone.feature_info
        selected = [info[i] for i in (2, 3, 4)]
        if [x['reduction'] for x in selected] != [8, 16, 32]:
            raise ValueError('pyramid expects C3/C4/C5 reductions 8/16/32')
        # Name also selects the adapter's own learning rate in the shared trainer.
        self.spatial_pool = ResidualPyramidPool(
            [x['num_chs'] for x in selected], self.feature_dim, self.n,
            width=int(cfg.get('pyramid_width', 256)),
            hidden=int(cfg.get('pyramid_attention_dim', 128)),
            gem_p=float(cfg.get('pyramid_gem_p', 3.)))

    def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
        if return_backbone_features:
            raise ValueError('pyramid feature replay is not supported')
        if x.ndim != 5 or x.shape[2] != 3 or x.shape[1] == 0:
            raise ValueError('images must have shape [B,K,3,H,W] with K > 0')
        b, k = x.shape[:2]
        valid = self._valid((b, k), window_mask, x.device).flatten()
        if not valid.any():
            return x.new_zeros((b, k, self.n, self.feature_dim))
        images = x.flatten(0, 1)[valid]
        final, intermediate = self.backbone.forward_intermediates(
            images, indices=[2, 3, 4], norm=False)
        # BiT intermediate C5 is pre-activation; use the actual pretrained
        # final normalization for BOTH the original semantic path and pyramid.
        encoded = self._spatial_decode(final) + self.spatial_pool(
            [intermediate[0], intermediate[1], final])
        output = encoded.new_zeros((b * k, self.n, self.feature_dim))
        output[valid] = encoded
        return output.view(b, k, self.n, self.feature_dim)


def build_model(cfg, pretrained=False):
    import timm

    arch = str(cfg['arch'])
    if not re.match(r'^resnet(?:\d|v2_)', arch):
        raise ValueError('ResNet route requires a ResNet or ResNetV2 backbone')
    # ResNets are fully convolutional: unlike CoAtNet, do not pass img_size.
    radimagenet = arch == 'resnet50.radimagenet'
    backbone = timm.create_model('resnet50' if radimagenet else arch, pretrained=False, num_classes=0,
                                 in_chans=3, global_pool='avg')
    if radimagenet:
        # The authors' PyTorch checkpoint uses torchvision's ResNet50 v1.5,
        # matching timm resnet50. Their example scales pixels into [-1, 1].
        backbone.pretrained_cfg = dict(backbone.pretrained_cfg, mean=(.5, .5, .5),
                                       std=(.5, .5, .5), tag='radimagenet',
                                       origin_url='https://github.com/BMEII-AI/RadImageNet')
    stats = NORMALIZATION_STATS.get(cfg.get('normalization'))
    expected = backbone.pretrained_cfg
    if stats is None or not all(torch.allclose(torch.tensor(actual), torch.tensor(expected[key]))
                                for actual, key in zip(stats, ('mean', 'std'))):
        raise ValueError(f'normalization does not match pretrained {arch}')
    if cfg.get('gradient_checkpointing', False):
        has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                     for m in backbone.modules())
        if has_bn and not cfg.get('freeze_bn', False):
            raise ValueError('ResNet checkpoint recomputation requires frozen BatchNorm statistics')
        backbone.set_grad_checkpointing(True)
    pyramid = cfg.get('resnet_pyramid', False)
    if not isinstance(pyramid, bool):
        raise ValueError('resnet_pyramid must be boolean')
    if pyramid and not arch.startswith('resnetv2_50x1_bit.'):
        raise ValueError('the audited pyramid recipe requires a BiT ResNetV2-50x1')
    model = (ResNetPyramidStudyModel if pyramid else ResNetStudyModel)(backbone, cfg)
    if pretrained:
        from safetensors.torch import load_file

        path = Path(cfg['initial_weights'])
        if path.suffix != '.safetensors' or not path.is_file():
            raise ValueError('initial_weights must be a local generic pretrained safetensors file')
        state = load_file(str(path))
        classifier = backbone.pretrained_cfg['classifier']
        # RadImageNet's official release contains the backbone only. Do not
        # manufacture random classifier tensors merely to satisfy an ImageNet schema.
        allowed = set() if radimagenet else {f'{classifier}.weight', f'{classifier}.bias'}
        extra = set(state) - set(backbone.state_dict())
        if extra != allowed:
            raise ValueError(f'unexpected pretrained classifier keys: {sorted(extra)}')
        state = {name: value for name, value in state.items() if name not in allowed}
        backbone.load_state_dict(state, strict=True)
    return model
