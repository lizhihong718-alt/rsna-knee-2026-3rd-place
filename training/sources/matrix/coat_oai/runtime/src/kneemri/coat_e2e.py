"""CoAtNet end-to-end depth model with shape-bucketed window encoding.

Whole-series bags give a different number of valid windows every step. On CoAtNet
(bf16, channels_last, depthwise MBConv) cuDNN builds a new execution plan for every
new batch shape: ~15 s for 8 windows and ~127 s for 128 windows on an H20, versus
none with cuDNN disabled (smoke/bench_shapes.py). ResNet BiT is unaffected.

The encoder therefore pads the flattened valid windows up to a multiple of
``encode_bucket`` by repeating valid windows and drops the extra maps. In eval mode
(BatchNorm running statistics, no dropout) this is exact. In training mode only the
batch statistics of the two BatchNorm stages see a few duplicated windows (at most
bucket-1 of ~100); outputs, losses and gradients use the original windows only.
"""
import torch

from kneemri.resnet_depth_e2e import DepthStudyModel

COAT_E2E_MODEL_KEYS = {'encode_bucket'}


class BucketedDepthStudyModel(DepthStudyModel):
    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        self.bucket = int(cfg.get('encode_bucket', 16))
        if self.bucket < 1:
            raise ValueError('encode_bucket must be a positive integer')

    def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
        if return_backbone_features:
            raise ValueError('feature replay is not supported with depth')
        if x.ndim != 5 or x.shape[2] != 3 or x.shape[1] == 0:
            raise ValueError('images must have shape [B,K,3,H,W] with K > 0')
        b, k = x.shape[:2]
        valid = self._valid((b, k), window_mask, x.device).flatten()
        side = x.shape[-1] // 32
        index = valid.nonzero().flatten()
        n = int(index.numel())
        if not n:
            return x.new_zeros((b, k, self.feature_dim, side, side))
        m = -(-n // self.bucket) * self.bucket
        if m > n:
            index = torch.cat([index, index[torch.arange(m - n, device=index.device) % n]])
        maps = self.backbone.forward_features(x.flatten(0, 1)[index])[:n]
        out = maps.new_zeros((b * k, *maps.shape[1:]))
        out[valid] = maps
        return out.view(b, k, *maps.shape[1:])
