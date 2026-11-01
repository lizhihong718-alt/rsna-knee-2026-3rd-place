"""Whole-series training bags and end-to-end depth context for the native384 ConvNeXt/DINO route.

Copied from reports/cnx_dino_224_20261009/runtime/src/kneemri/depth224.py, which generalises
reports/resnet_depth_20261008 resnet_depth_e2e.py (ResNet E2E_D, LB .961 vs .959 for its
random-12-window parent; Gold58 .9444) to map backbones. DepthContext, depth_layout and
DepthBags come unchanged from the verbatim resnet_depth.py (sha a8381285...). Changes vs
depth224: DinoPatchBackbone (this route's DINO wrapper) replaces DinoMapBackbone, the
empty-bag map side uses the patch size / ConvNeXt output_stride, and the plane-position
bag variant is dropped.
"""
import random

import numpy as np

from kneemri.coatnet_model import CoAtNetAblationModel, DinoPatchBackbone
from kneemri.resnet_depth import DepthBags, DepthContext

DEPTH_MODEL_KEYS = {'depth_e2e'}
DEPTH_DATA_KEYS = {'depth_canonical_positions', 'series_block_cap'}


def map_side(backbone, cfg):
    resolution = int(cfg.get('resolution', 384))
    if isinstance(backbone, DinoPatchBackbone):
        return resolution // backbone.patch
    return resolution // int(cfg.get('output_stride', 32) or 32)


class DepthStudyModel(CoAtNetAblationModel):
    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        if isinstance(backbone, DinoPatchBackbone) and backbone.gap != 'mean':
            raise ValueError('depth context re-decodes per study; DINO needs dino_gap: mean')
        d = cfg['depth_e2e']
        self.side = map_side(backbone, cfg)
        self.depth = DepthContext(self.feature_dim, width=d.get('width', 32), heads=d.get('heads', 4),
                                  regions=d.get('regions', 2), absolute=d.get('absolute', False),
                                  cross_series=d.get('cross_series', False))

    def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
        if return_backbone_features:
            raise ValueError('feature replay is not supported with depth')
        if x.ndim != 5 or x.shape[2] != 3 or x.shape[1] == 0:
            raise ValueError('images must have shape [B,K,3,H,W] with K > 0')
        b, k = x.shape[:2]
        valid = self._valid((b, k), window_mask, x.device).flatten()
        if not valid.any():
            return x.new_zeros((b, k, self.feature_dim, self.side, self.side))
        maps = self.backbone.forward_features(x.flatten(0, 1)[valid])
        out = maps.new_zeros((b * k, *maps.shape[1:]))
        out[valid] = maps
        return out.view(b, k, *maps.shape[1:])

    def head(self, feats, window_mask=None, group_ids=None, positions=None, series_ids=None,
             position_mask=None, plane_ids=None):
        if feats.ndim == 4:
            return super().head(feats, window_mask, group_ids, positions, series_ids, position_mask, plane_ids)
        if feats.ndim != 5:
            raise ValueError('depth head expects [B,K,C,H,W] feature maps')
        valid = self._valid(feats.shape[:2], window_mask, feats.device)
        rows = []
        for b in range(feats.shape[0]):
            index = valid[b].nonzero().flatten()
            if not len(index):
                rows.append(None)
                continue
            m = self.depth(feats[b, index], series_ids[b, index], positions[b, index, 0],
                           position_mask[b, index])
            rows.append((index, self._spatial_decode(m)))
        reference = next((r[1] for r in rows if r is not None), None)
        dtype = reference.dtype if reference is not None else feats.dtype
        decoded = feats.new_zeros((*valid.shape, self.n, self.feature_dim), dtype=dtype)
        for b, row in enumerate(rows):
            if row is not None:
                decoded[b, row[0]] = row[1]
        return super().head(decoded, window_mask, group_ids, positions, series_ids, position_mask, plane_ids)


class DepthSeriesBags(DepthBags):
    """Training bags of whole series: add random series until ``series_block_cap`` windows."""

    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
        super().__init__(cfg, all_ids, ids, labels, train, uint8_output)
        self.cap = int(cfg.get('series_block_cap', 32))

    def _selected(self, row, budget=None):
        if not self.train or not self.cfg.get('series_block_cap'):
            return super()._selected(row, budget)
        valid = np.flatnonzero(np.asarray(self.masks[row], dtype=bool))
        series = np.asarray(self.metadata['series_index'][row])[valid]
        rng = np.random.default_rng(random.getrandbits(64))
        chosen = []
        for value in rng.permutation(np.unique(series)):
            members = valid[series == value]
            if chosen and len(chosen) + len(members) > self.cap:
                continue
            chosen.extend(members.tolist())
        selected = np.sort(np.asarray(chosen[:self.cap], dtype=np.int64))
        return selected, len(valid), len(selected)


def build_model(cfg, pretrained=False):
    from kneemri import coatnet_model
    base = coatnet_model.build_model(cfg, pretrained=pretrained)
    if not cfg.get('depth_e2e'):
        return base
    # The pretrained backbone module is reused as is; the head is freshly initialised.
    return DepthStudyModel(base.backbone, cfg)
