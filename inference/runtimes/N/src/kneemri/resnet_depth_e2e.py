"""End-to-end depth-aware ResNet study model (stage 1, trained from ImageNet weights).

The late depth adapter (resnet_depth.DepthStageModel) learned a residual of ~0.5%
of the map norm and changed no ranking. Here the same DepthContext sits between
the BiT final maps and the spatial queries from the first update, so the encoder
can learn features that are useful across depth. Training draws whole series
(all their windows) instead of 12 scattered windows, so depth segments exist in
every training bag; evaluation and inference keep every window.
"""
import random

import numpy as np
import torch

from kneemri.resnet_depth import DepthBags, DepthContext
from kneemri.resnet_model import ResNetStudyModel

E2E_MODEL_KEYS = {'depth_e2e'}
E2E_DATA_KEYS = {'depth_canonical_positions', 'series_block_cap'}


class DepthStudyModel(ResNetStudyModel):
    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        d = cfg['depth_e2e']
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
            side = x.shape[-1] // int(self.backbone.feature_info[-1]['reduction'])
            return x.new_zeros((b, k, self.feature_dim, side, side))
        maps = self.backbone.forward_features(x.flatten(0, 1)[valid])
        out = maps.new_zeros((b * k, *maps.shape[1:]))
        out[valid] = maps
        return out.view(b, k, *maps.shape[1:])

    def head(self, feats, window_mask=None, group_ids=None, positions=None, series_ids=None,
             position_mask=None, plane_ids=None):
        if feats.ndim == 4:
            # Already depth-contextualised and spatially decoded (stage-2 wrapper path).
            return super().head(feats, window_mask, group_ids, positions, series_ids, position_mask, plane_ids)
        if feats.ndim != 5:
            raise ValueError('depth head expects [B,K,C,H,W] feature maps')
        valid = self._valid(feats.shape[:2], window_mask, feats.device)
        decoded = None
        rows = []
        for b in range(feats.shape[0]):
            index = valid[b].nonzero().flatten()
            if not len(index):
                rows.append(None)
                continue
            m = self.depth(feats[b, index], series_ids[b, index], positions[b, index, 0],
                           position_mask[b, index])
            rows.append((index, self._spatial_decode(m)))
        reference = next(r[1] for r in rows if r is not None) if any(r is not None for r in rows) else None
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
        if not self.train:
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
    from kneemri import resnet_model
    base = resnet_model.build_model(cfg, pretrained=pretrained)
    model = DepthStudyModel(base.backbone, cfg)
    if pretrained:
        model.backbone.load_state_dict(base.backbone.state_dict(), strict=True)
    return model
