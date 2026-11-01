"""Whole-series training bags and end-to-end depth context for any map backbone.

Generalises reports/resnet_depth_20261008 resnet_depth_e2e.py (ResNet E2E_D, LB .961
vs .959 for its random-12-window parent) to the ConvNeXt and DINO backbones of this
route. DepthContext, depth_layout and DepthBags are imported unchanged from the
verbatim copy src/kneemri/resnet_depth.py. Only two things differ from the source:
the empty-bag map side is taken from the backbone stride instead of a fixed 32, and
the model class is the shared CoAtNetAblationModel (ResNetStudyModel is an alias of it).
"""
import random

import numpy as np
import torch

from kneemri.coatnet_model import CoAtNetAblationModel, DinoMapBackbone
from kneemri.resnet_depth import DepthBags, DepthContext

DEPTH_MODEL_KEYS = {'depth_e2e'}
DEPTH_DATA_KEYS = {'depth_canonical_positions', 'series_block_cap', 'plane_positions'}


def map_side(backbone, cfg):
    if isinstance(backbone, DinoMapBackbone):
        return backbone.grid
    return int(cfg.get('resolution', 224)) // int(cfg.get('output_stride', 32) or 32)


class DepthStudyModel(CoAtNetAblationModel):
    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        if isinstance(backbone, DinoMapBackbone) and backbone.gap != 'mean':
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
        images = x.flatten(0, 1)[valid]
        if getattr(self, 'input_blur', None) is not None:
            images = self.input_blur(images)
        maps = self.backbone.forward_features(images)
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
            m = feats[b, index]
            normed = getattr(self, 'map_norm', None) is not None
            if normed:
                # As in BiT, depth context reads maps that already end in GroupNorm+activation.
                m = self.map_norm(m)
            m = self.depth(m, series_ids[b, index], positions[b, index, 0], position_mask[b, index])
            rows.append((index, self._spatial_decode(m, map_normed=normed)))
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
            # Canonical-position-only configs keep the random window sampling.
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


class PlanePositionBags(DepthSeriesBags):
    """Positions as [K,3]: one channel per plane (sagittal canonical lateral mm, coronal and
    axial series-relative mm, whose directions are uniform in this corpus); other channels 0.

    A 2.5D sagittal window cannot tell a medial from a lateral slice by itself; this hands
    the MIL head the slice's canonical medio-lateral coordinate (the depth route only used
    it inside DepthContext). Channel 0 equals the scalar position DepthContext reads.
    """

    def _item(self, uid, row, selected):
        item = super()._item(uid, row, selected)
        planes = item['plane_ids']
        scalar = item['positions'][:, :1]
        expanded = torch.zeros((len(planes), 3), dtype=scalar.dtype)
        for plane in range(3):
            rows = planes == plane
            expanded[rows, plane] = scalar[rows, 0]
        item['positions'] = expanded
        return item


def build_model(cfg, pretrained=False):
    from kneemri import coatnet_model
    base = coatnet_model.build_model(cfg, pretrained=pretrained)
    if not cfg.get('depth_e2e'):
        return base
    # The pretrained backbone module is reused as is; the head is freshly initialised.
    return DepthStudyModel(base.backbone, cfg)
