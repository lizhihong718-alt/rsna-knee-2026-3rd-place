"""Depth-aware second stage for the BiT ResNet study model.

The 2.5D model encodes every three-slice window independently, so windows of one
series never see each other before the study MIL. This module lets windows of the
same physical series exchange information on the final feature maps (before the
disease queries), as lizhihong's Stage4RelRegion hybrid does, with three changes:

* sagittal positions can be replaced by a canonical lateral coordinate
  (``lat_pos``: + is lateral for left and right knees alike); the physical
  coordinate agrees with that direction in only ~53% of sagittal windows;
* the region grid per window is configurable (2x2 in the original);
* an optional absolute (series-centred, canonical) coordinate per token.

The output projection is zero-initialised, so the model starts exactly at its
spatial base. Gold58 is never used here; this file only defines the model.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags


def depth_layout(series, positions, usable, gap_ratio=1.8, min_centres=4):
    """Group one study's windows into contiguous same-series depth segments.

    Arguments are numpy arrays over the study's valid windows. Returns
    ``centre`` [K] (index of the window's distinct physical centre, -1 when the
    window gets no depth context), plus per-centre segment ids and positions.
    A step larger than ``gap_ratio`` x the series' median step is a boundary,
    never an artificial neighbour; segments need ``min_centres`` centres.
    """
    centre = np.full(len(series), -1, np.int64)
    centre_segment, centre_position = [], []
    segment_id = 0
    for value in np.unique(series[usable]):
        if value < 0:
            continue
        members = np.flatnonzero(usable & (series == value))
        if len(members) < min_centres or not np.isfinite(positions[members]).all():
            continue
        unique, inverse = np.unique(positions[members], return_inverse=True)
        if len(unique) < min_centres:
            continue
        gaps = np.diff(unique)
        breaks = np.flatnonzero(gaps > gap_ratio * np.median(gaps)) + 1
        limits = [0, *breaks.tolist(), len(unique)]
        local = np.full(len(unique), -1, np.int64)
        for a, b in zip(limits[:-1], limits[1:]):
            if b - a < min_centres:
                continue
            for u in range(a, b):
                local[u] = len(centre_position)
                centre_segment.append(segment_id)
                centre_position.append(unique[u])
            segment_id += 1
        centre[members] = local[inverse]
    return centre, np.asarray(centre_segment, np.int64), np.asarray(centre_position, np.float32)


class RelativeTokenAttention(nn.Module):
    """Multi-head attention among region tokens with a relative-geometry bias."""

    def __init__(self, width, heads, cross_series=False):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.relative_bias = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads))
        self.cross_series = bool(cross_series)
        if self.cross_series:
            # Other segments start strongly down-weighted; training can open them.
            self.cross_bias = nn.Parameter(torch.full((heads,), -4.0))

    def forward(self, tokens, segment, position, xy):
        t, d = tokens.shape[0], self.width // self.heads
        q, k, v = self.qkv(self.norm(tokens)).view(t, 3, self.heads, d).permute(1, 2, 0, 3)
        geometry = torch.cat([((position[:, None] - position[None, :]) / 32.).clamp(-4, 4)[..., None],
                              xy[:, None, :] - xy[None, :, :]], -1)
        bias = self.relative_bias(geometry.to(tokens.dtype)).permute(2, 0, 1).float()
        same = segment[:, None] == segment[None, :]
        if self.cross_series:
            bias = torch.where(same, bias, self.cross_bias.float()[:, None, None].expand_as(bias))
        else:
            bias = bias.masked_fill(~same, float('-inf'))
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(d) + bias
        weights = scores.softmax(-1).to(v.dtype)
        return self.projection((weights @ v).transpose(0, 1).reshape(t, self.width))


class DepthContext(nn.Module):
    """Zero-initialised residual on [K,C,H,W] maps from other depths of the same series."""

    def __init__(self, dim, width=32, heads=4, regions=2, absolute=False, cross_series=False,
                 gap_ratio=1.8, min_centres=4):
        super().__init__()
        self.dim, self.width, self.regions = dim, width, int(regions)
        self.gap_ratio, self.min_centres = gap_ratio, min_centres
        self.input_norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, width)
        self.absolute = nn.Linear(1, width, bias=False) if absolute else None
        self.attention = RelativeTokenAttention(width, heads, cross_series)
        self.output_projection = nn.Linear(width, dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        axis = 2 * ((torch.arange(self.regions) + .5) / self.regions - .5)
        gy, gx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('region_xy', torch.stack([gx.flatten(), gy.flatten()], -1), persistent=False)

    def forward(self, maps, series, positions, position_mask):
        """One study: maps [K,C,H,W]; series/positions/position_mask [K]."""
        centre, centre_segment, centre_position = depth_layout(
            series.cpu().numpy(), positions.float().cpu().numpy(), position_mask.cpu().numpy(),
            self.gap_ratio, self.min_centres)
        windows = np.flatnonzero(centre >= 0)
        if not len(windows):
            return maps
        device = maps.device
        windows_t = torch.from_numpy(windows).to(device)
        inverse = torch.from_numpy(centre[windows]).to(device)
        n_centres, r = len(centre_position), self.regions
        x = maps.index_select(0, windows_t)
        h, w = x.shape[-2:]
        latent = self.input_projection(self.input_norm(x.permute(0, 2, 3, 1)))
        latent = F.adaptive_avg_pool2d(latent.permute(0, 3, 1, 2), r)
        summed = latent.new_zeros((n_centres, self.width, r, r)).index_add(0, inverse, latent)
        count = torch.bincount(inverse, minlength=n_centres).to(latent.dtype)[:, None, None, None]
        tokens = (summed / count).permute(0, 2, 3, 1).reshape(n_centres * r * r, self.width)
        position = torch.from_numpy(centre_position).to(device).repeat_interleave(r * r)
        segment = torch.from_numpy(centre_segment).to(device).repeat_interleave(r * r)
        xy = self.region_xy.to(device).repeat(n_centres, 1)
        if self.absolute is not None:
            tokens = tokens + self.absolute((position / 100.)[:, None].to(tokens.dtype))
        context = self.attention(tokens, segment, position, xy)
        context = context.view(n_centres, r, r, self.width).permute(0, 3, 1, 2)
        context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=False)
        delta = self.output_projection(context.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return maps.index_add(0, windows_t, delta.index_select(0, inverse).to(maps.dtype))


class DepthStageModel(nn.Module):
    """Frozen BiT prefix, trainable final stage(s)/norm, spatial queries, MIL and depth."""

    def __init__(self, base, depth=None, trainable_stages=(3,)):
        super().__init__()
        self.base = base
        stages = base.backbone.stages
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in base.backbone.modules()):
            raise ValueError('BiT GroupNorm backbone expected')
        self.trainable_stages = sorted(int(i) for i in trainable_stages)
        if self.trainable_stages != list(range(self.trainable_stages[0], len(stages))):
            raise ValueError('trainable stages must be a suffix of the backbone')
        self.first_trainable = self.trainable_stages[0]
        base.backbone.requires_grad_(False)
        for i in self.trainable_stages:
            stages[i].requires_grad_(True)
        base.backbone.norm.requires_grad_(True)
        for name, p in base.named_parameters():
            if not name.startswith('backbone.'):
                p.requires_grad_(True)
        self.depth = depth
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        backbone = self.base.backbone
        backbone.stem.train(False)
        for i, stage in enumerate(backbone.stages):
            stage.train(mode and i >= self.first_trainable)
        return self

    def parameter_groups(self, suffix_lr, spatial_lr, depth_lr):
        suffix, spatial, depth = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('base.backbone.'):
                suffix.append(p)
            elif name.startswith('base.'):
                spatial.append(p)
            else:
                depth.append(p)
        groups = [dict(params=suffix, lr=suffix_lr, role='suffix'),
                  dict(params=spatial, lr=spatial_lr, role='spatial_query_MIL')]
        if depth:
            groups.append(dict(params=depth, lr=depth_lr, role='depth'))
        return groups

    def prefix(self, images):
        backbone = self.base.backbone
        with torch.no_grad():
            x = backbone.stem(images)
            for stage in list(backbone.stages)[:self.first_trainable]:
                x = stage(x)
        return x

    def suffix(self, x):
        backbone = self.base.backbone
        for stage in list(backbone.stages)[self.first_trainable:]:
            x = stage(x)
        return backbone.norm(x)

    def study_logits(self, maps, metadata, use_depth=True):
        """maps [B,K,C,H,W] (zeros for padding) -> logits [B,12]."""
        mask = metadata['window_mask']
        features = maps.new_zeros((*mask.shape, self.base.n, self.base.feature_dim))
        for b in range(maps.shape[0]):
            valid = mask[b].nonzero().flatten()
            if not len(valid):
                continue
            m = maps[b, valid]
            depth = self.depth if self.depth is not None else getattr(self.base, 'depth', None)
            if use_depth and depth is not None:
                m = depth(m, metadata['series_ids'][b, valid], metadata['positions'][b, valid, 0],
                               metadata['position_mask'][b, valid])
            features[b, valid] = self.base._spatial_decode(m).to(features.dtype)
        return self.base.head(features, **metadata)


class DepthBags(CoAtNetOriginalNeighborBags):
    """Original-neighbour bags whose sagittal positions are canonical lateral mm."""

    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
        super().__init__(cfg, all_ids, ids, labels, train, uint8_output)
        self.lat_pos = self.lat_valid = None
        path = cfg.get('depth_canonical_positions')
        if path:
            with np.load(path) as z:
                if z['study_uids'].tolist() != list(all_ids):
                    raise ValueError('orientation rows differ from the corpus')
                self.lat_pos = z['lat_pos'].astype(np.float32)
                self.lat_valid = z['lat_pos_valid'].astype(bool)

    def _item(self, uid, row, selected):
        item = super()._item(uid, row, selected)
        if self.lat_pos is not None and len(selected):
            valid = self.lat_valid[row, selected]
            positions = item['positions'][:, 0].numpy().copy()
            if not np.allclose(np.abs(positions[valid]), np.abs(self.lat_pos[row, selected][valid]), atol=1e-3):
                raise ValueError(f'canonical lateral positions disagree with the corpus for {uid}')
            positions[valid] = self.lat_pos[row, selected][valid]
            item['positions'] = torch.from_numpy(positions[:, None])
        return item
