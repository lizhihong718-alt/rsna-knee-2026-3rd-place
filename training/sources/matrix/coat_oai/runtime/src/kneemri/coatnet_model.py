"""Configurable CoAtNet study heads for controlled image-only ablations.

The default head keeps the released Raptor ``backbone/norm/att/clsW/clsb``
state dictionary contract. The pinned competition submission is never modified.
Masks are explicit: True means a valid window, and padding is not encoded by the
backbone. Metadata is opt-in and must describe the actual acquisition.
"""
from collections.abc import Mapping
from pathlib import Path
import math

import torch
from torch import nn
import torch.nn.functional as F


def _get(cfg, name, default=None):
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _masked_softmax(scores, valid, dim=1):
    """FP32 attention, including a finite zero result for an entirely empty bag."""
    mask = valid
    while mask.ndim < scores.ndim:
        mask = mask.unsqueeze(-1)
    scores = scores.float().masked_fill(~mask, -torch.inf)
    any_valid = mask.any(dim=dim, keepdim=True)
    scores = torch.where(any_valid, scores, torch.zeros_like(scores))
    return torch.softmax(scores, dim=dim).masked_fill(~mask, 0.0)


class GatedAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, classes, dropout):
        super().__init__()
        self.v = nn.Linear(feature_dim, hidden_dim)
        self.u = nn.Linear(feature_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, classes)

    def forward(self, x):
        return self.out(self.drop(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))))


class ResidualGatedSpatialPool(nn.Module):
    """Released bounded disease-specific spatial pool with a protected GAP path."""

    def __init__(self, feature_dim, n_labels, hidden_dim, num_slots, *,
                 logit_cap=2.0, residual_mix=0.5, slot_conditioned=True):
        super().__init__()
        if min(feature_dim, n_labels, hidden_dim, num_slots) < 1:
            raise ValueError('residual spatial dimensions must be positive')
        if not math.isfinite(logit_cap) or logit_cap <= 0:
            raise ValueError('spatial_logit_cap must be finite and positive')
        if not math.isfinite(residual_mix) or not 0 <= residual_mix <= 1:
            raise ValueError('spatial_residual_mix must lie in [0,1]')
        self.feature_dim = int(feature_dim)
        self.n_labels = int(n_labels)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.logit_cap = float(logit_cap)
        self.residual_mix = float(residual_mix)
        self.value_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.gate_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.label_query = nn.Parameter(torch.zeros(self.n_labels, self.hidden_dim))
        if slot_conditioned:
            self.slot_query = nn.Embedding(
                self.num_slots, self.n_labels * self.hidden_dim)
            nn.init.zeros_(self.slot_query.weight)
        else:
            self.register_module('slot_query', None)

    def maximum_attention_weight(self, patches):
        if patches < 1:
            raise ValueError('patch count must be positive')
        numerator = math.exp(self.logit_cap)
        return numerator / (numerator + (patches - 1) * math.exp(-self.logit_cap))

    def maximum_effective_patch_weight(self, patches):
        attentive = self.maximum_attention_weight(patches)
        return (1 - self.residual_mix) / patches + self.residual_mix * attentive

    def forward(self, feature_map, slot_ids, return_attention=False):
        if feature_map.ndim != 4:
            raise ValueError(f'expected a 4D feature map, got {feature_map.shape}')
        if feature_map.shape[1] == self.feature_dim:
            tokens = feature_map.flatten(2).transpose(1, 2)
        elif feature_map.shape[-1] == self.feature_dim:
            tokens = feature_map.flatten(1, 2)
        else:
            raise ValueError('cannot identify the feature axis in CoAtNet spatial output')
        rows = int(tokens.shape[0])
        if slot_ids.shape != (rows,) or slot_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('slot IDs must be integer and align with spatial feature maps')
        if bool(((slot_ids < 0) | (slot_ids >= self.num_slots)).any()):
            raise ValueError('valid spatial rows require configured anatomical slot IDs')

        normalized = F.layer_norm(tokens.float(), (self.feature_dim,))
        gated = torch.tanh(self.value_projection(normalized)) * torch.sigmoid(
            self.gate_projection(normalized))
        query = self.label_query.unsqueeze(0).expand(rows, -1, -1)
        if self.slot_query is not None:
            query = query + self.slot_query(slot_ids).view(
                rows, self.n_labels, self.hidden_dim)
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            raw = torch.einsum('nph,nlh->npl', gated.float(), query.float())
            raw = raw / math.sqrt(self.hidden_dim)
            logits = self.logit_cap * torch.tanh(raw / self.logit_cap)
            attention = torch.softmax(logits, dim=1)
        global_value = tokens.mean(dim=1, keepdim=True)
        attentive_value = torch.einsum(
            'npl,npf->nlf', attention.to(tokens.dtype), tokens)
        values = global_value + self.residual_mix * (attentive_value - global_value)
        return (values, attention) if return_attention else values


class CoAtNetAblationModel(nn.Module):
    """Window feature aggregation with optional metadata and spatial queries.

    Input images have shape [B,K,3,H,W]. ``group_ids`` is [B,K], in
    [-1,num_groups): -1 explicitly represents unknown or mixed actual identity.
    Positions are [B,K,position_dim] in millimetres relative to the corresponding
    series centre; they are scaled by position_scale_mm before the MLP. A scalar
    position dimension also accepts [B,K]. Hierarchical aggregation defaults to
    the independent ``series_ids`` [B,K] field, a per-study integer identity;
    ``hierarchy_axis='group'`` explicitly groups by actual acquisition attributes
    instead. It never infers series identity from concatenated window order.
    """
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(backbone.num_features)
        self.n = int(_get(cfg, 'num_classes', 12))
        self.aggregate = str(_get(cfg, 'aggregate', 'attention'))
        self.hidden_dim = int(_get(cfg, 'attention_hidden', 256))
        self.dropout = float(_get(cfg, 'head_dropout', _get(cfg, 'dropout', 0.2)))
        self.temperature = float(_get(cfg, 'attention_temperature', _get(cfg, 'temperature', 1.0)))
        self.value_norm = str(_get(cfg, 'value_norm', 'normalized'))
        self.use_groups = bool(_get(cfg, 'group_embedding', False))
        self.use_positions = bool(_get(cfg, 'position_embedding', False))
        self.hierarchical = bool(_get(cfg, 'hierarchical', False))
        self.hierarchy_axis = str(_get(cfg, 'hierarchy_axis', 'series'))
        self.spatial_query = bool(_get(cfg, 'spatial_query', False))
        self.spatial_residual_gated = bool(_get(cfg, 'spatial_residual_gated', False))
        self.spatial_query_gap = bool(_get(cfg, 'spatial_query_gap', True))
        self.num_groups = int(_get(cfg, 'num_groups', 5))
        self.position_dim = int(_get(cfg, 'position_dim', 1))
        self.position_scale_mm = float(_get(cfg, 'position_scale_mm', 100.0))
        self.slot_embedding_dim = int(_get(cfg, 'slot_embedding_dim', 32))
        self.feature_projection_dim = int(_get(cfg, 'feature_projection_dim', 256))
        self.pool_mix_attention_init = float(_get(cfg, 'pool_mix_attention_init', 0.75))
        self.label_mixing = bool(_get(cfg, 'label_mixing', False))
        self.label_mixing_cap = float(_get(cfg, 'label_mixing_cap', 2.0))
        if self.aggregate not in {'attention', 'gated', 'gated_mean_mix', 'mean', 'max', 'mean_max'}:
            raise ValueError(f'unsupported aggregate: {self.aggregate}')
        if self.spatial_query and self.spatial_residual_gated:
            raise ValueError('spatial_query and spatial_residual_gated are mutually exclusive')
        if self.value_norm not in {'normalized', 'raw'}:
            raise ValueError('value_norm must be normalized or raw')
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError('attention_temperature must be finite and positive')
        if self.hierarchical and self.aggregate not in {'attention', 'gated'}:
            raise ValueError('hierarchical supports attention or gated only')
        if self.hierarchy_axis not in {'series', 'group', 'plane'}:
            raise ValueError('hierarchy_axis must be series, group or plane')
        if self.n != 12 or self.num_groups < 1 or self.position_dim < 1:
            raise ValueError('expected 12 labels and positive metadata dimensions')
        if self.position_scale_mm <= 0 or not math.isfinite(self.position_scale_mm):
            raise ValueError('position_scale_mm must be finite and positive')
        if self.label_mixing and (self.label_mixing_cap <= 0 or not math.isfinite(self.label_mixing_cap)):
            raise ValueError('label_mixing_cap must be finite and positive')
        if self.aggregate == 'gated_mean_mix':
            if self.slot_embedding_dim < 1 or self.feature_projection_dim < 1:
                raise ValueError('slot and feature projection dimensions must be positive')
            if not 0 < self.pool_mix_attention_init < 1 or not math.isfinite(self.pool_mix_attention_init):
                raise ValueError('pool_mix_attention_init must lie strictly between zero and one')
            if self.use_groups:
                raise ValueError('gated_mean_mix includes slot conditioning; group_embedding must be false')
            if self.value_norm != 'normalized':
                raise ValueError('gated_mean_mix uses its projected LayerNorm values')
        d = self.feature_dim
        if self.aggregate == 'gated_mean_mix':
            # Match the released ResNet head contract while keeping the image
            # encoder replaceable: concatenate a compact slot code, then project.
            pd = self.feature_projection_dim
            self.slot_embedding = nn.Embedding(self.num_groups, self.slot_embedding_dim)
            self.feature_proj = nn.Sequential(
                nn.Linear(d + self.slot_embedding_dim, pd), nn.LayerNorm(pd))
            self.attn_a = nn.Linear(pd, self.hidden_dim)
            self.attn_b = nn.Linear(pd, self.hidden_dim)
            self.attn_out = nn.Linear(self.hidden_dim, self.n, bias=False)
            initial_logit = math.log(self.pool_mix_attention_init / (1 - self.pool_mix_attention_init))
            self.pool_mix_logit = nn.Parameter(torch.full((self.n,), initial_logit))
            self.classifier_weight = nn.Parameter(torch.empty(self.n, pd))
            self.classifier_bias = nn.Parameter(torch.zeros(self.n))
            nn.init.trunc_normal_(self.classifier_weight, std=0.02)
        else:
            self.norm = nn.LayerNorm(d)
        if self.aggregate == 'attention':
            # Preserve the ep7 parameter names and the exact unmasked FP32 path.
            self.att = nn.Sequential(nn.Linear(d, self.hidden_dim), nn.Tanh(),
                                     nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, self.n))
        elif self.aggregate == 'gated':
            self.att = GatedAttention(d, self.hidden_dim, self.n, self.dropout)
        if self.aggregate != 'gated_mean_mix':
            out_dim = 2 * d if self.aggregate == 'mean_max' else d
            self.clsW = nn.Parameter(torch.zeros(self.n, out_dim))
            self.clsb = nn.Parameter(torch.zeros(self.n))
            nn.init.trunc_normal_(self.clsW, std=0.02)
        if self.use_groups:
            self.group_embed = nn.Embedding(self.num_groups + 1, d)
            if self.spatial_residual_gated:
                nn.init.zeros_(self.group_embed.weight)
            else:
                nn.init.normal_(self.group_embed.weight, std=0.02)
        if self.use_positions:
            self.position_mlp = nn.Sequential(nn.Linear(self.position_dim, 64), nn.Tanh(), nn.Linear(64, d))
        if self.hierarchical:
            self.group_att = nn.Sequential(nn.Linear(d, self.hidden_dim), nn.Tanh(),
                                           nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, self.n))
        if self.spatial_query:
            qd = int(_get(cfg, 'spatial_query_dim', 256))
            if qd < 1:
                raise ValueError('spatial_query_dim must be positive')
            self.spatial_dim = qd
            self.spatial_norm = nn.LayerNorm(d)
            self.spatial_keys = nn.Linear(d, qd)
            self.spatial_values = nn.Linear(d, qd)
            self.spatial_queries = nn.Parameter(torch.empty(self.n, qd))
            self.spatial_out = nn.Linear(qd, d)
            nn.init.trunc_normal_(self.spatial_queries, std=0.02)
        if self.spatial_residual_gated:
            self.spatial_pool = ResidualGatedSpatialPool(
                d, self.n, int(_get(cfg, 'spatial_hidden_dim', 128)), self.num_groups,
                logit_cap=float(_get(cfg, 'spatial_logit_cap', 2.0)),
                residual_mix=float(_get(cfg, 'spatial_residual_mix', 0.5)),
                slot_conditioned=bool(_get(cfg, 'spatial_slot_conditioned', True)))
        if self.label_mixing:
            self.label_mix_logits = nn.Parameter(torch.zeros(self.n, self.n))
            self.register_buffer('label_mix_mask', 1 - torch.eye(self.n), persistent=False)

    @staticmethod
    def _valid(shape, mask, device):
        if mask is None:
            return torch.ones(shape, device=device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(shape) or mask.dtype != torch.bool:
            raise ValueError('window_mask must be bool [B,K], with True for valid')
        return mask.to(device=device)

    def _groups(self, group_ids, valid):
        if group_ids is None:
            raise ValueError('group_ids are required by the selected metadata/hierarchy configuration')
        if group_ids.shape != valid.shape or group_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('group_ids must be integer [B,K]')
        group_ids = group_ids.to(valid.device)
        group_ids = torch.where(valid, group_ids, torch.full_like(group_ids, -1))
        if ((group_ids < -1) | (group_ids >= self.num_groups)).any():
            raise ValueError('actual group ID exceeds configured group vocabulary')
        return torch.where(group_ids >= 0, group_ids, self.num_groups).long()

    def _slots(self, group_ids, valid):
        """Strict public-corpus slot IDs; padding is mapped only after validation."""
        if group_ids is None:
            raise ValueError('group_ids are required for gated_mean_mix slot conditioning')
        if group_ids.shape != valid.shape or group_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('group_ids must be integer [B,K]')
        group_ids = group_ids.to(valid.device)
        invalid = valid & ((group_ids < 0) | (group_ids >= self.num_groups))
        if invalid.any():
            raise ValueError('valid window has no configured public slot ID')
        return torch.where(valid, group_ids, torch.zeros_like(group_ids)).long()

    @staticmethod
    def _series(series_ids, valid):
        if series_ids is None:
            raise ValueError('series_ids are required for hierarchical series aggregation')
        if series_ids.shape != valid.shape or series_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('series_ids must be integer [B,K], using per-study local identity')
        series_ids = series_ids.to(valid.device)
        series_ids = torch.where(valid, series_ids, torch.full_like(series_ids, -1))
        if (series_ids < -1).any():
            raise ValueError('series_ids must be nonnegative, or -1 for unknown/mixed identity')
        return series_ids.long()

    @staticmethod
    def _planes(plane_ids, valid):
        """Coarse first-level pooling identity: 0 sagittal, 1 coronal, 2 axial, -1 unknown."""
        if plane_ids is None:
            raise ValueError('plane_ids are required for plane hierarchy aggregation')
        if plane_ids.shape != valid.shape or plane_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('plane_ids must be integer [B,K]')
        plane_ids = plane_ids.to(valid.device)
        plane_ids = torch.where(valid, plane_ids, torch.full_like(plane_ids, -1))
        if (plane_ids < -1).any():
            raise ValueError('plane_ids must be nonnegative, or -1 for unknown/mixed identity')
        return plane_ids.long()

    def _pretrained_head_norm(self, values):
        head = getattr(self.backbone, 'head', None)
        norm = getattr(head, 'norm', None)
        weight = getattr(norm, 'weight', None)
        if weight is None:
            return values
        return F.layer_norm(values, (self.feature_dim,), weight=weight,
                            bias=getattr(norm, 'bias', None), eps=float(getattr(norm, 'eps', 1e-6)))

    def _spatial_decode(self, maps, slot_ids=None):
        if not isinstance(maps, torch.Tensor) or maps.ndim != 4:
            raise ValueError('spatial encoding requires a 4D CoAtNet feature map')
        if self.spatial_residual_gated:
            if slot_ids is None:
                raise ValueError('spatial_residual_gated requires anatomical slot IDs')
            return self._pretrained_head_norm(self.spatial_pool(maps, slot_ids))
        if maps.shape[1] == self.feature_dim:
            tokens = maps.flatten(2).transpose(1, 2)  # actual timm CoAtNet NCHW
        elif maps.shape[-1] == self.feature_dim:
            tokens = maps.flatten(1, 2)  # explicit support for NHWC variants
        else:
            raise ValueError('cannot identify the feature axis in CoAtNet spatial output')
        normalized = self.spatial_norm(tokens)
        keys = self.spatial_keys(normalized)
        values = self.spatial_values(normalized)
        scores = torch.einsum('qd,npd->nqp', self.spatial_queries.float(), keys.float()) / math.sqrt(self.spatial_dim)
        attention = torch.softmax(scores, dim=-1)
        local = torch.einsum('nqp,npd->nqd', attention, values.float()).to(values.dtype)
        local = self.spatial_out(local)
        if self.spatial_query_gap:
            # Use the backbone's actual head (including pre-logit transforms).
            gap = self.backbone.forward_head(maps)
            if gap.shape != (maps.shape[0], self.feature_dim):
                raise ValueError('unexpected CoAtNet pooled feature shape')
            local = local + gap.unsqueeze(1)
        return local

    def _spatial_encode(self, images, slot_ids=None):
        return self._spatial_decode(self.backbone.forward_features(images), slot_ids)

    def _spatial_slot_ids(self, group_ids, valid):
        if (group_ids is None or group_ids.shape != valid.shape
                or group_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('spatial_residual_gated requires integer group_ids [B,K]')
        group_ids = group_ids.to(valid.device)
        invalid = valid & ((group_ids < 0) | (group_ids >= self.num_groups))
        if invalid.any():
            raise ValueError('valid window has no configured spatial slot ID')
        return group_ids.flatten()[valid.flatten()].long()

    def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
        if x.ndim != 5 or x.shape[2] != 3 or x.shape[1] == 0:
            raise ValueError('images must have shape [B,K,3,H,W] with K > 0')
        b, k = x.shape[:2]
        valid = self._valid((b, k), window_mask, x.device)
        flat_valid = valid.flatten()
        spatial = self.spatial_query or self.spatial_residual_gated
        feature_shape = (self.n, self.feature_dim) if spatial else (self.feature_dim,)
        if not flat_valid.any():
            output = x.new_zeros((b, k, *feature_shape))
            if return_backbone_features:
                if not self.spatial_residual_gated:
                    raise ValueError('backbone feature replay requires spatial_residual_gated')
                return output, None
            return output
        # Padding is never sent through BN and cannot poison valid images with NaN.
        images = x.flatten(0, 1)[flat_valid]
        slot_ids = None
        if self.spatial_residual_gated:
            slot_ids = self._spatial_slot_ids(group_ids, valid)
        maps = self.backbone.forward_features(images) if spatial else None
        encoded = self._spatial_decode(maps, slot_ids) if spatial else self.backbone(images)
        if tuple(encoded.shape[1:]) != feature_shape:
            raise ValueError(f'backbone feature shape {encoded.shape} disagrees with {feature_shape}')
        output = encoded.new_zeros((b * k, *feature_shape))
        output[flat_valid] = encoded
        output = output.view(b, k, *feature_shape)
        if return_backbone_features:
            if not self.spatial_residual_gated:
                raise ValueError('backbone feature replay requires spatial_residual_gated')
            return output, maps
        return output

    def decode_backbone_features(self, feature_maps, window_mask, group_ids):
        """Replay the trainable spatial/head path from detached CoAtNet maps."""
        if not self.spatial_residual_gated:
            raise ValueError('backbone feature replay requires spatial_residual_gated')
        if window_mask.ndim != 2:
            raise ValueError('replay window_mask must have shape [B,K]')
        valid = self._valid(window_mask.shape, window_mask, window_mask.device)
        rows = int(valid.sum())
        if rows == 0:
            if feature_maps is not None:
                raise ValueError('empty replay bags must not carry feature maps')
            return self.clsW.new_zeros((*valid.shape, self.n, self.feature_dim))
        if not isinstance(feature_maps, torch.Tensor) or feature_maps.ndim != 4 or len(feature_maps) != rows:
            raise ValueError('replay feature maps must align with valid windows')
        slot_ids = self._spatial_slot_ids(group_ids, valid)
        encoded = self._spatial_decode(feature_maps, slot_ids)
        output = encoded.new_zeros((valid.numel(), self.n, self.feature_dim))
        output[valid.flatten()] = encoded
        return output.view(*valid.shape, self.n, self.feature_dim)

    def _scores(self, h):
        if self.aggregate == 'gated_mean_mix':
            gated = torch.tanh(self.attn_a(h)) * torch.sigmoid(self.attn_b(h))
            scores = self.attn_out(gated)
        else:
            scores = self.att(h)
        if h.ndim == 4:
            # Each disease query has its own window features and corresponding score.
            scores = scores.diagonal(dim1=-2, dim2=-1)
        return scores / self.temperature

    @staticmethod
    def _weighted_values(attention, values):
        attention = attention.to(values.dtype)
        if values.ndim == 3:
            return torch.einsum('bkn,bkf->bnf', attention, values)
        return torch.einsum('bkn,bknf->bnf', attention, values)

    def _pool(self, h, values, valid, hierarchy_ids):
        if self.hierarchical:
            features, masks = [], []
            scores = self._scores(h)
            # Identity numbers are only compared within each study. Enumerating
            # their union does not mix patients or imply identical anatomy.
            for group in torch.unique(hierarchy_ids[valid]):
                member = valid & (hierarchy_ids == group)
                a = _masked_softmax(scores, member)
                features.append(self._weighted_values(a, values))
                masks.append(member.any(dim=1))
            if not features:
                return values.new_zeros((values.shape[0], self.n, self.feature_dim))
            gf = torch.stack(features, dim=1)  # B,G,12,D
            gm = torch.stack(masks, dim=1)
            ga = self.group_att(gf).diagonal(dim1=-2, dim2=-1) / self.temperature
            return self._weighted_values(_masked_softmax(ga, gm), gf)
        if self.aggregate in {'attention', 'gated', 'gated_mean_mix'}:
            attended = self._weighted_values(_masked_softmax(self._scores(h), valid), values)
            if self.aggregate == 'gated_mean_mix':
                mask = valid
                while mask.ndim < values.ndim:
                    mask = mask.unsqueeze(-1)
                mean = values.sum(1) / mask.sum(1).clamp_min(1)
                if values.ndim == 3:
                    mean = mean.unsqueeze(1).expand(-1, self.n, -1)
                mix = self.pool_mix_logit.sigmoid().view(1, self.n, 1).to(values.dtype)
                return mix * attended + (1 - mix) * mean
            return attended
        mask = valid
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        count = mask.sum(1).clamp_min(1)
        mean = values.sum(1) / count
        maximum = values.masked_fill(~mask, -torch.inf).max(dim=1).values
        any_valid = valid.any(1)
        maximum = torch.where(any_valid.view(-1, *([1] * (maximum.ndim - 1))), maximum, torch.zeros_like(maximum))
        if values.ndim == 3:
            mean = mean.unsqueeze(1).expand(-1, self.n, -1)
            maximum = maximum.unsqueeze(1).expand(-1, self.n, -1)
        if self.aggregate == 'mean':
            return mean
        if self.aggregate == 'max':
            return maximum
        return torch.cat([mean, maximum], dim=-1)

    def _finalize_logits(self, logits, valid):
        if self.label_mixing:
            # The zero-initialized off-diagonal residual exactly preserves the
            # independent heads until training finds useful label relations.
            weights = torch.tanh(self.label_mix_logits) * self.label_mix_mask
            residual = torch.einsum('bi,oi->bo', torch.tanh(logits.float()), weights.float())
            logits = logits + (self.label_mixing_cap / (self.n - 1)) * residual.to(logits.dtype)
        return torch.where(valid.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))

    def head(self, feats, window_mask=None, group_ids=None, positions=None,
             series_ids=None, position_mask=None, plane_ids=None):
        if feats.ndim not in (3, 4) or feats.shape[-1] != self.feature_dim or feats.shape[1] == 0:
            raise ValueError('features must be [B,K,D] or [B,K,12,D] with K > 0')
        if feats.ndim == 4 and feats.shape[2] != self.n:
            raise ValueError('spatial disease feature axis must contain the 12 labels')
        valid = self._valid(feats.shape[:2], window_mask, feats.device)
        mask = valid
        while mask.ndim < feats.ndim:
            mask = mask.unsqueeze(-1)
        feats = torch.where(mask, feats, torch.zeros_like(feats))
        if not torch.isfinite(feats).all():
            raise ValueError('non-finite features in a valid window')
        need_groups = self.use_groups or (self.hierarchical and self.hierarchy_axis == 'group')
        groups = self._groups(group_ids, valid) if need_groups else None
        hierarchy_ids = None
        if self.hierarchical:
            if self.hierarchy_axis == 'series':
                hierarchy_ids = self._series(series_ids, valid)
            elif self.hierarchy_axis == 'group':
                hierarchy_ids = groups
            else:
                # Coarser than series/group: one pool per acquisition plane.
                hierarchy_ids = self._planes(plane_ids, valid)
        if self.use_groups:
            embedding = self.group_embed(groups)
            embedding = embedding.unsqueeze(2) if feats.ndim == 4 else embedding
            if not (self.spatial_residual_gated and self.aggregate == 'attention'):
                feats = feats + embedding
        else:
            embedding = None
        if self.use_positions:
            if positions is None:
                raise ValueError('positions in actual series-relative millimetres are required')
            positions = positions.to(device=feats.device, dtype=feats.dtype)
            if positions.ndim == 2 and self.position_dim == 1:
                positions = positions.unsqueeze(-1)
            if positions.shape != (*valid.shape, self.position_dim):
                raise ValueError('positions must have shape [B,K,position_dim]')
            position_valid = valid if position_mask is None else valid & self._valid(valid.shape, position_mask, feats.device)
            positions = torch.where(position_valid.unsqueeze(-1), positions, torch.zeros_like(positions))
            if not torch.isfinite(positions).all():
                raise ValueError('non-finite physical positions in a valid window')
            positional = self.position_mlp(positions / self.position_scale_mm)
            positional = torch.where(position_valid.unsqueeze(-1), positional, torch.zeros_like(positional))
            positional = positional.unsqueeze(2) if feats.ndim == 4 else positional
            if not (self.spatial_residual_gated and self.aggregate == 'attention'):
                feats = feats + positional
        else:
            positional = None
        if self.spatial_residual_gated and self.aggregate == 'attention':
            # Match the released second stage: normalize spatial values first,
            # and use slot/depth routes only for attention, never classification.
            values = torch.where(mask, self.norm(feats), torch.zeros_like(feats))
            h = values
            if embedding is not None:
                h = h + embedding
            if positional is not None:
                h = h + positional
            h = torch.where(mask, h, torch.zeros_like(h))
            pooled = self._pool(h, values, valid, hierarchy_ids)
            logits = (pooled * self.clsW).sum(-1) + self.clsb
            return self._finalize_logits(logits, valid)
        if self.aggregate == 'gated_mean_mix':
            slots = self.slot_embedding(self._slots(group_ids, valid))
            if feats.ndim == 4:
                slots = slots.unsqueeze(2).expand(-1, -1, self.n, -1)
            feats = self.feature_proj(torch.cat([feats, slots], dim=-1))
            mask = valid
            while mask.ndim < feats.ndim:
                mask = mask.unsqueeze(-1)
            h = values = torch.where(mask, feats, torch.zeros_like(feats))
            pooled = self._pool(h, values, valid, hierarchy_ids)
            logits = (pooled * self.classifier_weight).sum(-1) + self.classifier_bias
            return self._finalize_logits(logits, valid)
        # Mask before AND after LN: learned LN bias must not enter plain mean pools.
        feats = torch.where(mask, feats, torch.zeros_like(feats))
        h = torch.where(mask, self.norm(feats), torch.zeros_like(feats))
        values = h if self.value_norm == 'normalized' else feats
        pooled = self._pool(h, values, valid, hierarchy_ids)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return self._finalize_logits(logits, valid)

    def forward(self, x, window_mask=None, group_ids=None, positions=None,
                series_ids=None, position_mask=None, plane_ids=None):
        return self.head(self.encode(x, window_mask, group_ids), window_mask, group_ids,
                         positions, series_ids, position_mask, plane_ids)


def build_model(cfg, pretrained=False):
    """Build CoAtNet or ConvNeXt; pretrained=True always loads an explicit local file.

    ConvNeXt reuses the CoAtNet head unchanged: both expose an NCHW stride-32 map
    from forward_features and a pooled, normed vector from forward_head.
    (Ported from reports/convnext_route_20261002/runtime, the ConvNeXt LB path.)
    """
    if pretrained:
        return initialize_from_imagenet(cfg)
    import timm
    arch = str(_get(cfg, 'arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'))
    convnext = arch.startswith('convnext')
    if not (convnext or arch.startswith(('coatnet', 'coat_'))):
        raise ValueError('this ablation builder is restricted to CoAtNet and ConvNeXt backbones')
    kwargs = dict(pretrained=False, num_classes=0, in_chans=3, global_pool='avg')
    resolution = int(_get(cfg, 'resolution', 384))
    if resolution != 384 and not convnext:
        kwargs['img_size'] = resolution
    backbone = timm.create_model(arch, **kwargs)
    if _get(cfg, 'gradient_checkpointing', False):
        backbone.set_grad_checkpointing(True)
    return CoAtNetAblationModel(backbone, cfg)


def initialize_from_imagenet(cfg):
    """Load either the generic ImageNet backbone or an exported SSL backbone."""
    model = build_model(cfg, pretrained=False)
    external = _get(cfg, 'backbone_init_checkpoint', None)
    if external:
        path = Path(external)
        if not path.is_file() or path.suffix not in {'.pt', '.pth', '.ckpt'}:
            raise ValueError('backbone_init_checkpoint must point to a local torch checkpoint')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('backbone', checkpoint.get('state_dict', checkpoint))
        if not isinstance(state, dict):
            raise ValueError('backbone_init_checkpoint has no backbone state dictionary')
        expected = set(model.backbone.state_dict())
        if set(state) != expected:
            missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
            raise ValueError(f'external backbone keys differ (missing={missing[:5]}, extra={extra[:5]})')
        model.backbone.load_state_dict(state, strict=True)
        return model
    from safetensors.torch import load_file
    path = Path(_get(cfg, 'initial_weights', ''))
    if not path.is_file() or path.suffix != '.safetensors':
        raise ValueError('initial_weights must point to local generic ImageNet safetensors')
    state = load_file(str(path))
    extra = set(state) - set(model.backbone.state_dict())
    # Some ConvNeXt files were exported headless (num_classes=0); strict loading below
    # still requires every backbone tensor.
    headless_ok = str(_get(cfg, 'arch', '')).startswith('convnext') and not extra
    if extra != {'head.fc.weight', 'head.fc.bias'} and not headless_ok:
        raise ValueError(f'expected generic ImageNet classifier keys, got extras {sorted(extra)}')
    for key in extra:
        del state[key]
    model.backbone.load_state_dict(state, strict=True)
    return model
