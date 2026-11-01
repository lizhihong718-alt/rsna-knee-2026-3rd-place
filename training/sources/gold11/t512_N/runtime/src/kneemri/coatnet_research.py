"""Fold-isolated domain adaptation and single-backbone research extensions.

Auxiliary targets describe acquisition coverage, NOT clinical lesion visibility.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .coatnet_model import CoAtNetAblationModel
from .coatnet_original_data import CoAtNetOriginalNeighborBags, ARRAYS

RESEARCH_DEFAULTS = dict(progressive_unfreeze=False, head_only_epochs=2,
    last_stage_epochs=3, layer_decay=1.0, aux_tasks=[], aux_weight=0.,
    multiscale=False, wide_corpus=None, ssl_checkpoint=None)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_ssl_cohort(source, cohort):
    trained = set(source.get('trained_study_uids', []))
    groups = set(source.get('trained_groups', []))
    allowed = set(cohort['trained_study_uids'])
    protected = set(cohort.get('protected_validation_study_uids', []))
    if not trained or not groups or not trained <= allowed or trained & protected:
        raise ValueError('SSL training studies are not strictly isolated within the downstream training fold')
    if not groups <= set(cohort['trained_groups']):
        raise ValueError('SSL groups cross the downstream training boundary')
    if source.get('folds_sha256') != cohort.get('folds_sha256'):
        raise ValueError('SSL and downstream fold definitions differ')


def attach_ssl_provenance(cfg, cohort):
    path = cfg.get('ssl_checkpoint')
    if not path:
        return
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    validate_ssl_cohort(checkpoint['cohort'], cohort)
    if checkpoint.get('arch') != cfg['arch'] or checkpoint.get('res') != cfg['resolution']:
        raise ValueError('SSL backbone architecture/resolution differs')
    if checkpoint.get('smoke'):
        raise ValueError('smoke SSL checkpoint cannot initialize an experiment')
    cohort.update(model_origin='ImageNet_then_fold_isolated_medical_SSL_student',
                  ssl_checkpoint_sha256=sha256(path), ssl_method=checkpoint['method'],
                  ssl_training_cohort=checkpoint['cohort'])


class ResearchBags(CoAtNetOriginalNeighborBags):
    def __init__(self, cfg, all_ids, ids, labels, train):
        super().__init__(cfg, all_ids, ids, labels, train)
        self.wide_windows = None
        if cfg.get('multiscale'):
            root = Path(cfg['wide_corpus'])
            manifest = json.loads((root / 'manifest.json').read_text())
            if manifest['study_uids'] != self.manifest['study_uids'] or manifest['shape'] != self.manifest['shape']:
                raise ValueError('two fields of view must have identical study/window layout')
            narrow_pre,wide_pre=self.manifest['preprocessing'],manifest['preprocessing']
            for key in ['schema','source_schema','source_preprocess_signature','source_processed_dir','slots','span','image','neighbor_gap_ratio']:
                if narrow_pre.get(key) != wide_pre.get(key):
                    raise ValueError(f'paired FOV source preprocessing differs: {key}')
            if not wide_pre.get('crop_fov_mm',0) > narrow_pre.get('crop_fov_mm',0) > 0:
                raise ValueError('paired FOV requires a truly wider physical crop')
            if self.resolution != self.source_resolution:
                raise ValueError('paired FOV cache must match the model resolution')
            for key in [*ARRAYS, 'masks', 'complete']:
                if not np.array_equal(np.load(root / f'{key}.npy', mmap_mode='r'),
                                      np.load(self.root / f'{key}.npy', mmap_mode='r')):
                    raise ValueError(f'paired FOV geometry mismatch: {key}')
            if not np.load(root / 'complete.npy', mmap_mode='r').all():
                raise ValueError('wide corpus is incomplete')
            self.wide_root = root

    def __getstate__(self):
        state = super().__getstate__()
        state['wide_windows'] = None
        return state

    def _selected(self, row):
        result = super()._selected(row)
        self.paired_row, self.paired_indices = row, result[0]
        return result

    def _augment(self, images):
        if not self.cfg.get('multiscale'):
            return super()._augment(images)
        if self.wide_windows is None:
            self.wide_windows = np.load(self.wide_root / 'windows.npy', mmap_mode='r')
        wide = np.asarray(self.wide_windows[self.paired_row, self.paired_indices])
        # One geometric/intensity draw is shared by both fields and all slices.
        both = super()._augment(np.concatenate([images, wide], axis=0))
        return np.stack(np.split(both, 2, axis=0), axis=1)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.cfg.get('multiscale') and not item['window_mask'].any():
            item['images'] = item['images'].unsqueeze(1).expand(-1, 2, -1, -1, -1).clone()
        return item


class ResearchCoAtNet(CoAtNetAblationModel):
    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        self.research_cfg = cfg
        d = self.feature_dim
        if cfg.get('multiscale'):
            self.fov_gate = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, 128),
                                          nn.GELU(), nn.Linear(128, 1))
            nn.init.zeros_(self.fov_gate[-1].weight)
            nn.init.constant_(self.fov_gate[-1].bias, -2.)
        tasks = set(cfg.get('aux_tasks', []))
        self.aux_heads = nn.ModuleDict()
        for name, width in [('depth', 1), ('geometry', 1), ('slot', 6), ('series', 5)]:
            if name in tasks:
                self.aux_heads[name] = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width))

    def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
        if not self.research_cfg.get('multiscale'):
            return super().encode(x, window_mask, group_ids, return_backbone_features)
        if return_backbone_features:
            raise ValueError('rank replay is not enabled for paired FOV')
        if x.ndim != 6 or x.shape[2:4] != (2, 3):
            raise ValueError('paired FOV images must be [B,K,2,3,H,W]')
        b, k = x.shape[:2]
        mask = self._valid((b,k), window_mask, x.device)
        # Both views go through THE SAME backbone and spatial pool in one batch.
        masks = mask.unsqueeze(-1).expand(-1,-1,2).reshape(b,2*k)
        groups = group_ids.unsqueeze(-1).expand(-1,-1,2).reshape(b,2*k)
        feats = super().encode(x.reshape(b,2*k,3,*x.shape[-2:]), masks, groups)
        feats = feats.reshape(b,k,2,*feats.shape[2:])
        detail, wide = feats[:,:,0], feats[:,:,1]
        gate = torch.sigmoid(self.fov_gate(torch.cat([detail,wide],dim=-1)))
        # Feature fusion precedes the one study attention/classification head.
        return detail + gate * (wide-detail)

    def forward(self, x, window_mask=None, group_ids=None, positions=None,
                series_ids=None, position_mask=None, plane_ids=None, return_aux=False):
        if return_aux:
            if self.research_cfg.get('multiscale'):
                raise ValueError('paired-FOV and auxiliary effects are evaluated separately')
            features, maps = self.encode(x, window_mask, group_ids, return_backbone_features=True)
        else:
            features = self.encode(x, window_mask, group_ids)
        logits = self.head(features, window_mask, group_ids, positions, series_ids, position_mask, plane_ids)
        if not return_aux:
            return logits
        # Image-only GAP features bypass all slot-conditioned spatial pooling.
        valid = self._valid(features.shape[:2], window_mask, features.device)
        window = features.new_zeros((*valid.shape, self.feature_dim))
        if maps is not None:
            window[valid] = self.backbone.forward_head(maps)
        valid = self._valid(window.shape[:2], window_mask, window.device)
        pooled = (window * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
        outputs = {name: head(pooled if name in {'slot','series'} else window)
                   for name,head in self.aux_heads.items()}
        return logits, outputs


def auxiliary_loss(outputs, metadata):
    valid = metadata['window_mask'].bool()
    reliable = valid & metadata['position_mask'].bool()
    losses = {}
    for name, pred in outputs.items():
        if name == 'depth':
            target = metadata['positions'].float() / 100.  # series-relative mm, fixed scale
            losses[name] = F.smooth_l1_loss(pred[reliable].float(), target[reliable]) if reliable.any() else pred.sum()*0
        elif name == 'geometry':
            # A constant target carries no information about reliability.
            if reliable.any() and (valid & ~reliable).any():
                losses[name] = F.binary_cross_entropy_with_logits(pred.squeeze(-1)[valid].float(), reliable[valid].float())
        else:
            ids = metadata['group_ids' if name == 'slot' else 'series_ids']
            target = torch.stack([((ids == i) & valid).any(1) for i in range(pred.shape[-1])],1).float()
            keep = valid.any(1)
            losses[name] = F.binary_cross_entropy_with_logits(pred[keep].float(), target[keep]) if keep.any() else pred.sum()*0
    total = sum(losses.values()) if losses else sum(p.sum()*0 for p in outputs.values())
    return total / max(1,len(losses)), {k:float(v.detach()) for k,v in losses.items()}


def build_research_model(cfg, pretrained=False):
    import timm
    kwargs = dict(pretrained=False, num_classes=0, in_chans=3, global_pool='avg')
    if cfg['resolution'] != 384:
        kwargs['img_size'] = cfg['resolution']
    backbone = timm.create_model(cfg['arch'], **kwargs)
    if cfg.get('gradient_checkpointing'):
        backbone.set_grad_checkpointing(True)
    model = ResearchCoAtNet(backbone,cfg)
    if pretrained:
        if cfg.get('ssl_checkpoint'):
            state = torch.load(cfg['ssl_checkpoint'],map_location='cpu',weights_only=True)['backbone']
        else:
            from safetensors.torch import load_file
            state = load_file(cfg['initial_weights'])
            extra = set(state)-set(backbone.state_dict())
            if extra != {'head.fc.weight','head.fc.bias'}:
                raise ValueError(f'unexpected ImageNet keys: {extra}')
            state = {k:v for k,v in state.items() if k not in extra}
        backbone.load_state_dict(state,strict=True)
    return model


def layer_id(name, last_stage):
    if name.startswith('stem.'):
        return 0
    if name.startswith('stages.'):
        return 1 + int(name.split('.')[1])
    return last_stage + 2


def parameter_groups(model, cfg):
    if not 0 < cfg['layer_decay'] <= 1:
        raise ValueError('layer_decay must be in (0,1]')
    last = len(model.backbone.stages)-1
    buckets = {}
    for name,p in model.backbone.named_parameters():
        level = layer_id(name,last)
        buckets.setdefault(level,[]).append(p)
    groups = [{'params':p, 'lr':cfg['backbone_lr']*cfg['layer_decay']**(last+2-level),
               'name':f'backbone_layer_{level}'} for level,p in sorted(buckets.items())]
    spatial, head = [], []
    for name,p in model.named_parameters():
        if name.startswith('backbone.'):
            continue
        (spatial if name.startswith('spatial_pool.') and cfg['spatial_lr'] is not None else head).append(p)
    if spatial:
        groups.append(dict(params=spatial,lr=cfg['spatial_lr'],name='spatial'))
    groups.append(dict(params=head,lr=cfg['head_lr'],name='head'))
    return groups


def apply_unfreeze(model, cfg, epoch):
    phase = ('head' if epoch < cfg['head_only_epochs'] else 'last_stage'
             if epoch < cfg['head_only_epochs']+cfg['last_stage_epochs'] else 'all') if cfg['progressive_unfreeze'] else 'all'
    last_prefix = f'stages.{len(model.backbone.stages)-1}.'
    for name,p in model.backbone.named_parameters():
        p.requires_grad_(phase == 'all' or phase == 'last_stage' and name.startswith((last_prefix,'head.','norm.')))
    # Frozen BN/dropout must not keep adapting, including during checkpoint recomputation.
    if phase == 'head':
        model.backbone.eval()
    elif phase == 'last_stage':
        model.backbone.eval()
        model.backbone.stages[-1].train()
        model.backbone.head.train()
    return phase
