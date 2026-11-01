"""Deployment of the runtime_nq5 compartment coupling (oai_weak3 RK4C family; user request 2026-10-19: submit RK4C).

Training (reports/oai_weak3_20261018/runtime_nq5/src/kneemri/native_aux.py, native_couple: true) adds to each
compartment OA logit a_c * z_c, where z_c is the max over the compartment's MOAKS subregions of the auxiliary head's
expected full-thickness grade (sum of its 3 cumulative sigmoids) and a_c is a learned scalar. That term is part of the
trained model's output, so unlike the uncoupled N3 arms the auxiliary head and a_c stay in the deployed checkpoint.
This module rebuilds them (NativeOrdinal copied unchanged) and applies the same eval-mode hooks. The coupling map
(class column, auxiliary field indices) travels in config[KEY]; no MOAKS / OAI data is needed at inference.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

KEY = 'deploy_native_couple'
# Copied from native_aux.COUPLE (runtime_nq5).
COUPLE = {'Medial OA': ['FMC', 'FMP', 'TMA', 'TMC', 'TMP'], 'Lateral OA': ['FLC', 'FLP', 'TLA', 'TLC', 'TLP'],
          'PF OA': ['PM', 'PL', 'FMA', 'FLA']}


class NativeOrdinal(nn.Module):
    """Copied unchanged from runtime_nq5 native_aux.NativeOrdinal."""
    def __init__(self, dim, fields=28):
        super().__init__()
        self.score = nn.Linear(dim, fields)
        self.query_logits = nn.Parameter(torch.zeros(fields, 12))
        self.first = nn.Parameter(torch.full((fields,), -.5))
        self.gaps = nn.Parameter(torch.full((fields, 2), math.log(math.expm1(1.))))

    def forward(self, h):
        offsets = torch.cat([torch.zeros_like(self.first[:, None]), F.softplus(self.gaps).cumsum(-1)], -1)
        threshold = self.first[:, None] + offsets
        assert h.ndim == 3 and h.shape[1] == 12
        mixing = self.query_logits.float().softmax(-1)
        regional = torch.einsum('bcd,fc->bfd', h.float(), mixing)
        score = (regional * self.score.weight.float()[None]).sum(-1) + self.score.bias.float()[None]
        return score[:, :, None] - threshold[None]


def coupling_spec(fields, label_cols):
    """The (class column, F-field indices) pairs native_aux.install builds from the label file's field names."""
    fields = [str(f) for f in fields]
    pairs = [(label_cols.index(c), [fields.index(r + '_F') for r in regs if r + '_F' in fields]) for c, regs in COUPLE.items()]
    couple = [[c, f] for c, f in pairs if f]
    return dict(fields=len(fields), couple=couple, classes=[label_cols[c] for c, _ in couple],
                field_names=[[fields[i] for i in f] for _, f in couple])


def attach(base, spec):
    base.squeeze_aux = NativeOrdinal(base.feature_dim, int(spec['fields']))
    base.squeeze_couple = nn.Parameter(torch.zeros(len(spec['couple'])))
    base._native_couple = [(int(c), [int(i) for i in f]) for c, f in spec['couple']]


def install(rdi):
    """Patch the pooled-feature and logit hooks (no-ops for models without squeeze_couple) and the model builder."""
    from kneemri.coatnet_model import CoAtNetAblationModel
    if getattr(CoAtNetAblationModel, '_native_couple_deploy', False):
        return
    old_pool, old_finalize = CoAtNetAblationModel._pool, CoAtNetAblationModel._finalize_logits

    def pool(self, *args, **kwargs):
        h = old_pool(self, *args, **kwargs)
        if getattr(self, 'squeeze_couple', None) is not None:
            self._native_logits = self.squeeze_aux(h)
        return h

    def finalize(self, logits, valid):
        out = old_finalize(self, logits, valid)
        a = getattr(self, 'squeeze_couple', None)
        if a is None:
            return out
        e = torch.sigmoid(self._native_logits.detach().float()).sum(-1)  # [B,F] expected grade
        cols = torch.as_tensor([c for c, _ in self._native_couple], device=out.device)
        z = torch.stack([e[:, f].max(1).values for _, f in self._native_couple], 1)  # [B,C]
        add = torch.zeros(out.shape, device=out.device, dtype=torch.float32).index_add(1, cols, z * a.float()[None])
        return out + add.to(out.dtype)

    CoAtNetAblationModel._pool, CoAtNetAblationModel._finalize_logits = pool, finalize
    CoAtNetAblationModel._native_couple_deploy = True
    old_build = rdi.build_depth_model

    def build(cfg, pretrained=False):
        model = old_build(cfg, pretrained)
        if cfg.get(KEY):
            attach(model.base, cfg[KEY])
        return model

    rdi.build_depth_model = build
    rdi.resnet.trainer.DATA_KEYS.add(KEY)
