"""Swin Transformer (V1 / V2 / V2-CR) study models on the native224 / native384 corpora.

Model: the BiT E2E_D study model (resnet_depth_e2e.DepthStudyModel: whole-series
bags, depth context on the final maps, disease spatial queries + GAP, study MIL)
with the encoder swapped for a timm Swin. ``forward_features`` of Swin V1/V2 ends
in the final LayerNorm and returns NHWC [N,H,W,C]; it is permuted to the NCHW
[N,C,H,W] map the depth context and spatial queries expect (7x7 at a 224 encoder
input, 12x12 at 384 -- the same grid as BiT at stride 32). Nothing else in the
head changes.

Pixels: the corpus windows go to the network exactly as stored (resolution must
equal the corpus resolution: native224 -> 224, native384 -> 384). The encoder is
built for its input size (``img_size``) and ``window_size``; pretrained weights are
loaded through timm's own checkpoint filter, which drops the non-persistent
relative-position index / attention-mask buffers and bicubically resizes the
relative-position bias tables when the window differs from pretraining (V1).
V2 uses a continuous log-spaced position MLP, which transfers across windows
without resizing (``pretrained_window_sizes`` stay those of the timm arch).

Optional keys (all absent = plain port):
  window_size          attention window (default: the timm arch's pretraining window);
  drop_path_rate       stochastic depth (default 0.1 = timm's Swin default);
  model_input_size     bilinear resize of the windows inside the model before the encoder
                       (same pixels, finer encoder grid; never used unless declared);
  patch_size           patch-embedding stride (default 4). 2 doubles every token grid on the SAME input
                       tensor (no interpolation; 224 -> 112/56/28/14 tokens, final map 14x14). The pretrained
                       4x4 embedding is resampled to 2x2 by timm's pseudo-inverse resize (FlexiViT PI-resize),
                       so a 2x2 patch is embedded like the 4x4 patch of a 2x-upsampled window.
  layer_decay          layer-wise LR decay over Swin blocks (applied by train_swin.py);
  no_decay_norm_bias   exclude norms, biases, position tables and logit scales from weight decay.
  abs_pos              learned absolute position embedding, zero-initialised (start = the plain model), trained
                       at head_lr with no weight decay. 'stem': [1,H/p,W/p,C] added to the patch-embedding tokens
                       (inside backbone.patch_embed, key patch_embed.pos_embed); 'final': [1,C,h,w] added to the
                       final NCHW map before depth context / spatial queries (key final_pos). Swin has no absolute
                       position (relative bias only, translation-equivariant windows), unlike zero-padded CNNs; the
                       Swin arms trail BiT systematically on the lateral-compartment labels (LOA/LM).
  fused_attn           (infrastructure) Swin V1 window attention through F.scaled_dot_product_attention with the
                       relative-position bias (+ shift mask) as an additive mask; timm keeps it off by default
                       ('experimental'). Same function, different kernel: fp16 eval rel. diff ~1e-3, -30% memory,
                       ~8-12% faster at 384 (10-12 bench).
"""
from pathlib import Path
import re

import torch
from torch.nn import functional as F

from kneemri.coatnet_data import NORMALIZATION_STATS

MODEL_KEYS = {'window_size', 'drop_path_rate', 'model_input_size', 'patch_size', 'abs_pos'}
ABS_POS = (None, 'stem', 'final')
POS_NAMES = ('patch_embed.pos_embed', 'final_pos')
DATA_KEYS = {'cudnn_benchmark', 'layer_decay', 'no_decay_norm_bias', 'fused_attn', 'compile_backbone', 'compile_bucket'}
INFRA = {'cudnn_benchmark', 'fused_attn', 'compile_backbone', 'compile_bucket'}
ARCH_PATTERN = r'^swin(?:v2)?(?:_cr)?_(?:tiny|small|base|large)'


def family(arch):
    if arch.startswith('swinv2_cr_'):
        return 'v2cr'
    return 'v2' if arch.startswith('swinv2_') else 'v1'


def encoder_size(cfg):
    size = cfg.get('model_input_size')
    res = int(cfg['resolution'])
    if size is None or size == res:
        return res
    if isinstance(size, bool) or not isinstance(size, int) or size < res or size % 32:
        raise ValueError('model_input_size must be a multiple of 32 >= resolution')
    return size


def build_backbone(cfg):
    import timm

    arch = str(cfg['arch'])
    if not re.match(ARCH_PATTERN, arch):
        raise ValueError(f'Swin route requires a swin / swinv2 / swinv2_cr backbone, got {arch}')
    rate = float(cfg.get('drop_path_rate', 0.1))
    if not 0.0 <= rate < 0.6:
        raise ValueError('drop_path_rate must lie in [0, 0.6)')
    extra = dict(img_size=encoder_size(cfg))
    if cfg.get('window_size') is not None:
        window = cfg['window_size']
        if isinstance(window, bool) or not isinstance(window, int) or not 4 <= window <= 32:
            raise ValueError('window_size must be an int in [4, 32]')
        extra['window_size'] = window
    patch = cfg.get('patch_size', 4)
    if patch not in (2, 4) or isinstance(patch, bool):
        raise ValueError('patch_size must be 2 or 4')
    if patch != 4:
        extra['patch_size'] = patch
    backbone = timm.create_model(arch, pretrained=False, num_classes=0, global_pool='avg',
                                 drop_path_rate=rate, **extra)
    # The study model hands NCHW maps to forward_head (spatial-query GAP); timm's Swin head pools
    # NHWC. Same parameter-free average pool, NCHW layout (state dict unchanged: no head params).
    from timm.layers import ClassifierHead
    backbone.head = ClassifierHead(backbone.num_features, 0, pool_type='avg', drop_rate=0.0, input_fmt='NCHW')
    stats = NORMALIZATION_STATS.get(cfg.get('normalization'))
    expected = backbone.pretrained_cfg
    if stats is None or not all(torch.allclose(torch.tensor(actual), torch.tensor(expected[key]))
                                for actual, key in zip(stats, ('mean', 'std'))):
        raise ValueError(f'normalization does not match pretrained {arch}')
    fused = cfg.get('fused_attn', False)
    if not isinstance(fused, bool):
        raise ValueError('fused_attn must be boolean')
    if fused:
        if family(arch) == 'v2':
            n = install_v2_fused(backbone)
        elif family(arch) != 'v1':
            raise ValueError('fused_attn is implemented for Swin V1 (timm switch) and V2 (kneemri) only')
        else:
            n = 0
            for module in backbone.modules():
                if hasattr(module, 'fused_attn'):
                    module.fused_attn = True
                    n += 1
        if not n:
            raise ValueError('no fused-attention switch found')
    if cfg.get('gradient_checkpointing', False):
        # LayerNorm only (no running statistics), so recomputation is exact.
        backbone.set_grad_checkpointing(True)
    return backbone


def install_v2_fused(backbone):
    """Swin V2 cosine window attention through F.scaled_dot_product_attention (class swap, deepcopy-safe).

    timm computes softmax(normalize(q) normalize(k)^T * exp(clamp(logit_scale)) + 16 sigmoid(cpb) [+ shift mask]) v
    with an explicit [B*nW, nH, N, N] logit tensor (no fused kernel: 512/window 32 runs out of memory). Same function
    here with the per-head scale folded into the normalised query and the bias (+ shift mask) as the additive SDPA
    mask (scale=1). The bias enters as 16 sigmoid(cpb) - 8: a constant per logit row (softmax-invariant) that centres
    the [0, 16] range so the half-precision mask SDPA requires (autocast) loses half the rounding.
    """
    import math
    from timm.models import swin_transformer_v2 as v2

    class FusedWindowAttentionV2(v2.WindowAttention):
        def forward(self, x, mask=None):
            B_, N, C = x.shape
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
            # normalise in fp32 (a half-precision norm doubles the bf16 error of the whole V2-B feature map: .079 vs
            # eager .039 relative to fp32; fp32 norm: .040), then the SDPA operands in the autocast dtype.
            q = (F.normalize(q.float(), dim=-1) * logit_scale).to(v.dtype)
            k = F.normalize(k.float(), dim=-1).to(v.dtype)
            table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            bias = table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
            bias = (16 * torch.sigmoid(bias.float()) - 8).unsqueeze(0)
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                bias = bias + mask.reshape(-1, self.num_heads, N, N)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(v.dtype),
                                               dropout_p=self.attn_drop.p if self.training else 0., scale=1.0)
            x = x.transpose(1, 2).reshape(B_, N, C)
            return self.proj_drop(self.proj(x))

    n = 0
    for module in backbone.modules():
        if type(module) is v2.WindowAttention:
            module.__class__ = FusedWindowAttentionV2
            n += 1
    return n


def install_compile(backbone, bucket=16):
    """Training-only torch.compile of forward_features (throughput only; class swap, so the EMA deepcopy and every
    eval / no-grad call run the eager method on their own weights).

    Whole-series bags give a different window count every step: rows are padded to a multiple of ``bucket`` with
    copies of the last image and the padded outputs dropped (Swin = LayerNorm only, no batch statistics: each output
    row depends on its own input; dropped rows get no gradient), so the compiled graph sees a few static shapes.
    """
    if not isinstance(bucket, int) or isinstance(bucket, bool) or bucket < 1:
        raise ValueError('compile_bucket must be a positive int')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in backbone.modules()):
        raise ValueError('row padding is only exact for backbones without batch statistics')
    base = type(backbone)
    state = {}

    def forward_features(self, x):
        if not (self.training and torch.is_grad_enabled()):
            return base.forward_features(self, x)
        fn = state.get('fn')
        if fn is None:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            fn = state['fn'] = torch.compile(base.forward_features, dynamic=False)
        rows = x.shape[0]
        pad = (-rows) % bucket
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], 0)
        out = fn(self, x)
        return out[:rows] if pad else out

    backbone.__class__ = type(base.__name__ + 'Compiled', (base,), {'forward_features': forward_features})
    return backbone


def _filter(arch):
    from timm.models import swin_transformer, swin_transformer_v2, swin_transformer_v2_cr
    return {'v1': swin_transformer, 'v2': swin_transformer_v2,
            'v2cr': swin_transformer_v2_cr}[family(arch)].checkpoint_filter_fn


NON_PERSISTENT = ('relative_position_index', 'relative_coords_table', 'attn_mask')


def load_pretrained(backbone, cfg):
    from safetensors.torch import load_file

    path = Path(cfg['initial_weights'])
    if path.suffix != '.safetensors' or not path.is_file():
        raise ValueError('initial_weights must be a local generic pretrained safetensors file')
    if path.parent.name != str(cfg['arch']):
        raise ValueError('initial_weights directory must be named after the timm arch tag')
    state = _filter(str(cfg['arch']))(load_file(str(path)), backbone)
    own = backbone.state_dict()
    head = {k for k in state if k.startswith('head.')}
    extra = set(state) - set(own) - head
    if extra:
        raise ValueError(f'unexpected pretrained keys: {sorted(extra)[:8]}')
    missing = [k for k in own if k not in state and not k.endswith(NON_PERSISTENT)]
    if missing:
        raise ValueError(f'pretrained weights miss {missing[:8]}')
    bad = [k for k in state if k in own and state[k].shape != own[k].shape]
    if bad:
        raise ValueError(f'pretrained shape mismatch {bad[:8]}')
    result = backbone.load_state_dict({k: v for k, v in state.items() if k not in head}, strict=False)
    if result.unexpected_keys or any(not k.endswith(NON_PERSISTENT) for k in result.missing_keys):
        raise ValueError(f'load mismatch {result}')


def block_ids(backbone):
    """Parameter name -> depth index: patch embed 0, each Swin block 1..L in order, final norm L+1.

    A stage's patch-merging downsample sits in front of its first block (timm >= 0.9 layout) and
    shares that block's index.
    """
    ids, k = {}, 0
    names = [n for n, _ in backbone.named_parameters()]
    stages = backbone.layers if hasattr(backbone, 'layers') else backbone.stages
    counts = [len(s.blocks) for s in stages]
    starts = [1 + sum(counts[:i]) for i in range(len(counts))]
    total = sum(counts)
    for n in names:
        if n.startswith(('patch_embed.', 'absolute_pos_embed')):
            ids[n] = 0
            continue
        m = re.match(r'^(?:layers|stages)\.(\d+)\.(blocks\.(\d+)|downsample)', n)
        if m:
            s = int(m.group(1))
            ids[n] = starts[s] + (int(m.group(3)) if m.group(3) is not None else 0)
            continue
        if n.startswith(('norm.', 'head.')):
            ids[n] = total + 1
            continue
        raise ValueError(f'unmapped Swin parameter {n}')
    return ids, total + 1


def no_decay(name, param):
    return (param.ndim <= 1 or name.endswith('.bias') or 'relative_position_bias_table' in name
            or 'logit_scale' in name or 'cpb_mlp' in name or 'absolute_pos_embed' in name)


def split_backbone_group(backbone, group, decay=None, norm_bias=False):
    """One backbone param group -> groups by (depth, decay flag): lr * decay**(L+1 - depth)."""
    ids, depth = block_ids(backbone)
    names = {id(p): n for n, p in backbone.named_parameters()}
    buckets = {}
    for p in group['params']:
        n = names[id(p)]
        k = ids[n] if decay is not None else 0
        nd = bool(norm_bias and no_decay(n, p))
        buckets.setdefault((k, nd), []).append(p)
    out = []
    for (k, nd), ps in sorted(buckets.items()):
        g = dict(group, params=ps)
        if decay is not None:
            g['lr'] = group['lr'] * decay ** (depth - k)
        if nd:
            g['weight_decay'] = 0.0
        out.append(g)
    return out


def add_stem_pos_embed(backbone):
    """Zero-initialised learned [1,H,W,C] table added to the (NHWC, normed) patch tokens.

    Class swap (not an instance forward override), so the parameter key is patch_embed.pos_embed and the
    module survives copy.deepcopy (EMA shadow).
    """
    pe = backbone.patch_embed
    if str(getattr(pe, 'output_fmt', '')).split('.')[-1] != 'NHWC':
        raise ValueError('stem abs_pos expects NHWC patch tokens (Swin V1/V2)')
    base = type(pe)

    class PosPatchEmbed(base):
        def forward(self, x):
            return base.forward(self, x) + self.pos_embed

    pe.pos_embed = torch.nn.Parameter(torch.zeros(1, *pe.grid_size, pe.proj.out_channels))
    pe.__class__ = PosPatchEmbed


def build_model(cfg, pretrained=False):
    from kneemri.resnet_depth_e2e import DepthStudyModel

    if not cfg.get('depth_e2e'):
        raise ValueError('Swin route is implemented for the E2E depth study model')
    backbone = build_backbone(cfg)
    if pretrained:
        load_pretrained(backbone, cfg)
    abs_pos = cfg.get('abs_pos')
    if abs_pos not in ABS_POS:
        raise ValueError(f'abs_pos must be one of {ABS_POS}')
    if abs_pos == 'stem':
        add_stem_pos_embed(backbone)
    if cfg.get('compile_backbone', False):
        install_compile(backbone, cfg.get('compile_bucket', 16))
    elif cfg.get('compile_bucket') is not None:
        raise ValueError('compile_bucket requires compile_backbone')
    resize = None if encoder_size(cfg) == int(cfg['resolution']) else encoder_size(cfg)
    stride = 8 * int(cfg.get('patch_size', 4))

    class SwinDepthStudyModel(DepthStudyModel):
        """DepthStudyModel.encode with an NHWC -> NCHW Swin map and an optional in-model resize."""
        input_size = resize
        map_stride = stride

        def __init__(self, backbone, cfg):
            super().__init__(backbone, cfg)
            if cfg.get('abs_pos') == 'final':
                side = (self.input_size or int(cfg['resolution'])) // self.map_stride
                self.final_pos = torch.nn.Parameter(torch.zeros(1, self.feature_dim, side, side))

        def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
            if return_backbone_features:
                raise ValueError('feature replay is not supported with depth')
            if x.ndim != 5 or x.shape[2] != 3 or x.shape[1] == 0:
                raise ValueError('images must have shape [B,K,3,H,W] with K > 0')
            b, k = x.shape[:2]
            valid = self._valid((b, k), window_mask, x.device).flatten()
            side = (self.input_size or x.shape[-1]) // self.map_stride
            if not valid.any():
                return x.new_zeros((b, k, self.feature_dim, side, side))
            images = x.flatten(0, 1)[valid]
            if self.input_size is not None:
                images = F.interpolate(images, size=(self.input_size,) * 2, mode='bilinear', align_corners=False)
            maps = self.backbone.forward_features(images)
            if maps.shape[-1] == self.feature_dim and maps.shape[1] != self.feature_dim:
                maps = maps.permute(0, 3, 1, 2)
            if getattr(self, 'final_pos', None) is not None:
                maps = maps + self.final_pos
            out = maps.new_zeros((b * k, *maps.shape[1:]))
            out[valid] = maps
            return out.view(b, k, *maps.shape[1:])

    return SwinDepthStudyModel(backbone, cfg)
