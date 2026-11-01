"""Model stack for pipeline sections 9-13.

Hierarchy (cross-series after within-series, as designed):

    2.5D stacks [B,S,K,N,3,H,W]                      (section 9)
      -> timm DINOv3 slice encoder -> [B*S*K*N, D_enc]   (section 10)
      -> + structural embeddings (position/slot/plane/fluid/fat-sup)  (sec. 12)
      -> within-series Transformer over N -> [B,S,K,N,D_model]        (sec. 13)
      -> 12 pathology queries cross-attend ALL slice-level tokens      (cross-series)
         (memory mask = series_pad | slice_pad; no pooling by default)
      -> 12 logits

Attention uses torch.nn.functional.scaled_dot_product_attention (never a
hand-written softmax(QK^T)V); masks are functional — corrupting a padded
token must not change the output (enforced by tests).

The aggregator (structural embeddings -> within-series -> queries) runs on
CACHED DINO features (section 11), so aggregator training never runs DINO.
`KneeStudyModel` is the end-to-end wrapper used at inference time.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from kneemri.config import (abspath, EncoderConfig, ModelConfig, PathologyDecoderConfig,
                            SeriesTransformerConfig)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PLANE_CODE = {"sagittal": 0, "coronal": 1, "axial": 2}
CODE_UNKNOWN = 255


# ------------------------------------------------------------------ encoders
class SliceEncoder(nn.Module):
    """(N, 3, H, W) float images in [0,1] -> (N, D_enc) slice embeddings."""

    out_dim: int = 0


class TinyConvEncoder(SliceEncoder):
    """Small Conv/GAP encoder — tests + CPU sanity runs, no weights needed."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(image_float(x))

    def forward_tokens(self, x):
        spatial = self.net[:-2](image_float(x)).flatten(2).transpose(1, 2)
        return spatial.mean(1), spatial


def image_float(x: torch.Tensor) -> torch.Tensor:
    """Public encoder input: uint8 [0,255] or float [0,1], exactly one scaling."""
    if x.dtype == torch.uint8:
        return x.float().div(255.0)
    if not x.is_floating_point():
        raise TypeError("images must be uint8 [0,255] or floating point [0,1]")
    return x


class TimmSliceEncoder(SliceEncoder):
    """timm backbone (section 10): DINOv3 ViT-S by default.

    The requested backbone must exist in timm.list_models — if not, we raise
    listing the available dinov3 names instead of silently switching to a
    different architecture. Progressive unfreezing: `unfreeze_last_n_blocks`
    top ViT blocks (0 | 2 | 4 | ... | 'all'); the rest stays frozen.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        import timm

        self.cfg = cfg
        if cfg.name != "timm":
            raise ValueError(f"TimmSliceEncoder got name={cfg.name!r}")
        available = timm.list_models("*dinov3*") if "dinov3" in cfg.backbone_name \
            else timm.list_models(cfg.backbone_name.split("_")[0] + "*")
        if cfg.backbone_name not in available:
            raise ValueError(
                f"backbone '{cfg.backbone_name}' not in timm; available dinov3 "
                f"models: {available}")
        self.backbone = timm.create_model(cfg.backbone_name,
                                          pretrained=cfg.pretrained and not bool(cfg.weights),
                                          num_classes=0,
                                          img_size=cfg.image_size)
        if cfg.weights:
            state = torch.load(abspath(cfg.weights), map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state)
        self.out_dim = int(self.backbone.num_features)
        if cfg.feature_pool not in ('cls', 'cls_patch_mean'):
            raise ValueError('feature_pool must be cls or cls_patch_mean')
        if cfg.feature_pool == 'cls_patch_mean':
            self.out_dim *= 2
        pretrained_cfg = self.backbone.pretrained_cfg
        mean = cfg.mean if cfg.mean is not None else pretrained_cfg.get('mean', IMAGENET_MEAN)
        std = cfg.std if cfg.std is not None else pretrained_cfg.get('std', IMAGENET_STD)
        if len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError('encoder normalization requires 3 means and 3 positive stds')
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.apply_freeze(cfg.freeze_backbone, cfg.unfreeze_last_n_blocks)

    # ------------------------------------------------------------------
    def _backbone_blocks(self) -> nn.ModuleList:
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        raise AttributeError(f"cannot enumerate blocks of {type(self.backbone)}")

    def apply_freeze(self, freeze: bool, unfreeze_last_n: int | str) -> None:
        """freeze_backbone + unfreeze_last_n_blocks (0|2|4|...|'all')."""
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)
        if not freeze:
            return
        blocks = self._backbone_blocks()
        if unfreeze_last_n == "all":
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            return
        n = int(unfreeze_last_n)
        if not 0 <= n <= len(blocks):
            raise ValueError(f"unfreeze_last_n_blocks must be between 0 and {len(blocks)}")
        for block in blocks[len(blocks) - n:] if n > 0 else []:
            for p in block.parameters():
                p.requires_grad_(True)
        if n > 0 and hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():   # final layernorm
                p.requires_grad_(True)

    def unfrozen_parameter_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.backbone.eval()
            for block in self._backbone_blocks():
                if any(p.requires_grad for p in block.parameters()):
                    block.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.feature_pool == 'cls_patch_mean':
            cls, patch = self.forward_tokens(x)
            return torch.cat((cls, patch.mean(1)), dim=-1)
        x = (image_float(x) - self.mean) / self.std
        ctx = nullcontext() if any(p.requires_grad for p in self.backbone.parameters()) else torch.no_grad()
        with ctx:
            return self.backbone(x)

    def forward_tokens(self, x: torch.Tensor):
        x = (image_float(x) - self.mean) / self.std
        ctx = nullcontext() if any(p.requires_grad for p in self.backbone.parameters()) else torch.no_grad()
        with ctx:
            tokens = self.backbone.forward_features(x)
        prefix = self.backbone.num_prefix_tokens
        return tokens[:, 0], tokens[:, prefix:]


def build_encoder(cfg: EncoderConfig) -> SliceEncoder:
    if cfg.name == "timm":
        return TimmSliceEncoder(cfg)
    if cfg.name == "tiny_conv":
        return TinyConvEncoder()
    raise ValueError(f"unknown encoder name '{cfg.name}' (timm | tiny_conv)")


# --------------------------------------------- section 12: structural embeds
def fourier_features(pos: torch.Tensor, n_bands: int) -> torch.Tensor:
    """pos: (...,) in [0,1] -> (..., 2*n_bands) sin/cos at geometric freqs."""
    freqs = torch.pi * (2.0 ** torch.arange(n_bands, device=pos.device))
    ang = pos.unsqueeze(-1) * freqs          # (..., n_bands)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class StructuralEmbeddings(nn.Module):
    """token = DINO_feature + slice_position_embedding + slot_embedding
    + plane_embedding + fluid_embedding + fat_sup_embedding + rank_embedding.

    Continuous physical slice position (z normalized to [0,1] within the
    series extent) defaults to Fourier features + linear projection.
    Categorical codes use learned embeddings; 255 = unknown gets its own
    vector (embedding size = n_categories + 1).
    """

    def __init__(self, cfg: ModelConfig, n_slots: int, k_series: int):
        super().__init__()
        d = cfg.d_model
        self.n_bands = cfg.fourier_bands
        if cfg.positional not in ("fourier", "sinusoidal", "mlp"):
            raise ValueError(f"unknown positional '{cfg.positional}'")
        self.positional = cfg.positional
        pos_in = 2 * cfg.fourier_bands if cfg.positional in ("fourier", "sinusoidal") else 1
        self.pos_proj = (nn.Linear(pos_in, d) if cfg.positional != "mlp"
                         else nn.Sequential(nn.Linear(1, d), nn.GELU(),
                                            nn.Linear(d, d)))
        self.slot_emb = nn.Embedding(n_slots, d)
        self.rank_emb = nn.Embedding(k_series, d)
        self.plane_emb = nn.Embedding(4, d)          # 0/1/2 + unknown(3)
        self.fluid_emb = nn.Embedding(3, d)          # 0/1 + unknown(2)
        self.fat_sup_emb = nn.Embedding(3, d)        # 0/1 + unknown(2)

    def forward(self, features: torch.Tensor, z_norm: torch.Tensor,
                slot_ids: torch.Tensor, rank_ids: torch.Tensor,
                plane_codes: torch.Tensor, fluid_codes: torch.Tensor,
                fat_sup_codes: torch.Tensor) -> torch.Tensor:
        """features (B,S,K,N,D); z_norm (B,S,K,N); plane (S,);
        fluid/fat_sup (S,K) int64; code 255 = unknown -> last embedding row."""
        tokens = features
        if self.positional == "fourier":
            pos = fourier_features(z_norm.clamp(0, 1), self.n_bands)
            tokens = tokens + self.pos_proj(pos)
        elif self.positional == "mlp":
            tokens = tokens + self.pos_proj(z_norm.unsqueeze(-1))
        else:  # sinusoidal
            freq = torch.arange(1, self.n_bands + 1, device=z_norm.device)
            ang = z_norm.unsqueeze(-1) * freq * torch.pi
            tokens = tokens + self.pos_proj(torch.cat([ang.sin(), ang.cos()], -1))
        B, S, K, N, d = tokens.shape
        tokens = tokens + self.slot_emb(slot_ids).view(1, S, 1, 1, d)
        tokens = tokens + self.rank_emb(rank_ids).view(1, S, K, 1, d)
        # codes may be study-level (S,) / (S,K), broadcast over the batch, or
        # per-study (B,S) / (B,S,K) — fat-sup in particular can vary per study
        tokens = tokens + self._expand(
            self.plane_emb(plane_codes.clamp(max=3)), B, S, 1)
        tokens = tokens + self._expand(
            self.fluid_emb(self._unknown_to_extra(fluid_codes)), B, S, K)
        tokens = tokens + self._expand(
            self.fat_sup_emb(self._unknown_to_extra(fat_sup_codes)), B, S, K)
        return tokens

    @staticmethod
    def _expand(emb: torch.Tensor, B: int, S: int, K: int) -> torch.Tensor:
        """Broadcast a categorical embedding to (B, S, K, 1, d)."""
        d = emb.shape[-1]
        if emb.dim() == 2:                        # (S, d) study-level
            return emb.view(1, S, 1, 1, d).expand(B, S, K, 1, -1)
        if emb.dim() == 3:
            if emb.shape[1] == S:                 # (B, S, d) per-study
                return emb.view(B, S, 1, 1, d).expand(B, S, K, 1, -1)
            return emb.view(1, S, K, 1, d).expand(B, S, K, 1, -1)  # (S, K, d)
        if emb.dim() == 4:                        # (B, S, K, d) per-study per-rank
            return emb.unsqueeze(3)
        raise ValueError(f"unexpected embedding shape {tuple(emb.shape)}")

    @staticmethod
    def _unknown_to_extra(codes: torch.Tensor) -> torch.Tensor:
        """255 (=unknown) -> last embedding row; 0/1 stay."""
        return torch.where(codes == CODE_UNKNOWN,
                           torch.full_like(codes, 2), codes)


# ------------------------------------- section 13: SDPA pre-norm transformer
class SDPAAttention(nn.Module):
    """Multi-head attention dispatched through F.scaled_dot_product_attention.
    key_padding: (B, Nk) bool, True = EXCLUDED from attention."""

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.heads, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor,
                key_padding: torch.Tensor | None = None) -> torch.Tensor:
        B, Nq, d = x_q.shape
        Nk = x_kv.shape[1]
        q = self.q(x_q).view(B, Nq, self.heads, self.dh).transpose(1, 2)
        kv = self.kv(x_kv).view(B, Nk, 2, self.heads, self.dh)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        attn_mask = None
        if key_padding is not None:
            # SDPA bool mask semantics: True = ALLOWED to attend
            attn_mask = (~key_padding)[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Nq, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block (self- or cross-attention) + MLP, GELU, dropout."""

    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float,
                 cross: bool = False):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d) if cross else None
        self.attn = SDPAAttention(d, heads, dropout)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(ff_mult * d, d),
                                nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None,
                key_padding: torch.Tensor | None = None) -> torch.Tensor:
        kv = self.norm_kv(memory) if self.norm_kv is not None else self.norm_q(x)
        h = x + self.attn(self.norm_q(x), kv, key_padding=key_padding)
        return h + self.ff(self.norm_ff(h))


class WithinSeriesTransformer(nn.Module):
    """[B*M, N, D] slice tokens -> [B*M, N, D]; padded slices excluded."""

    def __init__(self, cfg: SeriesTransformerConfig, d: int, ff_mult: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.heads, ff_mult, cfg.dropout, cross=False)
            for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor,
                key_padding: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, key_padding=key_padding)
        return self.norm(tokens)


class QueryDecoder(nn.Module):
    """Section 15: pathology-specific learned queries.

    Q in R^(num_queries, D); per layer: [optional query self-attention]
    + cross-attention (Q = pathology queries, K/V = MRI tokens) + residual +
    LayerNorm + MLP. Each query learns WHERE to look — e.g. the ACL query
    attends sagittal tokens, the PF-OA query axial tokens — driven by the
    structural (slot/plane/fluid) embeddings carried on the tokens.
    """

    def __init__(self, cfg: PathologyDecoderConfig, d: int, ff_mult: int):
        super().__init__()
        self.query_self_attention = cfg.query_self_attention
        self.queries = nn.Parameter(torch.randn(cfg.num_queries, d) * 0.02)
        self.self_blocks = nn.ModuleList([
            TransformerBlock(d, cfg.heads, ff_mult, cfg.dropout, cross=False)
            for _ in range(cfg.layers)]) \
            if cfg.query_self_attention else None
        self.cross_blocks = nn.ModuleList([
            TransformerBlock(d, cfg.heads, ff_mult, cfg.dropout, cross=True)
            for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, memory: torch.Tensor,
                memory_padding: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for i in range(len(self.cross_blocks)):
            if self.self_blocks is not None:
                q = self.self_blocks[i](q)          # queries talk to each other
            q = self.cross_blocks[i](q, memory, key_padding=memory_padding)
        return self.norm(q)


# ----------------------------------------------------------------- aggregator
class AttentionMILAggregator(nn.Module):
    """Small per-disease attention baseline on the existing physical 2.5D reader.

This controlled alternative keeps all candidate/slice masks and the backbone
interface, while removing the unverified temporal and hierarchical transformers.
It supports both CLS and concatenated CLS/patch-mean features.
"""
    def __init__(self, cfg: ModelConfig, n_slots: int, k_series: int, encoder_dim: int):
        super().__init__()
        self.n_slots, self.k_series = n_slots, k_series
        c = cfg.n_pathologies
        self.norm = nn.LayerNorm(encoder_dim)
        self.attention = nn.Sequential(nn.Linear(encoder_dim, 256), nn.Tanh(),
                                       nn.Dropout(cfg.pathology_decoder.dropout), nn.Linear(256, c))
        self.head_weight = nn.Parameter(torch.empty(c, encoder_dim))
        self.head_bias = nn.Parameter(torch.zeros(c))
        nn.init.trunc_normal_(self.head_weight, std=0.02)

    def forward(self, features, z_norm, slice_pad, series_pad, plane_codes, fluid_codes, fat_sup_codes):
        b, s, k, n, d = features.shape
        if (s, k) != (self.n_slots, self.k_series):
            raise ValueError('attention MIL slot layout differs from its checkpoint')
        pad = (slice_pad | series_pad.unsqueeze(-1)).reshape(b, -1)
        # Mask before normalization: padding may contain arbitrary values or NaNs.
        h = self.norm(features.float().reshape(b, -1, d).masked_fill(pad[..., None], 0))
        score = self.attention(h).float().masked_fill(pad[..., None], torch.finfo(torch.float32).min)
        weight = score.softmax(1).masked_fill(pad[..., None], 0)
        pooled = torch.einsum('btc,btd->bcd', weight, h)
        logits = (pooled * self.head_weight[None]).sum(-1) + self.head_bias
        return logits.masked_fill(pad.all(1)[:, None], 0)


def build_aggregator(cfg, n_slots, k_series, encoder_dim):
    if cfg.aggregation == 'attention_mil':
        return AttentionMILAggregator(cfg, n_slots, k_series, encoder_dim)
    if cfg.aggregation == 'transformer':
        return SeriesAggregator(cfg, n_slots, k_series, encoder_dim)
    raise ValueError(f'unknown aggregator {cfg.aggregation}')


class SeriesAggregator(nn.Module):
    """Runs on cached DINO features (section 11): aggregator training never
    touches the image encoder.

    Inputs (all produced by the feature cache / dataset):
        features     (B,S,K,N,D_enc) float (cached DINO embeddings)
        z_norm       (B,S,K,N)       physical slice position in [0,1]
        slice_pad    (B,S,K,N)       True = duplicated/resampled token
        series_pad   (B,S,K)         True = absent/invalid series
        slot_ids (S,), rank_ids (S,K), plane/fluid/fat_sup codes
    """

    def __init__(self, cfg: ModelConfig, n_slots: int, k_series: int,
                 encoder_dim: int):
        super().__init__()
        self.cfg = cfg
        self.n_slots, self.k_series = n_slots, k_series
        if cfg.study_tokens.mode not in ("a", "b"):
            raise ValueError(
                f"unknown study_tokens.mode '{cfg.study_tokens.mode}' (a|b)")
        d = cfg.d_model
        self.input_proj = nn.Linear(encoder_dim, d)
        self.structural = StructuralEmbeddings(cfg, n_slots, k_series)
        self.within_series = WithinSeriesTransformer(
            cfg.series_transformer, d, cfg.ff_mult)
        self.query_decoder = QueryDecoder(cfg.pathology_decoder, d, cfg.ff_mult)
        self.head = nn.Linear(d, 1)
        if cfg.independent_heads:
            self.head = nn.Linear(d, cfg.n_pathologies)
        if cfg.pathology_decoder.num_queries != cfg.n_pathologies:
            raise ValueError('one query is required per pathology')
        if cfg.embedding_init_std is not None:
            for module in self.structural.modules():
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=cfg.embedding_init_std)

    def forward(self, features: torch.Tensor, z_norm: torch.Tensor,
                slice_pad: torch.Tensor, series_pad: torch.Tensor,
                plane_codes: torch.Tensor, fluid_codes: torch.Tensor,
                fat_sup_codes: torch.Tensor) -> torch.Tensor:
        B, S, K, N, De = features.shape
        if (S, K) != (self.n_slots, self.k_series):
            raise ValueError(
                f"aggregator built for (S,K)={(self.n_slots, self.k_series)} "
                f"but got {(S, K)} — rebuild with the cache layout")
        device = features.device
        tokens = self.input_proj(features)
        slot_ids = torch.arange(S, device=device)
        rank_ids = torch.arange(K, device=device).unsqueeze(0).expand(S, K)
        tokens = self.structural(tokens, z_norm, slot_ids, rank_ids,
                                 plane_codes, fluid_codes, fat_sup_codes)

        # --- within-series transformer (section 13): padded slices excluded
        pad = slice_pad | series_pad.unsqueeze(-1)      # (B,S,K,N)
        h = tokens.reshape(B * S * K, N, -1)
        h = self.within_series(h, pad.reshape(B * S * K, N))
        h = h.reshape(B, S, K, N, -1)
        h = torch.nan_to_num(h, nan=0.0)

        # --- section 14: study-level MRI tokens -------------------------
        # mode b (default): keep every slice token [B, S*K*N, D] — the
        # pathology queries can point at a specific series AND slice region.
        # mode a: pool each series into one token [B, S*K, D] (average view).
        if self.cfg.study_tokens.mode == "b":
            memory = h.reshape(B, S * K * N, -1)
            mem_pad = pad.reshape(B, S * K * N)
        elif self.cfg.study_tokens.mode == "a":
            valid = (~pad).unsqueeze(-1).to(h.dtype)
            series_tok = (h * valid).sum(3) / valid.sum(3).clamp(min=1)
            memory = series_tok.reshape(B, S * K, -1)
            mem_pad = series_pad.reshape(B, S * K)
        else:
            raise ValueError(
                f"unknown study_tokens.mode '{self.cfg.study_tokens.mode}'")

        # --- section 15: 12 pathology queries cross-attend the MRI tokens
        decoded = self.query_decoder(memory, mem_pad)    # (B, 12, d)
        if self.cfg.independent_heads:
            return (decoded * self.head.weight[None]).sum(-1) + self.head.bias
        return self.head(decoded).squeeze(-1)            # (B, 12)


# ------------------------------------------------------- end-to-end wrapper
class KneeStudyModel(nn.Module):
    """Images -> DINOv3 -> SeriesAggregator. Used at inference (and for the
    feature-cache build); aggregator training uses cached features directly."""

    def __init__(self, cfg: ModelConfig, n_slots: int, k_series: int,
                 slices_per_slot: int):
        super().__init__()
        self.cfg = cfg
        self.n_slots, self.k_series, self.k_slices = n_slots, k_series, slices_per_slot
        self.encoder = build_encoder(cfg.encoder)
        self.aggregator = build_aggregator(cfg, n_slots, k_series,
                                          self.encoder.out_dim)

    def forward(self, images: torch.Tensor, z_norm: torch.Tensor,
                slice_pad: torch.Tensor, series_pad: torch.Tensor,
                plane_codes: torch.Tensor, fluid_codes: torch.Tensor,
                fat_sup_codes: torch.Tensor) -> torch.Tensor:
        B, S, K, N, C, H, W = images.shape
        if (S, K) != (self.n_slots, self.k_series):
            raise ValueError(
                f"model built for (S,K,N)={(self.n_slots, self.k_series, self.k_slices)} "
                f"but got {(S, K, N)} — rebuild with the dataset layout")
        x = images.reshape(B * S * K * N, C, H, W)
        keep = (~(slice_pad | series_pad.unsqueeze(-1))).reshape(-1)
        idx = keep.nonzero().flatten()
        feats = torch.zeros(len(x), self.encoder.out_dim, device=x.device)
        for selected in idx.split(self.cfg.encode_chunk_size):
            if selected.numel() == 0:
                continue
            precision = torch.autocast(device_type=x.device.type, dtype=torch.float16,
                enabled=x.is_cuda) if getattr(self, "cached_precision", False) else nullcontext()
            with precision:
                if self.training and self.cfg.checkpoint_encoder and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    encoded = checkpoint(self.encoder, x[selected], use_reentrant=False).float()
                else:
                    encoded = self.encoder(x[selected]).float()
            feats = feats.index_copy(0, selected, encoded)
        feats = feats.reshape(B, S, K, N, -1)
        if getattr(self, "cached_precision", False):
            feats = feats.to(torch.float16).float()
            # The CLS cache stores positions as FP16 too. Fourier embeddings
            # amplify tiny coordinate changes; mirror the storage rounding.
            z_norm = z_norm.to(torch.float16).float()
        return self.aggregator(feats, z_norm, slice_pad, series_pad,
                               plane_codes, fluid_codes, fat_sup_codes)


def build_model(cfg, n_slots: int, k_series: int,
                slices_per_slot: int) -> KneeStudyModel:
    from kneemri.config import PreprocessPipelineConfig
    assert isinstance(cfg, PreprocessPipelineConfig)
    if cfg.model.architecture == "patch_query":
        from kneemri.hierarchical import PatchQueryModel
        return PatchQueryModel(cfg.model, n_slots, k_series)
    if cfg.model.architecture != "cls":
        raise ValueError(f"unknown model architecture {cfg.model.architecture}")
    return KneeStudyModel(cfg.model, n_slots, k_series, slices_per_slot)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor,
               target_mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none")
    return (loss * target_mask).sum() / target_mask.sum().clamp(min=1.0)
