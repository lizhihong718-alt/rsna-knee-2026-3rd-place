"""Patch -> pathology -> slices -> same-slot series -> cross-slot fusion."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from kneemri.model import build_encoder, TransformerBlock, SDPAAttention, WithinSeriesTransformer


class DiseasePool(nn.Module):
    """Each disease attends its own feature stream; all streams share weights."""
    def __init__(self, labels, d, heads, dropout):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(labels,d)*0.02)
        self.attention = SDPAAttention(d,heads,dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, features, padding):
        # features [B,L,C,D], padding [B,L]
        B,L,C,D = features.shape
        x = features.permute(0,2,1,3).reshape(B*C,L,D)
        pad = padding[:,None,:].expand(B,C,L).reshape(B*C,L)
        x = x.masked_fill(pad.unsqueeze(-1),0)
        q = self.queries[None].expand(B,C,D).reshape(B*C,1,D)
        out = self.norm(q + self.attention(q,self.norm(x),pad))
        out = out.reshape(B,C,D)
        return out.masked_fill(padding.all(-1)[:,None,None],0)


class PatchQueryModel(nn.Module):
    def __init__(self, cfg, n_slots, k_series):
        super().__init__()
        self.cfg,self.n_slots,self.k_series=cfg,n_slots,k_series
        d,C=cfg.d_model,cfg.n_pathologies
        if cfg.pathology_decoder.num_queries != C:
            raise ValueError('one query is required per pathology')
        if cfg.encoder.feature_pool != 'cls':
            raise ValueError('patch_query consumes separate CLS/patch tokens; use feature_pool=cls')
        self.encoder=build_encoder(cfg.encoder)
        self.visual_proj=nn.Linear(self.encoder.out_dim,d)
        self.patch_xy=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,d))
        self.queries=nn.Parameter(torch.randn(C,d)*0.02)
        self.spatial_blocks=nn.ModuleList([TransformerBlock(d,cfg.pathology_decoder.heads,cfg.ff_mult,cfg.pathology_decoder.dropout,cross=True) for _ in range(cfg.spatial_layers)])
        self.cls_gate=nn.Parameter(torch.tensor(-2.0))
        self.slot_emb=nn.Embedding(n_slots,d)
        self.rank_emb=nn.Embedding(k_series,d)
        self.plane_emb=nn.Embedding(4,d)
        self.fluid_emb=nn.Embedding(3,d)
        self.fs_emb=nn.Embedding(3,d)
        self.position=nn.Sequential(nn.Linear(7,d),nn.GELU(),nn.Linear(d,d))
        self.quality=nn.Sequential(nn.Linear(8,d),nn.GELU(),nn.Linear(d,d))
        self.slice_transformer=WithinSeriesTransformer(cfg.series_transformer,d,cfg.ff_mult)
        heads,drop=cfg.pathology_decoder.heads,cfg.pathology_decoder.dropout
        self.slice_pool=DiseasePool(C,d,heads,drop)
        self.series_pool=DiseasePool(C,d,heads,drop)
        self.slot_pool=DiseasePool(C,d,heads,drop)
        self.head_weight=nn.Parameter(torch.randn(C,d)/math.sqrt(d))
        self.head_bias=nn.Parameter(torch.zeros(C))
        self.missing_logits=nn.Parameter(torch.zeros(C))
        if cfg.embedding_init_std is not None:
            for module in (self.slot_emb,self.rank_emb,self.plane_emb,self.fluid_emb,self.fs_emb):
                nn.init.normal_(module.weight,std=cfg.embedding_init_std)

    def _spatial(self, images, pixel_valid, metadata):
        cls,patch=self.encoder.forward_tokens(images)
        patch=self.visual_proj(patch)
        P=patch.shape[1];side=math.isqrt(P)
        if side*side!=P:
            raise ValueError('expected a square patch grid')
        y,x=torch.meshgrid(torch.linspace(-1,1,side,device=patch.device),torch.linspace(-1,1,side,device=patch.device),indexing='ij')
        xy=torch.stack((x,y),-1).reshape(P,2)
        patch=patch+self.patch_xy(xy).unsqueeze(0)
        patch_pad=F.adaptive_avg_pool2d(pixel_valid[:,None].float(),(side,side)).flatten(1)<=0
        q=self.queries[None].expand(len(images),-1,-1)+metadata[:,None,:]
        for layer in self.spatial_blocks:
            q=layer(q,patch,key_padding=patch_pad)
        if self.cfg.global_residual:
            q=q+torch.sigmoid(self.cls_gate)*self.visual_proj(cls)[:,None,:]
        return q

    def forward(self, batch):
        images=batch['images']; B,S,K,N,_,H,W=images.shape
        if (S,K)!=(self.n_slots,self.k_series):
            raise ValueError('slot/rank shape differs from model manifest')
        if H!=self.cfg.encoder.image_size or W!=H:
            raise ValueError('model image size differs from dataset')
        pad=batch['slice_pad']|batch['series_pad'].unsqueeze(-1)
        d,C=self.cfg.d_model,self.cfg.n_pathologies
        device=images.device
        slot=self.slot_emb(torch.arange(S,device=device))[None,:,None,:]
        rank=self.rank_emb(torch.arange(K,device=device))[None,None,:,:]
        code=lambda name,unknown: batch[name].long().clamp(min=0,max=unknown)
        structure=slot+rank+self.plane_emb(code('plane_codes',3))+self.fluid_emb(code('fluid_codes',2))+self.fs_emb(code('fat_sup_codes',2))
        flat_structure=structure.unsqueeze(3).expand(B,S,K,N,d).reshape(-1,d)
        indices=(~pad).reshape(-1).nonzero().flatten()
        image_rows=images.reshape(-1,3,H,W)
        pixel_rows=batch['pixel_valid'].reshape(-1,H,W)
        rows=torch.zeros(B*S*K*N,C,d,device=device)
        for selected in indices.split(self.cfg.encode_chunk_size):
            if selected.numel() == 0:
                continue
            args=(image_rows[selected],pixel_rows[selected],flat_structure[selected])
            if self.training and self.cfg.checkpoint_encoder and torch.is_grad_enabled():
                value=checkpoint(self._spatial,*args,use_reentrant=False)
            else:
                value=self._spatial(*args)
            rows=rows.index_copy(0,selected,value.float())
        h=rows.reshape(B,S,K,N,C,d)
        physical=torch.cat((batch['z_norm'].unsqueeze(-1),batch['physical']),-1).float()
        h=h+self.position(physical).unsqueeze(-2)+structure[:,:, :,None,None,:]
        h=h+self.quality(batch['quality'].float())[:,:,:,None,None,:]
        h=h.masked_fill(pad[...,None,None],0)
        temporal=h.permute(0,1,2,4,3,5).reshape(B*S*K*C,N,d)
        temporal_pad=pad[:,:,:,None,:].expand(B,S,K,C,N).reshape(B*S*K*C,N)
        temporal=self.slice_transformer(temporal,temporal_pad)
        temporal=temporal.masked_fill(temporal_pad[...,None],0)
        temporal=temporal.reshape(B*S*K,C,N,d).permute(0,2,1,3)
        per_series=self.slice_pool(temporal,pad.reshape(B*S*K,N)).reshape(B,S,K,C,d)
        series_pad=pad.all(-1)
        per_slot=self.series_pool(per_series.reshape(B*S,K,C,d),series_pad.reshape(B*S,K)).reshape(B,S,C,d)
        study=self.slot_pool(per_slot,series_pad.all(-1))
        logits=(study*self.head_weight[None]).sum(-1)+self.head_bias
        empty=series_pad.all(-1).all(-1)
        return torch.where(empty[:,None],self.missing_logits[None].expand(B,-1),logits)
