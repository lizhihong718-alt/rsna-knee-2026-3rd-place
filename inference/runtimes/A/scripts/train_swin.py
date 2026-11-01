#!/usr/bin/env python3
"""Swin E2E_D study model on the native224 / native384 / native512 corpora (single resample, no crop-then-upscale).

train_effnet384.py (R2/E1 trainer: whole-series bags, end-to-end depth, native_aug warp keys,
weight EMA, bit-exact GPU Lanczos-4 training warp) with the encoder factory replaced by
kneemri.swin.build_model (timm Swin V1 / V2 / V2-CR). Pixels are the corpus windows as stored:
resolution must equal the corpus resolution (native224 -> 224, native384 -> 384), so no resampling
happens between the corpus and the network unless ``model_input_size`` is declared (in-model
bilinear resize, identical for training, Gold58 and inference).

Optimizer additions (absent = the shared trainer's single backbone group):
  layer_decay         backbone group split per Swin block, lr * decay**(L+1 - depth);
  no_decay_norm_bias  norms / biases / relative-position tables / logit scales get weight decay 0.
OneCycle uses each group's lr as its peak, so the top block keeps backbone_lr.

swin512 runtime (10-12): + resolution 512 on resnet_native512_crop140; kneemri.gpu_warp from effnet512 runtime_b
(``blur_prob`` / ``blur_sigma`` training-only device blur, ``degrade_before_warp``); kneemri.swin ``fused_attn`` for
Swin V2 (SDPA cosine attention) and training-only ``compile_backbone`` / ``compile_bucket`` (INFRA keys).
"""
import json
from pathlib import Path
import shutil

import torch

import train_resnet
from resnet_fullstrength import install
import kneemri.coatnet_original_data as original_data
from kneemri import gpu_warp
from kneemri import swin
import kneemri.coatnet_training_data as training_data
from kneemri import native_aug
from kneemri import resnet_depth_e2e as e2e

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('src/kneemri/resnet_depth.py', 'src/kneemri/resnet_depth_e2e.py', 'src/kneemri/native_aug.py',
           'src/kneemri/swin.py', 'src/kneemri/gpu_warp.py', 'scripts/train_swin.py',
           'scripts/train_resnet_n384.py')
CORPORA = {224: 'native224_crop140', 384: 'native384_crop140', 512: 'resnet_native512_crop140'}


def source_receipt(output, cfg, cohort, synthetic=False):
    receipt = train_resnet.source_receipt(output, cfg, cohort, synthetic)
    for name in SOURCES:
        source = ROOT / name
        receipt['sources'][name] = train_resnet.trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt.update(schema='swin-study-mil-v1', backbone_family='swin')
    receipt['training_bags'] = dict(sampling='whole_series', cap=cfg.get('series_block_cap', 32))
    receipt['training_degradation'] = {k: cfg.get(k) for k in sorted(native_aug.DEGRADE_KEYS)}
    receipt['input_pixels'] = dict(corpus=cfg['corpus'], resolution=cfg['resolution'], resample='none',
                                   encoder_input=swin.encoder_size(cfg))
    receipt['training_warp'] = dict(interpolation=cfg.get('warp_interpolation', 'linear'),
                                    device=cfg.get('warp_device', 'cpu'), bit_exact_vs='cv2.warpAffine')
    receipt['training_blur'] = {k: cfg.get(k) for k in sorted(gpu_warp.BLUR_KEYS | gpu_warp.ORDER_KEYS)}
    if cfg.get('depth_e2e'):
        receipt['depth_e2e'] = cfg['depth_e2e']
    return receipt


def main():
    trainer = train_resnet.trainer
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS | swin.MODEL_KEYS)
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS | native_aug.DEGRADE_KEYS | swin.DATA_KEYS | gpu_warp.WARP_KEYS
                             | gpu_warp.BLUR_KEYS | gpu_warp.ORDER_KEYS)
    trainer.INFRA_KEYS.update(swin.INFRA | gpu_warp.WARP_KEYS)
    shared_normalize = trainer.normalize_config

    def normalize_config(config):
        cfg = shared_normalize(config)
        value = cfg.get('cudnn_benchmark', False)
        if not isinstance(value, bool):
            raise ValueError('cudnn_benchmark must be boolean')
        torch.backends.cudnn.benchmark = value
        res = int(cfg['resolution'])
        if res not in CORPORA or CORPORA[res] not in str(cfg['corpus']):
            raise ValueError('Swin route: resolution 224/384/512 on native224_crop140/native384_crop140/resnet_native512_crop140')
        if cfg.get('warp_device', 'cpu') not in ('cpu', 'gpu'):
            raise ValueError('warp_device must be cpu or gpu')
        if cfg.get('input_downscale') is not None:
            raise ValueError('Swin route never resamples the corpus windows in the loader')
        return cfg

    trainer.normalize_config = normalize_config
    built = {}

    def build_model(cfg, pretrained=False):
        model = swin.build_model(cfg, pretrained)
        built.update(model=model, cfg=cfg)
        return model

    shared_adamw = torch.optim.AdamW

    def adamw(groups, **kwargs):
        groups = list(groups)
        cfg = built.get('cfg', {})
        decay, norm_bias = cfg.get('layer_decay'), bool(cfg.get('no_decay_norm_bias', False))
        if cfg.get('abs_pos') and 'model' in built:
            # Zero-initialised absolute position tables: head LR, no weight decay, own group.
            named = dict(built['model'].named_parameters())
            pos = [named[n] for n in ('backbone.' + swin.POS_NAMES[0], swin.POS_NAMES[1]) if n in named]
            if len(pos) != 1:
                raise ValueError('expected exactly one absolute position table')
            ids = {id(p) for p in pos}
            groups = [dict(g, params=[p for p in g['params'] if id(p) not in ids]) for g in groups]
            head_lr = groups[-1]['lr']
            extra_pos = [dict(params=pos, lr=head_lr, weight_decay=0.0)]
            print(json.dumps({'event': 'abs_pos_group', 'mode': cfg['abs_pos'], 'lr': head_lr,
                              'numel': int(sum(p.numel() for p in pos))}), flush=True)
        else:
            extra_pos = []
        if (decay is not None or norm_bias) and len(groups) >= 2 and 'model' in built:
            if decay is not None and not 0.0 < float(decay) <= 1.0:
                raise ValueError('layer_decay must lie in (0, 1]')
            backbone = built['model'].backbone
            if {id(p) for p in groups[0]['params']} != {id(p) for n, p in backbone.named_parameters()
                                                      if n != swin.POS_NAMES[0]}:
                raise ValueError('first optimizer group is not the backbone')
            groups = swin.split_backbone_group(backbone, groups[0], None if decay is None else float(decay),
                                               norm_bias) + groups[1:]
            print(json.dumps({'event': 'backbone_groups', 'layer_decay': decay, 'no_decay_norm_bias': norm_bias,
                              'groups': [(round(g['lr'], 9), g.get('weight_decay', 'default'), len(g['params']))
                                         for g in groups]}), flush=True)
        return shared_adamw(groups + extra_pos, **kwargs)

    trainer.torch.optim.AdamW = adamw
    trainer.build_model = build_model
    trainer.source_receipt = source_receipt
    install(trainer)
    # Training, validation and Gold58 datasets all resolve this name at call time; the
    # degradation mixin only acts when train=True. No downscale mixin: windows stay as stored.
    original_data.CoAtNetOriginalNeighborBags = gpu_warp.with_gpu_augment(
        native_aug.with_degradation(e2e.DepthSeriesBags))
    # make_loader imports the collate at call time; device_images is resolved at call time.
    training_data.coatnet_collate = gpu_warp.collate_with_augment(training_data.coatnet_collate)
    trainer.device_images = gpu_warp.device_images_with_augment(trainer.device_images)
    trainer.main()


if __name__ == '__main__':
    main()
