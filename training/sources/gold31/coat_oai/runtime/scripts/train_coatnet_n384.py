#!/usr/bin/env python3
"""CoAtNet PF0.959 full-strength trainer for native single-resample corpora.

train_coatnet_fullstrength.py unchanged, plus two optional training-bag hooks:
  * ``degrade_*`` (kneemri.native_aug): training-only resolution degradation;
  * ``series_block_cap``: whole-series training bags (resnet_depth_e2e.DepthSeriesBags,
    the ResNet E2E sampler) instead of 12 random windows;
  * ``depth_e2e``: resnet_depth_e2e.DepthStudyModel around the CoAtNet backbone (the
    ResNet study model *is* CoAtNetAblationModel, so only the depth context is added
    between the final maps and the spatial queries, as in the LB .961 ResNet E2E_D);
  * ``drop_path_rate``: timm stochastic depth inside the CoAtNet backbone (linear per
    block, training only; DropPath has no parameters, so pretrained loading and the
    inference model are unchanged).
Evaluation and inference always keep every native window; with neither key set this
is exactly the full-strength trainer.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_coatnet_fullstrength as base
import kneemri.coatnet_original_data as original_data
from kneemri import native_aug
from kneemri import resnet_depth_e2e as e2e
from kneemri import coat_e2e

trainer = base.trainer
_base_receipt = base.source_receipt
_base_build = trainer.build_model


def _build_backbone_model(cfg, pretrained):
    rate = cfg.get('drop_path_rate')
    if not rate:
        return _base_build(cfg, pretrained=pretrained)
    import timm
    create = timm.create_model
    timm.create_model = lambda *args, **kwargs: create(*args, drop_path_rate=float(rate), **kwargs)
    try:
        return _base_build(cfg, pretrained=pretrained)
    finally:
        timm.create_model = create


def build_model(cfg, pretrained=False):
    model = _build_backbone_model(cfg, pretrained)
    if not cfg.get('depth_e2e'):
        return model
    # Same construction as resnet_depth_e2e.build_model: the (pretrained, grad-
    # checkpointed) backbone object is reused, the MIL head is freshly initialised.
    # Shape-bucketed encoder (kneemri.coat_e2e): exact in eval, avoids per-shape cuDNN plans.
    return coat_e2e.BucketedDepthStudyModel(model.backbone, cfg)


def source_receipt(output, cfg, cohort, synthetic=False):
    receipt = _base_receipt(output, cfg, cohort, synthetic)
    for name in ('src/kneemri/native_aug.py', 'src/kneemri/coat_e2e.py', 'src/kneemri/resnet_depth.py',
                 'src/kneemri/resnet_depth_e2e.py', 'scripts/train_coatnet_n384.py'):
        source = ROOT / name
        receipt['sources'][name] = trainer.hash_file(source)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    receipt['training_bags'] = (dict(sampling='whole_series', cap=cfg['series_block_cap'])
                                if cfg.get('series_block_cap') else dict(sampling='random_windows'))
    receipt['training_degradation'] = {k: cfg.get(k) for k in sorted(native_aug.DEGRADE_KEYS)}
    if cfg.get('depth_e2e'):
        receipt['depth_e2e'] = cfg['depth_e2e']
    if cfg.get('drop_path_rate'):
        receipt['drop_path_rate'] = cfg['drop_path_rate']
    return receipt


def main():
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS | native_aug.DEGRADE_KEYS)
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS | coat_e2e.COAT_E2E_MODEL_KEYS | {'drop_path_rate'})
    trainer.build_model = build_model
    base.source_receipt = source_receipt
    import yaml, json
    args = sys.argv[1:]
    path = Path(args[args.index('--config') + 1])
    text = path.read_text()
    cfg = json.loads(text) if path.suffix == '.json' else yaml.safe_load(text)
    bags = e2e.DepthSeriesBags if cfg.get('series_block_cap') else original_data.CoAtNetOriginalNeighborBags
    original_data.CoAtNetOriginalNeighborBags = native_aug.with_degradation(bags)
    base.main()


if __name__ == '__main__':
    main()
