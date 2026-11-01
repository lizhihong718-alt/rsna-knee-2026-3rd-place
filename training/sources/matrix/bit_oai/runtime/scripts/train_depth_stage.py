#!/usr/bin/env python3
"""Second-stage all-window adaptation of a finished ResNet run, with or without depth.

Starts from a base run's fixed final checkpoint (same cohort, labels, FS loss
constants), freezes the BiT prefix and trains the final stage(s)/norm, the
spatial-query/MIL head and an optional depth module on every original window of
each study (micro-batch 1, effective batch 8, fixed cosine schedule).

Evaluation happens once, after the fixed final epoch: weak validation (when the
base cohort has one) and Gold58, each with depth on and off. Nothing is selected
on Gold58.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import train_resnet  # noqa: E402  (registers the ResNet keys on the shared trainer)
from kneemri.resnet_model import build_model  # noqa: E402
from kneemri.resnet_depth import DepthBags, DepthContext, DepthStageModel  # noqa: E402
from resnet_fullstrength import mass_statistics, weighted_loss  # noqa: E402

trainer = train_resnet.trainer
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def cosine_factor(step, total, warmup_fraction, floor):
    warm = max(1, round(total * warmup_fraction))
    if step < warm:
        return (step + 1) / warm
    phase = (step - warm) / max(1, total - warm)
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * phase))


def macro_auc(logits, targets):
    from sklearn.metrics import roc_auc_score
    per = {}
    for j, name in enumerate(LABELS):
        t = targets[:, j] > .5
        if t.any() and (~t).any():
            per[name] = float(roc_auc_score(t, logits[:, j]))
    return float(np.mean(list(per.values()))), per


def encode_study(model, batch, device, chunk, amp):
    """[1,K] bag -> maps [1,K,C,H,W]; the frozen prefix runs without gradients."""
    metadata = trainer.model_inputs(batch, device)
    mask = metadata['window_mask'][0]
    k = mask.shape[0]
    valid = mask.nonzero().flatten()
    prefix = []
    for start in range(0, len(valid), chunk):
        index = valid[start:start + chunk]
        images = trainer.device_images(batch, device, index.cpu())[0]
        with trainer.amp_context(device, amp):
            prefix.append(model.prefix(images.contiguous(memory_format=torch.channels_last)))
    prefix = torch.cat(prefix)
    with trainer.amp_context(device, amp):
        maps = model.suffix(prefix)
    full = maps.new_zeros((k, *maps.shape[1:]))
    full[valid] = maps
    return full[None], metadata


def evaluate(model, dataset, ids, targets, device, amp, chunk, use_depth_options):
    model.eval()
    out = {flag: [] for flag in use_depth_options}
    loader = trainer.make_loader(dataset, dict(eval_workers=3, batch_size=1, drop_last_train=False,
                                                workers=3), device)
    seen = []
    with torch.no_grad():
        for batch in loader:
            seen += batch['study_ids']
            maps, metadata = encode_study(model, batch, device, chunk, amp)
            for flag in use_depth_options:
                with trainer.amp_context(device, amp):
                    z = model.study_logits(maps, metadata, use_depth=flag).float()
                if not torch.isfinite(z).all():
                    raise FloatingPointError('non-finite evaluation logits')
                out[flag].append(z.cpu().numpy()[0])
    if seen != list(ids):
        raise RuntimeError('evaluation order differs from the cohort')
    result = {}
    for flag, z in out.items():
        z = np.asarray(z, np.float32)
        macro, per = macro_auc(z, targets)
        result['depth_on' if flag else 'depth_off'] = dict(macro=macro, per_class=per, logits=z)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'final.pt').exists():
        raise SystemExit('run already complete')
    base_dir = Path(cfg['base_run'])
    base_cfg = json.loads((base_dir / 'config.json').read_text())
    cohort = json.loads((base_dir / 'cohort.json').read_text())
    for key, value in cfg.get('data_overrides', {}).items():
        base_cfg[key] = value
    if cfg.get('depth_canonical_positions'):
        base_cfg['depth_canonical_positions'] = cfg['depth_canonical_positions']
    strategy = cohort['validation_strategy']
    train_ids = cohort['trained_study_uids']
    weak_ids = cohort['validation_study_uids'] if strategy == 'grouped_cv' else []
    gold_ids = cohort['gold_study_uids']
    if set(gold_ids) & set(train_ids) or set(weak_ids) & set(train_ids):
        raise ValueError('held-out studies leak into training')
    table = pd.read_csv(base_cfg['labels_csv'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    gold = pd.read_csv(base_cfg['gold_csv'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    y_train = table.loc[train_ids, LABELS].to_numpy(np.float32)
    y_weak = table.loc[weak_ids, LABELS].to_numpy(np.float32) if weak_ids else None
    y_gold = gold.loc[gold_ids, LABELS].to_numpy(np.float32)
    if not np.isfinite(y_gold).all():
        raise ValueError('Gold58 labels incomplete')
    pos_weight, normalizer = mass_statistics(y_train, trainer, base_cfg)
    np.testing.assert_allclose(normalizer, base_cfg['mass_normalizer'], rtol=0, atol=1e-9)

    seed = int(cfg.get('seed', 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    amp = torch.bfloat16
    eval_amp = None if cfg.get('eval_fp32', True) else torch.float16
    start = time.time()

    checkpoint = torch.load(base_dir / 'final.pt', map_location='cpu', weights_only=False)
    if base_cfg.get('depth_e2e'):
        from kneemri.resnet_depth_e2e import build_model as build_e2e
        base = build_e2e(base_cfg, pretrained=False)
    else:
        base = build_model(base_cfg, pretrained=False)
    base.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    depth = None
    if cfg['use_depth']:
        d = cfg['depth']
        depth = DepthContext(base.feature_dim, width=d.get('width', 32), heads=d.get('heads', 4),
                             regions=d.get('regions', 2), absolute=d.get('absolute', False),
                             cross_series=d.get('cross_series', False))
    model = DepthStageModel(base, depth, cfg.get('trainable_stages', [3])).to(device)
    model.base.backbone.to(memory_format=torch.channels_last)
    groups = model.parameter_groups(cfg['suffix_lr'], cfg['spatial_lr'], cfg['depth_lr'])
    params = [p for g in groups for p in g['params']]
    counts = {g['role']: sum(p.numel() for p in g['params']) for g in groups}
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.get('weight_decay', .01))

    manifest = json.loads((Path(base_cfg['corpus']) / 'manifest.json').read_text())
    all_ids = manifest['study_uids']
    labels = {u: table.loc[u, LABELS].to_numpy(np.float32) for u in train_ids + weak_ids}
    labels.update({u: y_gold[i] for i, u in enumerate(gold_ids)})
    augment = bool(cfg.get('augment', False))
    data_cfg = dict(base_cfg, eval_windows='all', eval_sampling='uniform', train_windows='all',
                    train_sampling='random')
    train_set = DepthBags(data_cfg, all_ids, train_ids, labels, train=augment, uint8_output=True)
    weak_set = DepthBags(data_cfg, all_ids, weak_ids, labels, train=False, uint8_output=True) if weak_ids else None
    gold_set = DepthBags(data_cfg, all_ids, gold_ids, labels, train=False, uint8_output=True)

    n_train = 16 if args.smoke else len(train_ids)
    epochs = 1 if args.smoke else int(cfg['epochs'])
    accumulation = 8
    total = math.ceil(n_train / accumulation) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_factor(s, total, cfg.get('warmup_fraction', .05), cfg.get('final_lr_ratio', .02)))
    pw = torch.tensor(pos_weight, device=device)
    loss_cfg = dict(base_cfg, mass_normalizer=list(map(float, normalizer)))
    chunk = int(cfg.get('encode_chunk', 32))
    history, updates = [], 0
    save_json(out / 'run.json', dict(config=cfg, base_config=base_cfg, trainable=counts, total_updates=total,
                                     train=n_train, weak=len(weak_ids), gold=len(gold_ids),
                                     base_final_sha256=sha(base_dir / 'final.pt')))
    print(json.dumps(dict(event='start', trainable=counts, total_updates=total, train=n_train)), flush=True)
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_ids))[:n_train]
        loader = trainer.make_loader(Subset(train_set, order.tolist()),
                                     dict(eval_workers=cfg.get('workers', 3), batch_size=1, drop_last_train=False,
                                          workers=cfg.get('workers', 3)), device)
        loss_sum, windows, begin, grad_max = 0., 0, time.time(), 0.
        for j, batch in enumerate(loader):
            if j % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
                group_size = min(accumulation, n_train - j)
            if batch['study_ids'] != [train_ids[order[j]]]:
                raise RuntimeError('training order differs from the sampled permutation')
            target = torch.from_numpy(y_train[order[j]:order[j] + 1]).to(device)
            maps, metadata = encode_study(model, batch, device, chunk, amp)
            with trainer.amp_context(device, amp):
                z = model.study_logits(maps, metadata).float()
            loss = weighted_loss(z, target, torch.ones_like(target), pw, loss_cfg,
                                 group_denominators=torch.ones(12, device=device)) / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            loss.backward()
            loss_sum += float(loss.detach()) * group_size
            windows += int(metadata['window_mask'].sum())
            if (j + 1) % accumulation == 0 or j + 1 == n_train:
                grad = float(torch.nn.utils.clip_grad_norm_(params, cfg.get('gradient_clip', 1.),
                                                            error_if_nonfinite=True))
                grad_max = max(grad_max, grad)
                optimizer.step()
                scheduler.step()
                updates += 1
                if updates % 50 == 0:
                    print(json.dumps(dict(event='step', epoch=epoch + 1, updates=updates, studies=j + 1,
                                          loss=loss_sum / (j + 1), elapsed_s=round(time.time() - start, 1))),
                          flush=True)
        row = dict(epoch=epoch + 1, updates=updates, train_loss=loss_sum / n_train, windows=windows,
                   grad_norm_max=grad_max, epoch_s=time.time() - begin,
                   lr=[g['lr'] for g in optimizer.param_groups])
        history.append(row)
        save_json(out / 'history.json', history)
        state = {n: v.detach().cpu() for n, v in model.state_dict().items()
                 if not n.startswith('base.backbone.') or any(
                     n.startswith(f'base.backbone.stages.{i}.') for i in model.trainable_stages)
                 or n.startswith('base.backbone.norm.')}
        torch.save(dict(epoch=epoch + 1, updates=updates, model=state, config=cfg, history=history),
                   out / 'last.partial.pt')
        (out / 'last.partial.pt').replace(out / 'last.pt')
        print(json.dumps(dict(event='epoch', **row)), flush=True)
    if updates != total:
        raise RuntimeError('schedule incomplete')
    (out / 'last.pt').replace(out / 'final.pt')
    result = dict(config=cfg, trainable=counts, history=history, train_s=time.time() - start)
    options = [True, False] if (depth is not None or getattr(model.base, 'depth', None) is not None) else [False]
    evals = []
    if weak_set is not None:
        evals.append(('weak', weak_set, weak_ids if not args.smoke else weak_ids, y_weak))
    evals.append(('gold58', gold_set, gold_ids, y_gold))
    for name, dataset, ids, targets in evals:
        if args.smoke:
            dataset = Subset(dataset, list(range(8)))
            ids, targets = ids[:8], targets[:8]
        res = evaluate(model, dataset, ids, targets, device, eval_amp, chunk, options)
        for key, value in res.items():
            np.savez(out / f'{name}_{key}.npz', study_uids=np.array(ids), logits=value['logits'],
                     targets=targets, label_names=np.array(LABELS))
            result[f'{name}_{key}'] = dict(macro=value['macro'], per_class=value['per_class'])
        print(json.dumps({f'{name}_{k}': v['macro'] for k, v in res.items()}), flush=True)
    result['elapsed_s'] = time.time() - start
    result['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    save_json(out / 'result.json', result)


if __name__ == '__main__':
    main()
