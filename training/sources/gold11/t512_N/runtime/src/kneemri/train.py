"""Training loop for the aggregator over the DINO feature cache.

    per fold: train on K-1 folds, validate on the held-out fold
    loss    = soft BCE (gold/pseudo weighted) + lambda_rank * pairwise AUC
    metric  = per-target ROC-AUC + macro mean (+ gold-only), never val loss

Everything runs on cached features — DINO is never in the training loop.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kneemri.cache import LABEL_COLS, FeatureCacheReader
from kneemri.config import PreprocessPipelineConfig
from kneemri.losses import CombinedLoss
from kneemri.metrics import evaluate_predictions, format_report
from kneemri.model import SeriesAggregator, build_aggregator


class CachedStudyDataset(Dataset):
    """Tensors for one cached study + loss targets/masks."""

    def __init__(self, cache: FeatureCacheReader, uids: list[str] | None = None,
                 cfg=None, training=False):
        self.cache = cache
        self.cfg,self.training,self.epoch=cfg,training,0
        self.label_table = None
        if cfg:
            from kneemri.labels import LabelTable
            self.label_table = LabelTable(cfg.dataset.labels_csv,cfg.dataset.train_csv,require_files=True)
        uids = uids or cache.study_uids
        self.rows = [cache.uid_to_row(u) for u in uids]
        self.uids = list(uids)
        if self.label_table:
            self.label_table.validate_coverage(self.uids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        it = self.cache.get(self.rows[i])
        gold_mask = np.full(len(it['labels']),it['is_gold'],bool)
        if self.label_table:
            # Labels are a training input, not part of the frozen visual representation.
            # Changing cfg.dataset.labels_csv must actually change the targets.
            it['labels'],gold_mask = self.label_table.get(it['study_uid'])
            it['label_mask'] = np.isfinite(it['labels']).astype(np.float32)
            it['is_gold'] = bool(gold_mask.all())
        if self.cfg:
            import hashlib
            from kneemri.dataset import select_candidates
            seed=int(hashlib.sha256(f'{self.cfg.seed}:{self.epoch}:{it["study_uid"]}'.encode()).hexdigest()[:16],16)
            it["series_valid"] = select_candidates(it["series_valid"],it["quality"],
                self.cfg.dataset.max_series_per_study,np.random.default_rng(seed) if self.training else None)
        if self.training and not it['series_valid'].any():
            it['label_mask'][:] = 0
        return {
            "features": torch.from_numpy(it["features"]),
            "z_norm": torch.from_numpy(it["z_norm"]),
            "slice_pad": torch.from_numpy(it["slice_mask"]),
            "series_pad": ~torch.from_numpy(it["series_valid"]),
            "plane": torch.from_numpy(it["meta_codes"][:, :, 0].astype(np.int64)),
            "fluid": torch.from_numpy(it["meta_codes"][:, :, 1].astype(np.int64)),
            "fs": torch.from_numpy(it["meta_codes"][:, :, 2].astype(np.int64)),
            "targets": torch.from_numpy(it["labels"]),
            "label_mask": torch.from_numpy(it["label_mask"]),
            "is_gold": torch.tensor(it["is_gold"]),
            "gold_mask": torch.from_numpy(gold_mask),
            "study_uid": it["study_uid"],
        }


def move_batch(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def gather_predictions(model: SeriesAggregator, loader: DataLoader,
                       device: str) -> dict:
    """All logits/targets/masks of a split, for metric computation."""
    model.eval()
    logits, targets, masks, gold, uids = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        score = model(batch["features"], batch["z_norm"],
                            batch["slice_pad"], batch["series_pad"],
                            batch["plane"], batch["fluid"],
                            batch["fs"]).float()
        score = score.masked_fill(batch['series_pad'].flatten(1).all(1)[:,None],0)
        logits.append(score.cpu())
        targets.append(batch["targets"].cpu())
        masks.append(batch["label_mask"].cpu())
        gold.append(batch["is_gold"].cpu())
        uids += batch["study_uid"]
    return {
        "logits": torch.cat(logits).numpy(),
        "targets": torch.cat(targets).numpy(),
        "label_mask": torch.cat(masks).numpy(),
        "is_gold": torch.cat(gold).numpy().astype(bool),
        "study_uids": uids,
    }


def train_fold(cfg: PreprocessPipelineConfig, val_fold: int,
               folds_df, out_dir: str, epochs: int = 10,
               batch_size: int = 16, lr: float = 3e-4,
               num_workers: int = 4, device: str | None = None,
               log_every: int = 100) -> dict:
    """Train one fold; returns the metric history and best checkpoint path."""
    if not cfg.model.encoder.freeze_backbone or cfg.model.encoder.unfreeze_last_n_blocks != 0:
        raise ValueError('cached features cannot fine-tune the encoder; use --mode images')
    if cfg.dataset.train_centers_per_series not in (None, cfg.dataset.centers_per_series or cfg.sampling.slices_per_slot):
        raise ValueError('cached training cannot resample original slices; use --mode images')
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    from kneemri.cv import fold_split
    train_uids, val_uids = fold_split(folds_df, val_fold)
    cache = FeatureCacheReader(cfg.dataset.cache_dir)
    if len(cache) != cache.total_declared:
        raise ValueError('feature cache is incomplete; finish cache generation before training')
    if cache.slot_ids != list(cfg.slots.definitions):
        raise ValueError("cache slot manifest differs from training configuration")
    expected_n = cfg.dataset.centers_per_series or cfg.sampling.slices_per_slot
    if cache.slices_per_slot != expected_n or cache.k_series != cfg.slots.k_series_per_slot:
        raise ValueError('cache sampling/layout differs from training configuration')
    provenance = cache.manifest.get('provenance',{})
    for key, expected in [('preprocessing',asdict(cfg.preprocessing)),('sampling',asdict(cfg.sampling))]:
        if key in provenance and provenance[key] != expected:
            raise ValueError(f'cache {key} differs from training configuration')
    if cfg.dataset.channel_mode != provenance.get('channel_mode','adjacent'):
        raise ValueError('cache 2D/2.5D channel construction differs from configuration')
    if cfg.dataset.position_frame != provenance.get('position_frame','physical'):
        raise ValueError('cache coordinate frame differs from configuration')
    missing = set(train_uids + val_uids)-set(cache.study_uids)
    if missing:
        raise ValueError(f'{len(missing)} fold studies are absent from the feature cache')
    train_ds = CachedStudyDataset(cache, [u for u in train_uids
                                          if u in set(cache.study_uids)],cfg=cfg,training=True)
    val_ds = CachedStudyDataset(cache, [u for u in val_uids
                                        if u in set(cache.study_uids)],cfg=cfg)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    model = build_aggregator(cfg.model, n_slots=cache.n_slots,
                            k_series=cache.k_series,
                            encoder_dim=cache.feat_dim).to(device)
    from kneemri.model import build_encoder
    encoder_state = build_encoder(cfg.model.encoder).cpu().state_dict()
    import hashlib
    actual_hash = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in encoder_state.values())).hexdigest()
    if provenance.get('weight_sha256',actual_hash) != actual_hash:
        raise ValueError('cache encoder weights/normalization differ; rebuild visual cache')
    if provenance.get('encoder',{}).get('image_size',cfg.model.encoder.image_size) != cfg.model.encoder.image_size:
        raise ValueError('cache encoder image size differs from configuration')
    if provenance.get('encoder',{}).get('feature_pool','cls') != cfg.model.encoder.feature_pool:
        raise ValueError('cache feature pooling differs from configuration')
    criterion = CombinedLoss(cfg.loss.gold_weight,
                             cfg.loss.pseudo_default_weight,
                             cfg.loss.lambda_rank,cfg.loss.normalization)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * max(1, len(train_dl))))

    history = []
    best_auc, best_path = -1.0, ""
    for epoch in range(epochs):
        train_ds.epoch=epoch
        model.train()
        t0, running = time.time(), []
        for step, batch in enumerate(train_dl):
            batch = move_batch(batch, device)
            logits = model(batch["features"], batch["z_norm"],
                           batch["slice_pad"], batch["series_pad"],
                           batch["plane"], batch["fluid"], batch["fs"])
            loss, parts = criterion(logits, batch["targets"],
                                    batch["label_mask"], batch["gold_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running.append(loss.item())
            if log_every and (step + 1) % log_every == 0:
                print(f"[fold {val_fold}] ep {epoch} step {step + 1}/{len(train_dl)} "
                      f"loss {np.mean(running[-log_every:]):.4f} "
                      f"bce {parts['soft_bce']:.4f} rank {parts['rank']:.4f}",
                      flush=True)

        preds = gather_predictions(model, val_dl, device)
        report = evaluate_predictions(preds["logits"], preds["targets"],
                                      preds["label_mask"], preds["is_gold"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(running)),
                        **report, "elapsed_s": time.time()-t0})
        print(f"[fold {val_fold}] epoch {epoch}: train {np.mean(running):.4f} "
              f"macro AUC {report['macro_auc']:.4f} "
              f"({time.time() - t0:.0f}s)\n{format_report(report)}", flush=True)

        ckpt = os.path.join(out_dir, f"fold{val_fold}_last.pt")
        checkpoint_data = {"schema_version": 2, "architecture": "cls",
                    "pipeline_config": asdict(cfg), "slot_ids": cache.slot_ids,
                    "label_cols": LABEL_COLS, "encoder": encoder_state,
                    "cache_manifest": cache.manifest, "epoch": epoch,
                    "train_uids": train_uids, "val_uids": val_uids,
                    "label_fingerprints": train_ds.label_table.fingerprints,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "metrics": report, "model": model.state_dict(),
                    "config": {"n_slots": cache.n_slots,
                               "k_series": cache.k_series,
                               "encoder_dim": cache.feat_dim}}
        torch.save(checkpoint_data, ckpt)
        if np.isfinite(report["macro_auc"]) and \
                report["macro_auc"] > best_auc:
            best_auc = report["macro_auc"]
            best_path = os.path.join(out_dir, f"fold{val_fold}_best.pt")
            checkpoint_data.update(macro_auc=best_auc, val_uids=val_uids)
            torch.save(checkpoint_data, best_path)
            np.savez(os.path.join(out_dir,f"fold{val_fold}_oof.npz"), **preds)

    with open(os.path.join(out_dir, f"fold{val_fold}_history.json"), "w") as f:
        json.dump({"history": history, "best_macro_auc": best_auc,
                   "val_uids": val_uids}, f, indent=2)
    print(f"[fold {val_fold}] best macro AUC {best_auc:.4f} -> {best_path}")
    return {"best_macro_auc": best_auc, "history": history}
