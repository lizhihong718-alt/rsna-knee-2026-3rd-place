#!/usr/bin/env python3
"""Run a CoAtNet ablation checkpoint on a matching original-neighbour corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_coatnet_ablation as trainer
from kneemri.coatnet_model import build_model
from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags
from kneemri.labels import LABEL_COLS
from kneemri.coatnet_training_data import coatnet_collate


def preprocessing_contract(manifest):
    ignored = {"source_processed_dir", "source_train_csv_sha256"}
    return {key: value for key, value in manifest["preprocessing"].items()
            if key not in ignored}


def checkpoint_config(checkpoint, config=None):
    cfg = trainer.normalize_config(checkpoint["config"])
    if checkpoint.get("lab") != LABEL_COLS or checkpoint.get("arch") != cfg["arch"] \
            or checkpoint.get("res") != cfg["resolution"]:
        raise ValueError("checkpoint model contract does not match its config")
    if cfg.get("corpus_format") != "original_neighbors":
        raise ValueError("this runtime requires original_neighbors")
    if config is not None:
        supplied = trainer.normalize_config(config)
        keys = trainer.MODEL_KEYS | {"arch", "resolution", "normalization", "corpus_format",
                                    "eval_windows", "eval_sampling", "eval_amp", "num_classes"}
        changed = sorted(k for k in keys if supplied.get(k) != cfg.get(k))
        if changed:
            raise ValueError(f"inference config differs from checkpoint: {changed}")
    return cfg


def training_manifest(checkpoint, path=None):
    path = Path(path) if path else Path(checkpoint["config"]["corpus"]) / "manifest.json"
    if trainer.hash_file(path) != checkpoint["cohort"]["corpus_manifest_sha256"]:
        raise ValueError("training corpus manifest differs from checkpoint provenance")
    return json.loads(path.read_text())


def validate_corpus(manifest, trained_manifest, ids):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("inference requires nonempty, unique study IDs")
    all_ids = manifest["study_uids"]
    if len(set(all_ids)) != len(all_ids) or not set(ids) <= set(all_ids):
        raise ValueError("requested studies missing from corpus or corpus IDs duplicated")
    if preprocessing_contract(manifest) != preprocessing_contract(trained_manifest):
        raise ValueError("inference corpus preprocessing differs from training (including crop)")
    return all_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ids-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--training-manifest", type=Path)
    ap.add_argument("--encode-chunk", type=int)
    ap.add_argument("--eval-amp", choices=["bf16", "fp16", "off"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint_config(checkpoint, yaml.safe_load(args.config.read_text()))
    trained_manifest = training_manifest(checkpoint, args.training_manifest)
    cfg["corpus"] = str(args.corpus.resolve())
    cfg["eval_workers"] = 0
    if args.encode_chunk is not None:
        if args.encode_chunk < 1:
            raise ValueError("encode-chunk must be positive")
        cfg["eval_encode_chunk"] = args.encode_chunk
    if args.eval_amp:
        cfg["eval_amp"] = args.eval_amp
    torch.set_num_threads(cfg["num_threads"])
    torch.backends.cudnn.benchmark = False
    ids = pd.read_csv(args.ids_csv, dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    all_ids = validate_corpus(manifest, trained_manifest, ids)
    done = np.load(args.corpus / "complete.npy", mmap_mode="r")
    if done.shape != (len(all_ids),) or not done.all():
        raise ValueError("inference corpus is incomplete")
    labels = {uid: np.zeros(12, dtype=np.float32) for uid in ids}
    ds = CoAtNetOriginalNeighborBags(cfg, all_ids, ids, labels, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=coatnet_collate)
    device = torch.device(args.device)
    trainer.seed_all(cfg["seed"])
    model = build_model(cfg, pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dtype = trainer.select_eval_amp(cfg, device)
    values, seen = [], []
    with torch.no_grad():
        for batch in loader:
            logits = trainer.inference_logits(model, batch, device, dtype, cfg["eval_encode_chunk"])
            values.append(logits.numpy())
            seen.extend(batch["study_ids"])
    probabilities = torch.from_numpy(np.concatenate(values)).sigmoid().numpy()
    if seen != ids or not np.isfinite(probabilities).all():
        raise ValueError("prediction ordering or finiteness contract failed")
    result = pd.DataFrame(probabilities, columns=LABEL_COLS)
    result.insert(0, "StudyInstanceUID", seen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    args.out.with_suffix(".receipt.json").write_text(json.dumps({
        "checkpoint_sha256": trainer.hash_file(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "input": preprocessing_contract(manifest),
        "normalization": cfg["normalization"], "eval_windows": cfg["eval_windows"],
        "eval_amp": cfg["eval_amp"], "encode_chunk": cfg["eval_encode_chunk"],
        "studies": len(ids), "submission_sha256": trainer.hash_file(args.out),
        "ensemble": False, "tta": False,
    }, indent=2) + "\n")
    print(f"wrote {args.out}: {result.shape}; checkpoint_epoch={checkpoint.get('epoch', -1)+1}")


if __name__ == "__main__":
    main()
