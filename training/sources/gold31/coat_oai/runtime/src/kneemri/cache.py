"""DINO feature cache (pipeline section 11).

Aggregator training must NEVER run DINO, so frozen-encoder features are
precomputed for every study and stored in a consolidated, fixed-stride
memmap store — no tens of thousands of small files, O(1) random access,
FP16 storage (~0.6 MB/study with S=6, K=2, N=32, D=384 -> ~2.6 GB total,
which sits in the page cache).

Files under the cache root:
    features.f16     (n, S, K, N, D)  cached 2.5D-token DINO features
    slice_mask.u1    (n, S, K, N)     1 = duplicated/resampled token (pad)
    series_valid.u1  (n, S, K)        1 = series rank present
    slot_valid.u1    (n, S)           1 = slot has >=1 valid series
    positions.i32    (n, S, K, N)     sampled slice indices
    z_norm.f16       (n, S, K, N)     physical slice position in [0,1]
    meta_codes.u1    (n, S, K, 3)     plane / fluid / fat-sup codes
    index.parquet    study_uid, labels (12), confidence, laterality_consensus
    done.json        resume bookkeeping
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

def build_study_slice_dataset(k_series_per_slot: int, image_size: int,
                              files: list[str], cfg=None, include_quality=False):
    from kneemri.dataset import KneeMRIDataset
    class CacheImages(KneeMRIDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            valid = ~item["series_pad"]
            payload = (item["study_uid"], item["images"].numpy(), item["slice_pad"].numpy(),
                    valid.numpy(), valid.any(-1).numpy(), item["positions"].numpy(),
                    item["z_norm"].numpy(), item["meta_codes"].numpy().astype(np.uint8))
            return (*payload, item["quality"].numpy()) if include_quality else payload
    ds = CacheImages("", cfg=cfg, files=files, resize_to=image_size,
                     k_series=k_series_per_slot, apply_budget=False)
    ds.shape = (ds.n_slots, ds.k_series, ds.slices_per_slot)
    return ds


DTYPES = {
    "quality.f32": np.float32,
    "features.f16": np.float16,
    "slice_mask.u1": np.uint8,
    "series_valid.u1": np.uint8,
    "slot_valid.u1": np.uint8,
    "positions.i32": np.int32,
    "z_norm.f16": np.float16,
    "meta_codes.u1": np.uint8,
}
LABEL_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
              "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
              "Contusion", "Fracture"]


@dataclass
class CacheLayout:
    n_studies: int
    n_slots: int
    k_series: int
    slices_per_slot: int
    feat_dim: int

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        n, S, K, N, D = (self.n_studies, self.n_slots, self.k_series,
                         self.slices_per_slot, self.feat_dim)
        return {
            "quality.f32": (n, S, K, 8),
            "features.f16": (n, S, K, N, D),
            "slice_mask.u1": (n, S, K, N),
            "series_valid.u1": (n, S, K),
            "slot_valid.u1": (n, S),
            "positions.i32": (n, S, K, N),
            "z_norm.f16": (n, S, K, N),
            "meta_codes.u1": (n, S, K, 3),
        }


class FeatureCacheWriter:
    """Create/populate the store. `study_uids` fixes the row order. Resume
    skips studies recorded in done.json (features already written)."""

    def __init__(self, root: str, layout: CacheLayout, study_uids: list[str],
                 resume: bool = True, slot_ids: list[str] | None = None,
                 provenance: dict | None = None):
        self.root = root
        self.layout = layout
        self.study_uids = list(study_uids)
        assert len(self.study_uids) == layout.n_studies
        if len(set(study_uids)) != len(study_uids):
            raise ValueError("duplicate study UID in cache")
        self._uid_to_row = {u: i for i, u in enumerate(study_uids)}
        os.makedirs(root, exist_ok=True)
        manifest = {"schema_version": 2, "normalization": "uint8_to_unit_imagenet",
                    "slot_ids": slot_ids or [f"slot_{i}" for i in range(layout.n_slots)],
                    "label_cols": LABEL_COLS, "study_uids": self.study_uids,
                    "provenance": provenance or {},
                    "n_slots": layout.n_slots, "k_series": layout.k_series,
                    "slices_per_slot": layout.slices_per_slot,
                    "feat_dim": layout.feat_dim, "n_studies": layout.n_studies,
                    "shapes": {k: list(v) for k, v in layout.shapes.items()}}
        manifest_path = os.path.join(root, "layout.json")
        restoring = resume and os.path.exists(manifest_path)
        if restoring:
            with open(manifest_path) as f:
                if json.load(f) != manifest:
                    raise ValueError("cache manifest differs or is legacy; use a new cache directory / --fresh")
        self._maps: dict[str, np.memmap] = {}
        for fname, shape in layout.shapes.items():
            path = os.path.join(root, fname)
            if restoring and (not os.path.exists(path) or os.path.getsize(path) != int(np.prod(shape)) * np.dtype(DTYPES[fname]).itemsize):
                raise ValueError(f"incomplete cache storage: {fname}")
            self._maps[fname] = np.memmap(
                path, dtype=DTYPES[fname], mode="r+" if restoring else "w+", shape=shape)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        self._done_path = os.path.join(root, "done.json")
        self.done: set[str] = set()
        if restoring and os.path.exists(self._done_path):
            self.done = set(json.load(open(self._done_path)))
        self._labels: dict[str, dict] = {}
        if self.done:
            records = pd.read_parquet(os.path.join(root, "index.parquet"))
            self._labels = records.set_index("StudyInstanceUID").to_dict("index")

    def row(self, study_uid: str) -> int:
        return self._uid_to_row[study_uid]

    def write_study(self, study_uid: str, features: np.ndarray,
                    slice_mask: np.ndarray, series_valid: np.ndarray,
                    slot_valid: np.ndarray, positions: np.ndarray,
                    z_norm: np.ndarray, meta_codes: np.ndarray,
                    labels: np.ndarray | None, confidence: float = 1.0,
                    laterality_consensus: str = "",
                    is_gold: bool = False, quality: np.ndarray | None = None) -> None:
        i = self.row(study_uid)
        S, K = self.layout.n_slots, self.layout.k_series
        assert features.shape == (S, K, self.layout.slices_per_slot,
                                  self.layout.feat_dim)
        self._maps["features.f16"][i] = features.astype(np.float16)
        self._maps["slice_mask.u1"][i] = slice_mask.astype(np.uint8)
        self._maps["series_valid.u1"][i] = series_valid.astype(np.uint8)
        self._maps["slot_valid.u1"][i] = slot_valid.astype(np.uint8)
        self._maps["positions.i32"][i] = positions.astype(np.int32)
        self._maps["z_norm.f16"][i] = z_norm.astype(np.float16)
        self._maps["meta_codes.u1"][i] = meta_codes.astype(np.uint8)
        if quality is None:
            quality = np.zeros((S,K,8),np.float32)
            quality[:,:,0] = 1 / (1 + np.arange(K)[None,:])
        self._maps["quality.f32"][i] = quality
        rec = {"confidence": confidence, "is_gold": int(is_gold),
               "laterality_consensus": laterality_consensus}
        if labels is not None:
            rec.update({c: float(v) for c, v in zip(LABEL_COLS, labels)})
        self._labels[study_uid] = rec
        self.done.add(study_uid)

    def flush(self) -> None:
        for m in self._maps.values():
            m.flush()
        df = pd.DataFrame([
            {**self._labels.get(uid, {}), "StudyInstanceUID": uid, "storage_row": i}
            for i, uid in enumerate(self.study_uids)])
        path = os.path.join(self.root, "index.parquet")
        df.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        with open(self._done_path + ".tmp", "w") as f:
            json.dump(sorted(self.done), f)
        os.replace(self._done_path + ".tmp", self._done_path)

    def close(self) -> None:
        self.flush()
        for m in self._maps.values():
            del m
        self._maps.clear()


class FeatureCacheReader:
    """Random-access reader; memmaps are lazily paged by the OS."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "done.json")) as f:
            done = sorted(json.load(f))
        self.index = pd.read_parquet(os.path.join(root, "index.parquet"))
        # authoritative order = index parquet order; keep only done rows
        self.index = self.index[self.index["StudyInstanceUID"].isin(done)] \
            .reset_index(drop=True)
        self.study_uids = self.index["StudyInstanceUID"].tolist()
        self.n = len(self.study_uids)
        if self.n == 0:
            raise FileNotFoundError(f"empty cache under {root}")
        # the writer created shaped memmaps; total size / known layout gives
        # the per-study strides. Layout is read from a sidecar json written
        # by the writer for exactly this purpose.
        with open(os.path.join(root, "layout.json")) as f:
            lay = json.load(f)
        if lay.get("schema_version") != 2:
            raise ValueError("legacy feature cache: rebuild with corrected normalization and slot manifest")
        self.manifest = lay
        self.slot_ids = lay["slot_ids"]
        self._storage_rows = self.index["storage_row"].to_numpy(int)
        self._uid_to_row = {u: i for i, u in enumerate(self.study_uids)}
        self.n_slots = lay["n_slots"]
        self.k_series = lay["k_series"]
        self.slices_per_slot = lay["slices_per_slot"]
        self.feat_dim = lay["feat_dim"]
        self.total_declared = lay["n_studies"]
        self._maps = {
            fname: np.memmap(os.path.join(root, fname), dtype=dt, mode="r")
            for fname, dt in DTYPES.items()
        }
        self._shapes = {
            fname: tuple(lay["shapes"][fname])
            for fname in DTYPES
        }

    def _row(self, fname: str, i: int) -> np.ndarray:
        """Row view of a shaped memmap (the writer wrote shaped memmaps, so
        direct [i] indexing works — but they were opened flat; reshape)."""
        shape = self._shapes[fname]
        flat = self._maps[fname]
        per = int(np.prod(shape[1:]))
        storage_row = self._storage_rows[i]
        return flat[storage_row * per:(storage_row + 1) * per].reshape(shape[1:])

    def __len__(self) -> int:
        return self.n

    def uid_to_row(self, uid: str) -> int:
        return self._uid_to_row[uid]

    def get(self, idx: int) -> dict:
        i = idx
        row_uid = self.study_uids[i]
        return {
            "study_uid": row_uid,
            "features": np.array(self._row("features.f16", i), dtype=np.float32),
            "slice_mask": np.array(self._row("slice_mask.u1", i), dtype=bool),
            "series_valid": np.array(self._row("series_valid.u1", i), dtype=bool),
            "slot_valid": np.array(self._row("slot_valid.u1", i), dtype=bool),
            "positions": np.array(self._row("positions.i32", i)),
            "z_norm": np.array(self._row("z_norm.f16", i), dtype=np.float32),
            "meta_codes": np.array(self._row("meta_codes.u1", i)),
            "quality": np.array(self._row("quality.f32", i), dtype=np.float32),
            "labels": self.index.loc[i, LABEL_COLS].to_numpy(np.float32)
            if all(c in self.index.columns for c in LABEL_COLS)
            else np.full(len(LABEL_COLS), np.nan, np.float32),
            "label_mask": self.index.loc[i, LABEL_COLS].notna().to_numpy(
                np.float32)
            if all(c in self.index.columns for c in LABEL_COLS)
            else np.zeros(len(LABEL_COLS), np.float32),
            "is_gold": bool(self.index.loc[i, "is_gold"])
            if "is_gold" in self.index.columns else False,
            "confidence": float(self.index.loc[i, "confidence"])
            if "confidence" in self.index.columns else 1.0,
        }
