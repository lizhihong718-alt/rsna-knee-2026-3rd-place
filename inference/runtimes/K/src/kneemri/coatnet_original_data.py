"""CoAtNet study bags whose RGB channels are consecutive source MRI slices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from kneemri.coatnet_data import normalization_tensors, select_windows


ARRAYS = {
    "series_index": np.int16,
    "group_id": np.int8,
    "plane_id": np.int8,
    "position_mm": np.float32,
    "position_reliable": np.uint8,
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _budget(value, allow_list=False):
    values = value if allow_list and isinstance(value, list) else [value]
    if not values or not all(v == "all" or isinstance(v, int) and not isinstance(v, bool) and v > 0
                             for v in values):
        raise ValueError("window budget must contain positive integers or 'all'")
    return value


class CoAtNetOriginalNeighborBags(Dataset):
    """Read fixed candidate windows built from true source-slice neighbours.

    Candidate centres are fixed by the corpus. Training draws a fresh subset
    without replacement, while evaluation is deterministic and can use all
    candidates. Gold labels remain supplied by the separately audited cohort.
    """

    def __init__(self, cfg, all_ids, ids, labels, train):
        self.cfg = cfg
        self.ids = list(ids)
        self.labels = labels
        self.train = bool(train)
        self.root = Path(cfg["corpus"])
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        pre = self.manifest.get("preprocessing", {})
        if (pre.get("schema") != "coatnet-original-neighbors-v1"
                or pre.get("window_semantics") != "original_consecutive_source_slices"):
            raise ValueError("original-neighbour dataset requires its explicit corpus schema")
        if self.manifest.get("study_uids") != list(all_ids):
            raise ValueError("corpus row order differs from the audited cohort")
        self.rows = {uid: row for row, uid in enumerate(all_ids)}
        if not set(self.ids) <= set(self.rows) or not set(self.ids) <= set(labels):
            raise ValueError("dataset studies or labels are incomplete")
        shape = tuple(self.manifest.get("shape", []))
        if len(shape) != 5 or shape[0] != len(all_ids) or shape[2] != 3:
            raise ValueError("windows must have shape [study, candidate, 3, H, W]")
        self.source_resolution = int(shape[-1])
        self.storage_dtype = pre.get('storage_dtype', 'uint8')
        if self.storage_dtype not in {'uint8', 'float16'}:
            raise ValueError('unsupported original-neighbour storage dtype')
        if self.storage_dtype == 'float16' and pre.get('storage_value_range') != [0.0, 1.0]:
            raise ValueError('float16 corpus must explicitly store unit-range values')
        if shape[-2] != shape[-1]:
            raise ValueError("cached windows must be square")
        self.resolution = int(cfg["resolution"])
        if self.resolution > self.source_resolution:
            raise ValueError("training resolution cannot exceed cached source resolution")
        self.budget = _budget(cfg["train_windows" if train else "eval_windows"], allow_list=train)
        self.strategy = str(cfg.get("train_sampling" if train else "eval_sampling",
                                    "random" if train else "uniform"))
        if self.strategy not in {"random", "uniform", "group_balanced", "stratified"}:
            raise ValueError("unsupported window sampling strategy")
        if not train and self.strategy == "random":
            raise ValueError("evaluation sampling must be deterministic")
        self.gain = float(cfg.get("intensity_gain", 0.0))
        self.rotation = float(cfg.get("rotation_deg", 0.0))
        self.scale_jitter = float(cfg.get("scale_jitter", 0.0))
        self.gamma_jitter = float(cfg.get("gamma_jitter", 0.0))
        self.mean, self.std = normalization_tensors(cfg.get("normalization"))
        self.manifest_sha256 = _sha256(manifest_path)
        self.windows = self.masks = None
        self.metadata = {}

    def _open(self):
        if self.windows is not None:
            return
        self.windows = np.load(self.root / "windows.npy", mmap_mode="r")
        self.masks = np.load(self.root / "masks.npy", mmap_mode="r")
        self.metadata = {name: np.load(self.root / f"{name}.npy", mmap_mode="r")
                         for name in ARRAYS}
        expected = self.windows.shape[:2]
        if (self.windows.shape != tuple(self.manifest["shape"])
                or self.windows.dtype != np.dtype(self.storage_dtype)):
            raise ValueError("cached window array differs from manifest")
        if self.masks.shape != expected or self.masks.dtype != np.uint8:
            raise ValueError("cached window mask differs from manifest")
        for name, dtype in ARRAYS.items():
            if self.metadata[name].shape != expected or self.metadata[name].dtype != dtype:
                raise ValueError(f"cached {name} differs from manifest")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["windows"] = state["masks"] = None
        state["metadata"] = {}
        return state

    def __len__(self):
        return len(self.ids)

    def _selected(self, row):
        valid = np.flatnonzero(np.asarray(self.masks[row], dtype=bool))
        candidates = np.repeat(valid[:, None], 3, axis=1)
        budget = random.choice(self.budget) if isinstance(self.budget, list) else self.budget
        groups = np.asarray(self.metadata["series_index"][row])[valid]
        rng = np.random.default_rng(random.getrandbits(64)) if self.train else None
        selected = select_windows(candidates, budget, strategy=self.strategy, rng=rng,
                                  group_ids=groups if self.strategy in {"group_balanced", "stratified"} else None)
        return selected[:, 0], len(valid), budget

    def _augment(self, images):
        unit_range = images.dtype == np.float16
        if unit_range:
            # OpenCV does not accept float16; retain its precision in float32.
            images = images.astype(np.float32)
            if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError('float16 cache contains invalid unit-range values')
        if not self.train:
            return images.astype(np.float32) if unit_range else images.astype(np.float32) / 255.0
        if self.rotation or self.scale_jitter:
            import cv2
            cv2.setNumThreads(1)
            h = images.shape[-1]
            angle = random.uniform(-self.rotation, self.rotation)
            scale = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
            affine = cv2.getRotationMatrix2D(((h - 1) / 2, (h - 1) / 2), angle, scale)
            flat = images.reshape(-1, h, h)
            images = np.stack([cv2.warpAffine(frame, affine, (h, h), flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT) for frame in flat])
            images = images.reshape(-1, 3, h, h)
        values = images.astype(np.float32) if unit_range else images.astype(np.float32) / 255.0
        if self.gamma_jitter:
            values = values ** random.uniform(1 - self.gamma_jitter, 1 + self.gamma_jitter)
        if self.gain:
            values *= 1 + random.uniform(-self.gain, self.gain)
        return np.clip(values, 0, 1)

    def __getitem__(self, index):
        self._open()
        uid = self.ids[index]
        row = self.rows[uid]
        selected, candidate_count, budget = self._selected(row)
        if len(selected):
            raw = np.asarray(self.windows[row, selected])
            x = torch.from_numpy(self._augment(raw))
            if x.shape[-1] != self.resolution:
                x = torch.nn.functional.interpolate(x, size=(self.resolution, self.resolution),
                                                     mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            valid = torch.ones(len(selected), dtype=torch.bool)
            groups = np.asarray(self.metadata["group_id"][row, selected], dtype=np.int64)
            series = np.asarray(self.metadata["series_index"][row, selected], dtype=np.int64)
            planes = np.asarray(self.metadata["plane_id"][row, selected], dtype=np.int64)
            positions = np.asarray(self.metadata["position_mm"][row, selected], dtype=np.float32)[:, None]
            position_mask = np.asarray(self.metadata["position_reliable"][row, selected], dtype=bool)
        else:
            x = torch.zeros((1, 3, self.resolution, self.resolution), dtype=torch.float32)
            valid = torch.zeros(1, dtype=torch.bool)
            groups = series = planes = np.full(1, -1, np.int64)
            positions = np.zeros((1, 1), np.float32)
            position_mask = np.zeros(1, bool)
        label = np.asarray(self.labels[uid], dtype=np.float32)
        if label.shape != (12,):
            raise ValueError(f"expected the explicit 12-label contract for {uid}")
        trace = {
            "study_id": uid,
            "window_semantics": "original_consecutive_source_slices",
            "sampling": self.strategy,
            "requested_budget": budget,
            "candidate_count": candidate_count,
            "selected_count": int(len(selected)),
            "selected_candidate_indices": selected.tolist(),
            "empty_bag": not bool(len(selected)),
            "corpus_manifest_sha256": self.manifest_sha256,
        }
        return {
            "images": x,
            "targets": torch.from_numpy(label.copy()),
            "window_mask": valid,
            "group_ids": torch.from_numpy(groups.copy()),
            "series_ids": torch.from_numpy(series.copy()),
            "plane_ids": torch.from_numpy(planes.copy()),
            "positions": torch.from_numpy(positions.copy()),
            "position_mask": torch.from_numpy(position_mask.copy()),
            "study_id": uid,
            "label_source": (self.cfg.get("train_label_source", "report_weak") if self.train
                             else self.cfg.get("eval_label_source", "official_gold")),
            "provenance": trace,
        }
