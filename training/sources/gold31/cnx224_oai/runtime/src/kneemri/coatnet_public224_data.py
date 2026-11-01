"""CoAtNet bags over the public six-slot 224px physical-crop corpus."""
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
    "valid_slice_mask": np.uint8,
    "triplet_valid_mask": np.uint8,
    "source_index": np.int32,
    "physical_position": np.float32,
    "stack_id_per_slice": np.int16,
    "order_safe": np.uint8,
}
PLANE_BY_SLOT = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _budget(value, allow_list=False):
    values = value if allow_list and isinstance(value, list) else [value]
    if not values or not all(v == "all" or isinstance(v, int) and not isinstance(v, bool) and v > 0
                             for v in values):
        raise ValueError("window budget must contain positive integers or 'all'")
    return value


class CoAtNetPublic224Bags(Dataset):
    """Read valid physical-neighbour triplets from the public 96-slice cache."""

    def __init__(self, cfg, all_ids, ids, labels, train):
        self.cfg = cfg
        self.ids = list(ids)
        self.labels = labels
        self.train = bool(train)
        self.root = Path(cfg["corpus"])
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        pre = self.manifest.get("preprocessing", {})
        if (pre.get("schema") != "coatnet-public224-triplets-v1"
                or pre.get("window_semantics") != "public_gap_gated_physical_triplets"
                or pre.get("source_dataset_version") != "RSNA_KNEE_RESNET224_V1"):
            raise ValueError("public224 dataset requires its explicit corpus schema")
        if self.manifest.get("study_uids") != list(all_ids):
            raise ValueError("corpus row order differs from the audited cohort")
        self.rows = {uid: row for row, uid in enumerate(all_ids)}
        if not set(self.ids) <= set(self.rows) or not set(self.ids) <= set(labels):
            raise ValueError("dataset studies or labels are incomplete")
        shape = tuple(self.manifest.get("volume_shape", []))
        if len(shape) != 4 or shape[0] != len(all_ids) or shape[1] != 96 or shape[-2:] != (224, 224):
            raise ValueError("public224 volumes must have shape [study,96,224,224]")
        self.source_resolution = shape[-1]
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
        self.volumes = None
        self.arrays = {}
        self.triplets = self.slot_ids = None

    def _open(self):
        if self.volumes is not None:
            return
        self.volumes = np.load(self.root / "volumes.npy", mmap_mode="r")
        self.arrays = {name: np.load(self.root / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
        self.triplets = np.load(self.root / "triplet_indices.npy")
        self.slot_ids = np.load(self.root / "slot_id.npy")
        shape = tuple(self.manifest["volume_shape"])
        if self.volumes.shape != shape or self.volumes.dtype != np.uint8:
            raise ValueError("cached public224 volume differs from manifest")
        row_shape = shape[:2]
        triplet_shape = (shape[0], len(self.triplets))
        for name, dtype in ARRAYS.items():
            expected = triplet_shape if name == "triplet_valid_mask" else row_shape
            if self.arrays[name].shape != expected or self.arrays[name].dtype != dtype:
                raise ValueError(f"cached {name} differs from manifest")
        if (self.triplets.shape != (84, 3) or self.triplets.dtype != np.int32
                or self.slot_ids.shape != (96,) or self.slot_ids.dtype != np.int16):
            raise ValueError("public224 static triplet contract differs")
        if (self.triplets < 0).any() or (self.triplets >= 96).any():
            raise ValueError("public224 triplet index is out of bounds")
        triplet_slots = self.slot_ids[self.triplets]
        if not (triplet_slots == triplet_slots[:, :1]).all():
            raise ValueError("public224 triplet crosses slots")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["volumes"] = state["triplets"] = state["slot_ids"] = None
        state["arrays"] = {}
        return state

    def __len__(self):
        return len(self.ids)

    def _selected(self, row):
        valid = np.flatnonzero(np.asarray(self.arrays["triplet_valid_mask"][row], dtype=bool))
        candidates = np.repeat(valid[:, None], 3, axis=1)
        budget = random.choice(self.budget) if isinstance(self.budget, list) else self.budget
        groups = self.slot_ids[self.triplets[valid, 1]]
        rng = np.random.default_rng(random.getrandbits(64)) if self.train else None
        selected = select_windows(candidates, budget, strategy=self.strategy, rng=rng,
                                  group_ids=groups if self.strategy in {"group_balanced", "stratified"} else None)
        return selected[:, 0], len(valid), budget

    def _augment(self, images):
        if not self.train:
            return images.astype(np.float32) / 255.0
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
        values = images.astype(np.float32) / 255.0
        if self.gamma_jitter:
            values = values ** random.uniform(1 - self.gamma_jitter, 1 + self.gamma_jitter)
        if self.gain:
            values *= 1 + random.uniform(-self.gain, self.gain)
        return np.clip(values, 0, 1)

    def _metadata(self, row, triplets):
        centres = triplets[:, 1]
        groups = np.asarray(self.slot_ids[centres], dtype=np.int64)
        series = np.asarray(self.arrays["stack_id_per_slice"][row, centres], dtype=np.int64)
        planes = PLANE_BY_SLOT[groups]
        physical = np.asarray(self.arrays["physical_position"][row], dtype=np.float32)
        safe = np.asarray(self.arrays["order_safe"][row], dtype=bool)
        valid_slices = np.asarray(self.arrays["valid_slice_mask"][row], dtype=bool)
        known = np.isfinite(physical[triplets]).all(1) & safe[triplets].all(1)
        same_stack = (np.asarray(self.arrays["stack_id_per_slice"][row, triplets])
                      == series[:, None]).all(1) & (series >= 0)
        known &= same_stack
        positions = np.zeros((len(triplets), 1), dtype=np.float32)
        for identity in np.unique(series[known]):
            support = valid_slices & safe & np.isfinite(physical)
            support &= np.asarray(self.arrays["stack_id_per_slice"][row]) == identity
            selected = known & (series == identity)
            if support.any():
                midpoint = (float(physical[support].min()) + float(physical[support].max())) / 2
                positions[selected, 0] = physical[centres[selected]] - midpoint
            else:
                known[selected] = False
        return groups, series, planes, positions, known

    def __getitem__(self, index):
        self._open()
        uid = self.ids[index]
        row = self.rows[uid]
        selected, candidate_count, budget = self._selected(row)
        triplets = self.triplets[selected]
        if len(triplets):
            raw = np.asarray(self.volumes[row, triplets])
            x = torch.from_numpy(self._augment(raw))
            if x.shape[-1] != self.resolution:
                x = torch.nn.functional.interpolate(x, size=(self.resolution, self.resolution),
                                                     mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            valid = torch.ones(len(triplets), dtype=torch.bool)
            groups, series, planes, positions, position_mask = self._metadata(row, triplets)
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
            "window_semantics": "public_gap_gated_physical_triplets",
            "sampling": self.strategy,
            "requested_budget": budget,
            "candidate_count": candidate_count,
            "selected_count": int(len(selected)),
            "selected_candidate_indices": selected.tolist(),
            "cached_slice_indices": triplets.tolist(),
            "source_slice_indices": np.asarray(self.arrays["source_index"][row, triplets]).tolist(),
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
