"""Exact bag construction over the released two-part 44-slice Raptor corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from kneemri.coatnet_data import normalization_tensors


SLOT_OFFSETS = np.asarray([0, 12, 22, 30, 36, 44], dtype=np.int64)
PLANE_BY_SLOT = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _budget(value, allow_list=False):
    values = value if allow_list and isinstance(value, list) else [value]
    if not values or not all(v == "all" or isinstance(v, int) and not isinstance(v, bool) and v > 0
                             for v in values):
        raise ValueError("window budget must contain positive integers or 'all'")
    return value


class CoAtNetPublishedRaptorBags(Dataset):
    """Reproduce the public trainer's concatenated-stack window semantics."""

    def __init__(self, cfg, all_ids, ids, labels, train):
        self.cfg = cfg
        self.ids = list(ids)
        self.labels = labels
        self.train = bool(train)
        self.root = Path(cfg["corpus"])
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        pre = self.manifest.get("preprocessing", {})
        if (pre.get("schema") != "coatnet-published-raptor44-v1"
                or pre.get("window_semantics") != "released_concatenated_stack_neighbors"
                or pre.get("slot_offsets") != SLOT_OFFSETS.tolist()):
            raise ValueError("released Raptor dataset requires its explicit corpus schema")
        if self.manifest.get("study_uids") != list(all_ids):
            raise ValueError("corpus row order differs from the audited cohort")
        self.rows = {uid: row for row, uid in enumerate(all_ids)}
        if not set(self.ids) <= set(self.rows) or not set(self.ids) <= set(labels):
            raise ValueError("dataset studies or labels are incomplete")
        shape = tuple(self.manifest.get("shape", []))
        parts = self.manifest.get("parts", [])
        self.part_shapes = [tuple(part.get("shape", [])) for part in parts]
        if (len(shape) != 4 or shape[0] != len(all_ids) or shape[1:] != (44, 336, 336)
                or len(parts) != 2 or any(part[1:] != shape[1:] for part in self.part_shapes)
                or sum(part[0] for part in self.part_shapes) != shape[0]):
            raise ValueError("released Raptor volumes must combine to [study,44,336,336]")
        self.part_paths = [(self.root / part["volume_path"]).resolve() for part in parts]
        self.part_starts = np.cumsum([0] + [part[0] for part in self.part_shapes])
        self.source_resolution = 336
        self.resolution = int(cfg["resolution"])
        if self.resolution < 1:
            raise ValueError("training resolution must be positive")
        self.budget = _budget(cfg["train_windows" if train else "eval_windows"], allow_list=train)
        self.strategy = str(cfg.get("train_sampling" if train else "eval_sampling",
                                    "random" if train else "uniform"))
        if self.strategy not in {"random", "uniform"}:
            raise ValueError("the released trainer supports random training and uniform evaluation only")
        if self.train and self.strategy != "random":
            raise ValueError("released Raptor training requires random window sampling")
        if not self.train and self.strategy != "uniform":
            raise ValueError("released Raptor evaluation requires uniform window sampling")
        self.gain = float(cfg.get("intensity_gain", 0.0))
        self.rotation = float(cfg.get("rotation_deg", 0.0))
        self.scale_jitter = float(cfg.get("scale_jitter", 0.0))
        self.gamma_jitter = float(cfg.get("gamma_jitter", 0.0))
        self.mean, self.std = normalization_tensors(cfg.get("normalization"))
        self.manifest_sha256 = _sha256(manifest_path)
        self.volumes = None
        self.masks = None

    def _open(self):
        if self.volumes is not None:
            return
        self.volumes = [np.load(path, mmap_mode="r") for path in self.part_paths]
        self.masks = np.load(self.root / "masks.npy", mmap_mode="r")
        if any(volume.shape != shape or volume.dtype != np.uint8
               for volume, shape in zip(self.volumes, self.part_shapes)):
            raise ValueError("released Raptor volume part differs from manifest")
        if self.masks.shape != (len(self.rows), 44) or self.masks.dtype != np.uint8:
            raise ValueError("released Raptor masks differ from manifest")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["volumes"] = state["masks"] = None
        return state

    def __len__(self):
        return len(self.ids)

    def _volume(self, row):
        part = int(np.searchsorted(self.part_starts[1:], row, side="right"))
        return self.volumes[part][row - self.part_starts[part]]

    @staticmethod
    def _centres(mask):
        valid = np.flatnonzero(mask)
        if len(valid) < 3:
            valid = np.arange(min(3, len(mask)))
        lo, hi = int(valid.min()), int(valid.max())
        centres = np.arange(lo + 1, hi, dtype=np.int64)
        if not len(centres):
            centres = np.asarray([max(1, min((lo + hi) // 2, len(mask) - 2))], dtype=np.int64)
        return centres

    def _selected(self, row):
        centres = self._centres(np.asarray(self.masks[row], dtype=bool))
        budget = random.choice(self.budget) if isinstance(self.budget, list) else self.budget
        if budget == "all":
            return centres, len(centres), budget
        if self.train:
            pool = centres.tolist() * (budget // len(centres) + 1)
            random.shuffle(pool)
            selected = np.asarray(pool[:budget], dtype=np.int64)
        else:
            indices = np.linspace(0, len(centres) - 1, budget).round().astype(np.int64)
            selected = centres[indices]
        return selected, len(centres), budget

    def _augment_geometry(self, images):
        if not self.train:
            return images
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
        return images

    def _augment_intensity(self, values):
        if not self.train:
            return values
        if self.gamma_jitter:
            values = values ** random.uniform(1 - self.gamma_jitter, 1 + self.gamma_jitter)
        if self.gain:
            values *= 1 + random.uniform(-self.gain, self.gain)
        return values.clamp(0, 1)

    def __getitem__(self, index):
        self._open()
        uid = self.ids[index]
        row = self.rows[uid]
        centres, candidate_count, budget = self._selected(row)
        volume = self._volume(row)
        triplets = centres[:, None] + np.asarray([-1, 0, 1], dtype=np.int64)
        raw = self._augment_geometry(np.asarray(volume[triplets]))
        x = torch.from_numpy(raw.astype(np.float32) / 255.0)
        if x.shape[-1] != self.resolution:
            x = torch.nn.functional.interpolate(x, size=(self.resolution, self.resolution),
                                                 mode="bilinear", align_corners=False)
        # The released trainer resizes first, then applies one bag-consistent
        # intensity draw. The order differs at pixels clipped by a positive gain.
        x = self._augment_intensity(x)
        x = (x - self.mean) / self.std
        valid = torch.ones(len(centres), dtype=torch.bool)
        groups = np.searchsorted(SLOT_OFFSETS[1:], centres, side="right").astype(np.int64)
        planes = PLANE_BY_SLOT[groups]
        positions = np.zeros((len(centres), 1), dtype=np.float32)
        position_mask = np.zeros(len(centres), dtype=bool)
        label = np.asarray(self.labels[uid], dtype=np.float32)
        if label.shape != (12,):
            raise ValueError(f"expected the explicit 12-label contract for {uid}")
        trace = {
            "study_id": uid,
            "window_semantics": "released_concatenated_stack_neighbors",
            "sampling": self.strategy,
            "requested_budget": budget,
            "candidate_count": candidate_count,
            "selected_count": int(len(centres)),
            "selected_centres": centres.tolist(),
            "cached_slice_indices": triplets.tolist(),
            "cross_slot_windows": int(sum(len(set(groups_for_window)) > 1 for groups_for_window in
                                          np.searchsorted(SLOT_OFFSETS[1:], triplets, side="right"))),
            "empty_bag": False,
            "corpus_manifest_sha256": self.manifest_sha256,
        }
        return {
            "images": x,
            "targets": torch.from_numpy(label.copy()),
            "window_mask": valid,
            "group_ids": torch.from_numpy(groups.copy()),
            "series_ids": torch.from_numpy(groups.copy()),
            "plane_ids": torch.from_numpy(planes.copy()),
            "positions": torch.from_numpy(positions),
            "position_mask": torch.from_numpy(position_mask),
            "study_id": uid,
            "label_source": (self.cfg.get("train_label_source", "report_weak") if self.train
                             else self.cfg.get("eval_label_source", "official_gold")),
            "provenance": trace,
        }
