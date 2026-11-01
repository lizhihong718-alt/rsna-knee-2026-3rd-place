"""Training bags over the unchanged Raptor64 corpus with explicit provenance.

The legacy compatibility switch preserves historical repeated sampling and input
normalization. Other selections cap the budget at available candidate windows.
Safe cached neighbours are not advertised as original adjacent DICOM slices.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from kneemri import raptor_reference as raptor
from kneemri.coatnet_data import candidate_windows, normalization_tensors, select_windows


ACTUAL_GROUPS = {"Sagittal_fluid": 0, "Sagittal_nonfluid": 1,
                 "Coronal_fluid": 2, "Coronal_nonfluid": 3, "Axial": 4}
PROVENANCE_ARRAYS = ("slot_index", "series_index", "original_index",
                     "physical_position_mm", "geometry_reliable", "actual_plane", "fluid")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _budget(value, allow_list=False):
    if allow_list and isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("variable window budgets cannot be empty")
        return [_budget(item) for item in value]
    if value == "all":
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("window budget must be a positive integer or 'all'")
    return int(value)


def actual_group_ids(planes, fluids):
    """Five attribute codes from actual acquisition properties, never slot order."""
    planes, fluids = np.asarray(planes), np.asarray(fluids)
    if planes.shape != fluids.shape:
        raise ValueError("actual plane and fluid arrays must have equal shape")
    groups = np.full(planes.shape, -1, dtype=np.int64)
    for plane, fluid, group in [(0, 1, 0), (0, 0, 1), (1, 1, 2), (1, 0, 3)]:
        groups[(planes == plane) & (fluids == fluid)] = group
    groups[planes == 2] = 4
    return groups


def window_metadata(windows, mask, metadata):
    """Return actual identities and known, series-relative physical positions."""
    windows = np.asarray(windows, dtype=np.int64).reshape(-1, 3)
    mask = np.asarray(mask, dtype=bool)
    series = np.asarray(metadata["series_index"])
    plane = np.asarray(metadata["actual_plane"])
    fluid = np.asarray(metadata["fluid"])
    physical = np.asarray(metadata["physical_position_mm"])
    reliable = np.asarray(metadata["geometry_reliable"], bool) & np.isfinite(physical)
    groups = actual_group_ids(plane, fluid)
    identities = series[windows]
    same = (identities >= 0).all(1) & (identities == identities[:, :1]).all(1)
    same &= mask[windows].all(1)
    series_ids = np.where(same, identities[:, 1], -1).astype(np.int64)
    homogeneous = (groups[windows] >= 0).all(1) & (groups[windows] == groups[windows[:, :1]]).all(1)
    group_ids = np.where(same & homogeneous, groups[windows[:, 1]], -1).astype(np.int64)
    # Coarser than group: one pool per actual acquisition plane (fluid ignored).
    same_plane = (plane[windows] >= 0).all(1) & (plane[windows] == plane[windows[:, :1]]).all(1)
    plane_ids = np.where(same & same_plane, plane[windows[:, 1]], -1).astype(np.int64)
    known = same & reliable[windows].all(1)
    positions = np.zeros((len(windows), 1), dtype=np.float32)
    for identity in np.unique(series_ids[known]):
        all_positions = physical[(series == identity) & reliable & mask]
        midpoint = (float(all_positions.min()) + float(all_positions.max())) / 2
        selected = known & (series_ids == identity)
        positions[selected, 0] = physical[windows[selected, 1]] - midpoint
    return {"group_ids": group_ids, "series_ids": series_ids, "plane_ids": plane_ids,
            "positions": positions, "position_mask": known}


class CoAtNetStudyBags(Dataset):
    """Compatible constructor to StudyBags, yielding dictionaries for ragged bags.

    ``window_mode`` is legacy or same_series. ``train_sampling`` supports random,
    uniform, group_balanced and stratified; ``eval_sampling`` defaults to uniform.
    ``legacy_sampling`` explicitly enables the original duplicate-capable sampler
    and requires legacy windows, no deduplication, and random/uniform selection.
    ``train_windows`` accepts an integer, 'all', or a list of those budgets.
    ``eval_windows`` accepts an integer or 'all'. The dataset never changes labels.
    """
    def __init__(self, cfg, all_ids, ids, labels, train):
        self.cfg = dict(cfg)
        self.ids = list(ids)
        self.all_ids = list(all_ids)
        self.labels = labels
        self.train = bool(train)
        self.rows = {uid: i for i, uid in enumerate(self.all_ids)}
        if len(self.rows) != len(self.all_ids) or not set(self.ids) <= set(self.rows):
            raise ValueError("duplicate corpus IDs or selected studies missing from corpus")
        self.root = Path(cfg["corpus"])
        self.provenance_root = Path(cfg.get("provenance", self.root.with_name(self.root.name + "_provenance")))
        self.mode = str(cfg.get("window_mode", "legacy"))
        self.strategy = str(cfg.get("train_sampling" if train else "eval_sampling", "random" if train else "uniform"))
        self.deduplicate = bool(cfg.get("deduplicate_windows", False))
        self.legacy_sampling = bool(cfg.get("legacy_sampling", False))
        self.resolution = int(cfg["resolution"])
        self.gain = float(cfg.get("intensity_gain", 0.0))
        self.budget = _budget(cfg["train_windows" if train else "eval_windows"], allow_list=train)
        self.include_provenance = bool(cfg.get("include_provenance", not train))
        if self.mode not in {"legacy", "no_cross_series", "same_series"}:
            raise ValueError("window_mode must be legacy, no_cross_series or same_series")
        if self.strategy not in {"random", "uniform", "group_balanced", "stratified"}:
            raise ValueError("unsupported window sampling strategy")
        if not train and self.strategy == "random":
            raise ValueError("random validation sampling is not deterministic")
        if self.legacy_sampling and (self.mode != "legacy" or self.deduplicate
                                     or self.strategy != ("random" if train else "uniform")):
            raise ValueError("legacy_sampling requires legacy windows without deduplication and default sampling")
        self.mean, self.std = normalization_tensors(cfg.get("normalization"))
        if self.resolution < 1 or not np.isfinite(self.gain) or not 0 <= self.gain <= 1:
            raise ValueError("invalid input resolution or intensity gain")
        self.volumes = self.masks = None
        self.metadata = None
        self._study_reports = {}
        self._validate_provenance()

    def _validate_provenance(self):
        source = json.loads((self.root / "manifest.json").read_text())
        manifest = json.loads((self.provenance_root / "manifest.json").read_text())
        if (manifest.get("schema") != "raptor64-provenance-v1"
                or manifest.get("study_uids") != self.all_ids
                or source.get("study_uids") != self.all_ids
                or not manifest.get("complete_corpus")
                or manifest.get("source_manifest_sha256") != _sha256(self.root / "manifest.json")
                or manifest.get("source_masks_sha256") != _sha256(self.root / "masks.npy")):
            raise ValueError("provenance is incomplete, reordered, or not bound to this image cache")
        complete = np.load(self.provenance_root / "complete.npy", mmap_mode="r")
        if complete.shape != (len(self.all_ids),) or not complete.all():
            raise ValueError("provenance reconstruction is not complete")
        shape = np.load(self.root / "masks.npy", mmap_mode="r").shape
        if list(shape) != manifest.get("shape"):
            raise ValueError("provenance shape differs from image cache")
        for name in PROVENANCE_ARRAYS:
            if np.load(self.provenance_root / f"{name}.npy", mmap_mode="r").shape != shape:
                raise ValueError(f"invalid provenance shape: {name}")
        self.provenance_manifest_sha256 = _sha256(self.provenance_root / "manifest.json")

    def _open(self):
        if self.volumes is None:
            self.volumes = np.load(self.root / "volumes.npy", mmap_mode="r")
            self.masks = np.load(self.root / "masks.npy", mmap_mode="r")
            self.metadata = {name: np.load(self.provenance_root / f"{name}.npy", mmap_mode="r")
                             for name in PROVENANCE_ARRAYS}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(volumes=None, masks=None, metadata=None, _study_reports={})
        return state

    def __len__(self):
        return len(self.ids)

    def _selection(self, mask, metadata):
        candidates = candidate_windows(mask, metadata["series_index"], metadata["original_index"],
                                       mode=self.mode, deduplicate=self.deduplicate)
        budget = random.choice(self.budget) if isinstance(self.budget, list) else self.budget
        if budget == "all" or not len(candidates):
            return candidates.copy(), len(candidates), budget
        if self.legacy_sampling:
            if self.train:
                # Preserve pool multiplicity AND Python RNG calls from the ep7 trainer.
                indices = list(range(len(candidates))) * (budget // len(candidates) + 1)
                random.shuffle(indices)
                selected = candidates[indices[:budget]]
            else:
                indices = np.linspace(0, len(candidates) - 1, budget).round().astype(np.int64)
                selected = candidates[indices]
        else:
            metadata_out = window_metadata(candidates, mask, metadata)
            rng = np.random.default_rng(random.getrandbits(64)) if self.train else None
            selected = select_windows(candidates, budget, strategy=self.strategy, rng=rng,
                                      group_ids=metadata_out["series_ids" if self.strategy == "stratified" else "group_ids"])
        return selected, len(candidates), budget

    def _report(self, uid, row):
        if uid not in self._study_reports:
            report = json.loads((self.provenance_root / "studies" / f"{uid}.json").read_text())
            if (report.get("study_uid") != uid or report.get("cache_row") != row
                    or report.get("cached_volume_row_sha256") != hashlib.sha256(np.asarray(self.volumes[row]).tobytes()).hexdigest()
                    or report.get("cached_mask_sha256") != hashlib.sha256(np.asarray(self.masks[row]).tobytes()).hexdigest()):
                raise ValueError(f"study provenance does not match cached pixels: {uid}")
            self._study_reports[uid] = report
        return self._study_reports[uid]

    def _trace(self, uid, row, windows, mask, metadata, window_meta, candidate_count, budget):
        trace = {"study_id": uid, "window_mode": self.mode, "sampling": self.strategy,
                 "legacy_sampling": self.legacy_sampling, "deduplicate_windows": self.deduplicate,
                 "requested_budget": budget, "candidate_count": candidate_count,
                 "selected_count": len(windows), "empty_bag": not bool(len(windows)) or not bool(np.asarray(mask).any()),
                 "valid_cached_slice_count": int(np.asarray(mask, bool).sum()),
                 "position_reference": "millimetres relative to reliable cached series extent midpoint",
                 "provenance_manifest_sha256": self.provenance_manifest_sha256, "windows": []}
        if not self.include_provenance:
            # These indices resolve through the immutable per-study source record
            # without loading and retaining its large header table in every worker.
            trace["source_record"] = str(self.provenance_root / "studies" / f"{uid}.json")
            trace["cached_indices"] = windows.tolist()
            trace["series_indices"] = np.asarray(metadata["series_index"])[windows].tolist()
            trace["original_indices"] = np.asarray(metadata["original_index"])[windows].tolist()
            return trace
        report = self._report(uid, row)
        slices = report["slices"]
        seen = set()
        unique_sources = set()
        for index, triple in enumerate(windows):
            source = [slices[int(i)] for i in triple]
            keys = tuple((item.get("series_uid"), item.get("original_index")) for item in source)
            identities = {key[0] for key in keys if key[0] is not None}
            unique_sources.update(key for key in keys if key[0] is not None and key[1] >= 0)
            record = {"window_index": index, "cached_indices": triple.tolist(),
                      "series_uids": [item.get("series_uid") for item in source],
                      "original_indices": [item.get("original_index") for item in source],
                      "source_files": [item.get("file") for item in source],
                      "physical_positions_mm": [item.get("physical_position_mm") for item in source],
                      "actual_planes": [item.get("actual_plane") for item in source],
                      "actual_fluid": [item.get("actual_fluid") for item in source],
                      "requested_slot_indices": [item.get("slot_index") for item in source],
                      "group_id": int(window_meta["group_ids"][index]),
                      "series_id": int(window_meta["series_ids"][index]),
                      "relative_position_mm": float(window_meta["positions"][index, 0]) if window_meta["position_mask"][index] else None,
                      "position_reliable": bool(window_meta["position_mask"][index]),
                      "cross_series": len(identities) > 1,
                      "contains_source_padding": any(item.get("source_padding", False) for item in source),
                      "contains_invalid_image": not bool(np.asarray(mask)[triple].all()),
                      "repeated_source_inside_window": len(set(keys)) < 3,
                      "repeated_window": keys in seen,
                      "original_adjacent": len(identities) == 1 and all(key[0] is not None for key in keys)
                                           and bool((np.diff([key[1] for key in keys]) == 1).all())}
            trace["windows"].append(record)
            seen.add(keys)
        trace["unique_source_slice_count"] = len(unique_sources)
        trace["unique_window_count"] = len(seen)
        return trace

    def __getitem__(self, index):
        self._open()
        uid = self.ids[index]
        row = self.rows[uid]
        volume, mask = self.volumes[row], self.masks[row]
        metadata = {name: array[row] for name, array in self.metadata.items()}
        windows, candidate_count, budget = self._selection(mask, metadata)
        window_meta = window_metadata(windows, mask, metadata)
        provenance = self._trace(uid, row, windows, mask, metadata, window_meta, candidate_count, budget)
        if len(windows):
            x = torch.from_numpy(np.asarray(volume[windows], dtype=np.float32) / 255.0)
            if x.shape[-2:] != (self.resolution, self.resolution):
                x = torch.nn.functional.interpolate(x, size=(self.resolution, self.resolution),
                                                    mode="bilinear", align_corners=False)
            if self.train:
                x = (x * (1 + random.uniform(-self.gain, self.gain))).clamp(0, 1)
            x = (x - self.mean) / self.std
            valid = torch.ones(len(windows), dtype=torch.bool)
            # Legacy's interior missing groups participated in attention. Preserve
            # that historical comparison; truly empty studies remain neutral.
            if not np.asarray(mask).any():
                valid.zero_()
        else:
            x = torch.zeros((1, 3, self.resolution, self.resolution), dtype=torch.float32)
            valid = torch.zeros(1, dtype=torch.bool)
            window_meta = {"group_ids": np.full(1, -1, np.int64),
                           "series_ids": np.full(1, -1, np.int64),
                           "plane_ids": np.full(1, -1, np.int64),
                           "positions": np.zeros((1, 1), np.float32),
                           "position_mask": np.zeros(1, bool)}
        labels = np.asarray(self.labels[uid], dtype=np.float32)
        if labels.shape != (12,):
            raise ValueError(f"expected the explicit 12-label contract for {uid}")
        return {"images": x, "targets": torch.from_numpy(labels.copy()), "window_mask": valid,
                **{name: torch.from_numpy(value.copy()) for name, value in window_meta.items()},
                "study_id": uid, "label_source": (self.cfg.get("train_label_source", "report_weak") if self.train
                                                   else self.cfg.get("eval_label_source", "official_gold")),
                "provenance": provenance}


def coatnet_collate(items):
    """Pad study bags to a stable K >= 1; padding never becomes valid evidence."""
    if not items:
        raise ValueError("cannot collate an empty study batch")
    max_windows = max(1, *(item["images"].shape[0] for item in items))
    # Position width: 1 for scalar series-relative mm, 3 for per-plane coordinates.
    pdim = items[0]["positions"].reshape(items[0]["images"].shape[0], -1).shape[1]
    shape = tuple(items[0]["images"].shape[1:])
    if any(tuple(item["images"].shape[1:]) != shape for item in items):
        raise ValueError("all studies in a batch require the same image shape")
    batch = {"images": items[0]["images"].new_zeros((len(items), max_windows, *shape)),
             "targets": torch.stack([item["targets"] for item in items]),
             "window_mask": torch.zeros((len(items), max_windows), dtype=torch.bool),
             "group_ids": torch.full((len(items), max_windows), -1, dtype=torch.long),
             "series_ids": torch.full((len(items), max_windows), -1, dtype=torch.long),
             "plane_ids": torch.full((len(items), max_windows), -1, dtype=torch.long),
             "positions": torch.zeros((len(items), max_windows, pdim), dtype=torch.float32),
             "position_mask": torch.zeros((len(items), max_windows), dtype=torch.bool),
             "study_ids": [item["study_id"] for item in items],
             "label_source": [item.get("label_source", "unspecified") for item in items],
             "provenance": [item["provenance"] for item in items]}
    for row, item in enumerate(items):
        k = item["images"].shape[0]
        for key in ("images", "window_mask", "group_ids", "series_ids", "plane_ids", "positions", "position_mask"):
            value = item[key].reshape(k, pdim) if key == "positions" else item[key]
            batch[key][row, :k] = value
    # Compact uint8 bags carry their per-study intensity tables; all or none.
    luts = [item.get("intensity_lut") for item in items]
    if any(lut is not None for lut in luts):
        if any(lut is None for lut in luts) or batch["images"].dtype != torch.uint8:
            raise ValueError("uint8 bags and intensity tables must be collated together")
        batch["intensity_lut"] = torch.stack(luts)
    if "budget_positions" in items[0]:
        batch["budget_positions"] = [item["budget_positions"] for item in items]
    return batch
