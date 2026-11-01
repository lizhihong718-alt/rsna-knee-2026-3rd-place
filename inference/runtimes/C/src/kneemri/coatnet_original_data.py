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
# cv2 interpolation flags for the training warp (values fixed by OpenCV).
WARP_FLAGS = {"linear": 1, "cubic": 2, "area": 3, "lanczos4": 4}


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

    ``uint8_output`` keeps the (augmented) pixels as uint8 and returns the
    per-study intensity transform as an ``intensity_lut`` [3,256] FP32 table.
    Every intensity step is elementwise on a uint8 value, so gathering through
    the table (``expand_intensity``) reproduces the float path bit for bit while
    moving a quarter of the bytes. It only applies when no resize follows the
    intensity transform; otherwise the float path is used unchanged.
    """

    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
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
        self.uint8_output = bool(uint8_output) and self.resolution == self.source_resolution
        self.manifest_sha256 = _sha256(manifest_path)
        self.windows = self.masks = None
        self.metadata = {}
        # Optional laterality canonicalization for studies without a usable
        # Laterality tag (per-window horizontal mirror / prev-next swap),
        # applied identically wherever pixels are read: training, Gold58 and
        # multi-budget evaluation all go through _item.
        self.flip = self.swap = None
        orientation = cfg.get("canonical_orientation")
        if orientation:
            with np.load(orientation) as z:
                if z["study_uids"].tolist() != list(all_ids):
                    raise ValueError("orientation artifact rows differ from the corpus")
                self.flip, self.swap = z["flip"].astype(bool), z["swap"].astype(bool)
            if self.flip.shape != (len(all_ids), shape[1]) or self.swap.shape != self.flip.shape:
                raise ValueError("orientation artifact must be [study, candidate]")

    def _open(self):
        if self.windows is not None:
            return
        self.windows = np.load(self.root / "windows.npy", mmap_mode="r")
        self.masks = np.load(self.root / "masks.npy", mmap_mode="r")
        self.metadata = {name: np.load(self.root / f"{name}.npy", mmap_mode="r")
                         for name in ARRAYS}
        expected = self.windows.shape[:2]
        if self.windows.shape != tuple(self.manifest["shape"]) or self.windows.dtype != np.uint8:
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

    def _selected(self, row, budget=None):
        valid = np.flatnonzero(np.asarray(self.masks[row], dtype=bool))
        candidates = np.repeat(valid[:, None], 3, axis=1)
        if budget is None:
            budget = random.choice(self.budget) if isinstance(self.budget, list) else self.budget
        groups = np.asarray(self.metadata["series_index"][row])[valid]
        rng = np.random.default_rng(random.getrandbits(64)) if self.train else None
        selected = select_windows(candidates, budget, strategy=self.strategy, rng=rng,
                                  group_ids=groups if self.strategy in {"group_balanced", "stratified"} else None)
        return selected[:, 0], len(valid), budget

    def _canonical(self, row, selected, raw):
        """uint8 [K,3,H,W] -> canonical laterality (mirror W axis; reverse slice channels)."""
        if self.flip is None:
            return raw
        flip, swap = self.flip[row, selected], self.swap[row, selected]
        if flip.any() or swap.any():
            raw = np.array(raw, copy=True)
            raw[flip] = raw[flip][..., ::-1]
            raw[swap] = raw[swap][:, ::-1]
        return raw

    def _augment(self, images):
        return self._intensity(self._geometric(images))

    def _blur(self, images):
        """Training-only random Gaussian blur of whole windows (runtime_b; default off).

        ``blur_prob`` > 0 blurs a study's selected windows with probability p using one
        sigma ~ U(blur_sigma) px, mimicking the sharpness spread between upsampled and
        downsampled native series. The draw happens only when enabled (legacy RNG order).
        """
        prob = float(self.cfg.get("blur_prob", 0.0))
        if not self.train or prob <= 0 or random.random() >= prob:
            return images
        import cv2
        lo, hi = self.cfg.get("blur_sigma", [0.3, 1.0])
        sigma = random.uniform(float(lo), float(hi))
        h = images.shape[-1]
        flat = images.reshape(-1, h, h)
        out = np.stack([cv2.GaussianBlur(frame, (0, 0), sigma, borderType=cv2.BORDER_REFLECT) for frame in flat])
        return out.reshape(images.shape)

    def _resample(self, images):
        """Training-only random resolution degradation of a whole bag (native512 route; default off).

        With probability ``resample_prob`` every selected window goes INTER_AREA down to a
        random fraction f ~ U(resample_scale) of its side and INTER_LINEAR back up: the very
        operation that produced the 64% upsampled native512 series (raw 140 mm extent < 512 px,
        analysis/spectrum_512.json), including its linear-interpolation imaging pattern. So
        that pattern and the missing band stop being a scanner signature. Draws happen only
        when enabled (legacy RNG order).
        """
        prob = float(self.cfg.get("resample_prob", 0.0))
        if not self.train or prob <= 0 or random.random() >= prob:
            return images
        import cv2
        lo, hi = self.cfg.get("resample_scale", [0.5, 0.9])
        f = random.uniform(float(lo), float(hi))
        h = images.shape[-1]
        s = max(8, int(round(h * f)))
        flat = images.reshape(-1, h, h)
        out = np.stack([cv2.resize(cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA), (h, h),
                                   interpolation=cv2.INTER_LINEAR) for frame in flat])
        return out.reshape(images.shape)

    def _geometric(self, images):
        """uint8 -> uint8 spatial augmentation; draws rotation then scale.

        native384 audit (reports/cnx_dino_384_20261009/analysis/spectrum_384.json):
        an INTER_LINEAR warp keeps only ~0.65x of the 0.5-1.0 Nyquist power while
        evaluation windows are never warped. ``warp_interpolation`` (default linear,
        the legacy path) and ``geometric_prob`` (default 1, legacy) close that gap;
        the probability draw happens only when it is below 1, so legacy RNG order holds.
        """
        images = self._resample(self._blur(images))
        if self.train and (self.rotation or self.scale_jitter):
            import cv2
            cv2.setNumThreads(1)
            h = images.shape[-1]
            prob = float(self.cfg.get("geometric_prob", 1.0))
            if prob < 1.0 and random.random() >= prob:
                return images
            angle = random.uniform(-self.rotation, self.rotation)
            scale = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
            affine = cv2.getRotationMatrix2D(((h - 1) / 2, (h - 1) / 2), angle, scale)
            flag = WARP_FLAGS[str(self.cfg.get("warp_interpolation", "linear"))]
            flat = images.reshape(-1, h, h)
            images = np.stack([cv2.warpAffine(frame, affine, (h, h), flags=flag,
                                               borderMode=cv2.BORDER_CONSTANT) for frame in flat])
            images = images.reshape(-1, 3, h, h)
        return images

    def _intensity(self, images):
        """Elementwise uint8 -> [0,1] FP32 transform; draws gamma then gain."""
        if not self.train:
            return images.astype(np.float32) / 255.0
        values = images.astype(np.float32) / 255.0
        if self.gamma_jitter:
            values = values ** random.uniform(1 - self.gamma_jitter, 1 + self.gamma_jitter)
        if self.gain:
            values *= 1 + random.uniform(-self.gain, self.gain)
        return np.clip(values, 0, 1)

    def _normalized_lut(self, values):
        """[256] FP32 intensities -> [3,256] with the float path's normalization op."""
        table = torch.from_numpy(values).view(1, 1, 256)
        return ((table - self.mean) / self.std).reshape(3, 256).contiguous()

    def __getitem__(self, index):
        self._open()
        uid = self.ids[index]
        row = self.rows[uid]
        selected, candidate_count, budget = self._selected(row)
        item = self._item(uid, row, selected)
        item["provenance"] = self._trace(uid, selected, candidate_count, budget)
        return item

    def _trace(self, uid, selected, candidate_count, budget):
        return {
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

    def _item(self, uid, row, selected):
        """Pixels, targets and window metadata for already selected candidates."""
        lut = None
        if len(selected):
            raw = self._canonical(row, selected, np.asarray(self.windows[row, selected]))
            if self.uint8_output:
                # Same RNG draws in the same order as _augment: geometry on the
                # pixels, then the intensity transform on all 256 possible values.
                x = torch.from_numpy(np.ascontiguousarray(self._geometric(raw)))
                lut = self._normalized_lut(self._intensity(np.arange(256, dtype=np.uint8)))
            else:
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
            x = torch.zeros((1, 3, self.resolution, self.resolution),
                            dtype=torch.uint8 if self.uint8_output else torch.float32)
            if self.uint8_output:
                # Padding maps to exact zeros, as the float path's zero tensor.
                lut = torch.zeros(3, 256)
            valid = torch.zeros(1, dtype=torch.bool)
            groups = series = planes = np.full(1, -1, np.int64)
            positions = np.zeros((1, 1), np.float32)
            position_mask = np.zeros(1, bool)
        label = np.asarray(self.labels[uid], dtype=np.float32)
        if label.shape != (12,):
            raise ValueError(f"expected the explicit 12-label contract for {uid}")
        if self.train and self.cfg.get("target_mode", "soft") == "hard":
            # Astra stores calibrated soft probabilities, while the competition
            # metric ranks the explicit >0.5 disease states.  Hard targets are
            # an opt-in training ablation; evaluation keeps the original values.
            label = (label > 0.5).astype(np.float32)
        item = {
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
        }
        if self.uint8_output:
            item["intensity_lut"] = lut
        return item


class MultiBudgetEvaluationBags(Dataset):
    """Deterministic evaluation of several window budgets from one pixel read.

    Every budget selects from the same per-study candidate list, so each is a
    subset of the union of all requested budgets. An item holds that union once
    plus, per budget, its positions inside the union and the provenance trace
    the single-budget dataset would have produced.
    """

    def __init__(self, cfg, all_ids, ids, labels, budgets, uint8_output=True):
        self.budgets = [_budget(budget) for budget in budgets]
        if not self.budgets or len({str(budget) for budget in self.budgets}) != len(self.budgets):
            raise ValueError("evaluation budgets must be nonempty and distinct")
        self.base = CoAtNetOriginalNeighborBags(dict(cfg, eval_windows=self.budgets[0]), all_ids, ids,
                                                labels, train=False, uint8_output=uint8_output)
        self.uint8_output = self.base.uint8_output

    def __len__(self):
        return len(self.base)

    def pixel_bytes(self):
        """Bytes of the pixel tensors all items hold, from masks alone (no pixel reads)."""
        base = self.base
        base._open()
        per_window = 3 * base.resolution ** 2 * (1 if self.uint8_output else 4)
        total = 0
        for uid in base.ids:
            row = base.rows[uid]
            union = set()
            for budget in self.budgets:
                union.update(base._selected(row, budget)[0].tolist())
            total += max(1, len(union)) * per_window
        return total

    def __getitem__(self, index):
        base = self.base
        base._open()
        uid = base.ids[index]
        row = base.rows[uid]
        selections = [base._selected(row, budget) for budget in self.budgets]
        union = np.unique(np.concatenate([selected for selected, _, _ in selections]))
        item = base._item(uid, row, union)
        positions, traces = {}, {}
        for budget, (selected, candidate_count, _) in zip(self.budgets, selections):
            # An empty study has one invalid padding window, at position 0.
            positions[str(budget)] = (np.searchsorted(union, selected) if len(selected)
                                      else np.zeros(1, dtype=np.int64))
            traces[str(budget)] = base._trace(uid, selected, candidate_count, budget)
        item.update(budget_positions=positions, provenance=traces)
        return item


def expand_intensity(images, lut):
    """Gather uint8 [B,K,3,H,W] through per-study [B,3,256] tables -> FP32.

    Runs on whichever device holds the tensors; the result equals the float
    dataset path exactly because each output is the same FP32 table entry.
    """
    if images.dtype != torch.uint8 or images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("expand_intensity expects uint8 images [B,K,3,H,W]")
    if lut.shape != (images.shape[0], 3, 256):
        raise ValueError("intensity tables must be [B,3,256]")
    index = images.long()
    index += torch.arange(images.shape[0] * 3, device=images.device).view(-1, 1, 3, 1, 1) * 256
    return torch.take(lut.to(device=images.device, dtype=torch.float32).reshape(-1), index)
