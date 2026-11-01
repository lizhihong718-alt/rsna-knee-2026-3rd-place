#!/usr/bin/env python3
"""Build low-resolution 2.5D CoAtNet windows from full-volume NPZ files.

Unlike the published Raptor stack, each three-channel item is centred on one
sampled slice and uses its actual consecutive source neighbours. Invalid or
widely separated neighbours conservatively repeat the centre channel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


SLOTS = (
    ("sagittal_fluid_fs", 18, 0),
    ("sagittal_nonfluid", 14, 0),
    ("coronal_fluid_fs", 12, 1),
    ("coronal_nonfluid", 8, 1),
    ("axial_any", 12, 2),
)
MAX_WINDOWS = sum(item[1] for item in SLOTS)
SPAN = (0.06, 0.94)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _present(z, prefix):
    return bool(np.asarray(z.get(prefix + "_valid", 0)).item()) and prefix + "_volume" in z


def _quality(z, prefix):
    value = np.asarray(z.get(prefix + "_quality", np.zeros(8)), dtype=np.float32)
    return float(value[0]) if value.size else 0.0


def _prefix(z, slot):
    if slot != "axial_any":
        prefix = f"slot_{slot}_s0"
        return prefix if _present(z, prefix) else None
    options = [f"slot_axial_fluid_fs_s{k}" for k in range(2)]
    options += [f"slot_axial_nonfluid_s{k}" for k in range(2)]
    options = [prefix for prefix in options if _present(z, prefix)]
    return max(options, key=lambda prefix: _quality(z, prefix), default=None)


def _centres(count, budget):
    if count < 1:
        return np.empty(0, dtype=np.int64)
    positions = np.linspace(SPAN[0] * (count - 1), SPAN[1] * (count - 1), budget)
    # Stable uniqueness prevents a short series from pretending repeated centres
    # are independent evidence.
    return np.unique(np.rint(positions).astype(np.int64))


def _resize(frames, resolution):
    if frames.shape[-2:] == (resolution, resolution):
        return frames
    import cv2
    cv2.setNumThreads(1)
    return np.stack([cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
                     for frame in frames])


def _physical_center_crop(frames, spacing_mm_per_px, fov_mm):
    """Crop a square physical field from already orientation-normalized frames."""
    frames = np.asarray(frames)
    spacing = np.asarray(spacing_mm_per_px, dtype=np.float64)
    if frames.ndim != 3 or spacing.shape != (2,) or not np.isfinite(spacing).all() \
            or not (spacing > 0).all() or not np.isfinite(fov_mm) or fov_mm <= 0:
        raise ValueError("physical cropping requires [N,H,W], positive spacing and fov")
    target_h = max(1, int(round(float(fov_mm) / spacing[0])))
    target_w = max(1, int(round(float(fov_mm) / spacing[1])))
    height, width = frames.shape[-2:]
    if target_h > 4 * height or target_w > 4 * width:
        raise ValueError("requested physical crop is implausibly larger than the source field")
    pad_h, pad_w = max(0, target_h - height), max(0, target_w - width)
    if pad_h or pad_w:
        frames = np.pad(frames, ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                                 (pad_w // 2, pad_w - pad_w // 2)))
    top = (frames.shape[-2] - target_h) // 2
    left = (frames.shape[-1] - target_w) // 2
    return np.ascontiguousarray(frames[:, top:top + target_h, left:left + target_w])


def _extract_series(z, prefix, budget, resolution, crop_fov_mm=None):
    volume = np.asarray(z[prefix + "_volume"])
    if volume.dtype != np.uint8 or volume.ndim != 3:
        raise ValueError(f"{prefix} volume must be uint8 [slice,H,W]")
    count = len(volume)
    centres = _centres(count, budget)
    source = np.asarray(z.get(prefix + "_source_indices", np.arange(count)), dtype=np.int64)
    positions = np.asarray(z.get(prefix + "_z", np.full(count, np.nan)), dtype=np.float64)
    if source.shape != (count,) or positions.shape != (count,):
        raise ValueError(f"{prefix} source metadata differs from volume depth")
    geometry = bool(np.asarray(z.get(prefix + "_geometry_valid", False)).item())
    finite_geometry = geometry and np.isfinite(positions).all() and np.ptp(positions) > 0
    neighbours = np.clip(centres[:, None] + np.array([-1, 0, 1]), 0, count - 1)
    if finite_geometry:
        spacing = np.abs(np.diff(positions))
        positive = spacing[np.isfinite(spacing) & (spacing > 0.05)]
        typical = float(np.median(positive)) if len(positive) else 0.0
    else:
        typical = 0.0
    for channel in (0, 2):
        candidate = neighbours[:, channel]
        distance = np.abs(positions[candidate] - positions[centres])
        adjacent = np.abs(source[candidate] - source[centres]) == 1
        acceptable = (finite_geometry & adjacent & (distance > 0.05)
                      & (distance <= max(typical * 1.8, 0.05)))
        neighbours[:, channel] = np.where(acceptable, candidate, centres)
    unique, inverse = np.unique(neighbours, return_inverse=True)
    selected = volume[unique]
    if crop_fov_mm is not None:
        spacing = np.asarray(z.get(prefix + "_output_spacing_mm_per_px", []), dtype=np.float64)
        selected = _physical_center_crop(selected, spacing, crop_fov_mm)
    resized = _resize(selected, resolution)
    windows = resized[inverse].reshape(len(centres), 3, resolution, resolution)
    relative = np.zeros(len(centres), dtype=np.float32)
    reliable = np.zeros(len(centres), dtype=np.uint8)
    if finite_geometry:
        relative = (positions[centres] - (positions.min() + positions.max()) / 2).astype(np.float32)
        reliable[:] = 1
    meta = np.asarray(z.get(prefix + "_meta_codes", np.full(3, 255)), dtype=np.int64)
    fluid = int(meta[1]) if meta.shape == (3,) and int(meta[1]) in (0, 1) else 0
    return windows, relative, reliable, fluid, centres, neighbours


def build_one(row, uid, path, resolution, signature, crop_fov_mm=None):
    windows = np.zeros((MAX_WINDOWS, 3, resolution, resolution), dtype=np.uint8)
    mask = np.zeros(MAX_WINDOWS, dtype=np.uint8)
    series = np.full(MAX_WINDOWS, -1, dtype=np.int16)
    groups = np.full(MAX_WINDOWS, -1, dtype=np.int8)
    planes = np.full(MAX_WINDOWS, -1, dtype=np.int8)
    position = np.zeros(MAX_WINDOWS, dtype=np.float32)
    reliable = np.zeros(MAX_WINDOWS, dtype=np.uint8)
    details = []
    with (nullcontext(path) if isinstance(path, dict) else np.load(path)) as z:
        if int(np.asarray(z.get("schema_version", 0)).item()) != 2:
            raise ValueError("full-volume NPZ must use schema version 2")
        if str(np.asarray(z.get("preprocess_signature", "")).item()) != signature:
            raise ValueError("full-volume NPZ preprocessing signature changed")
        cursor = 0
        for series_id, (slot, budget, plane) in enumerate(SLOTS):
            prefix = _prefix(z, slot)
            if prefix is None:
                details.append({"slot": slot, "present": False, "windows": 0})
                cursor += budget
                continue
            result = _extract_series(z, prefix, budget, resolution, crop_fov_mm)
            image, relative, is_reliable, fluid, centres, neighbours = result
            count = len(image)
            target = slice(cursor, cursor + count)
            windows[target] = image
            mask[target] = 1
            series[target] = series_id
            groups[target] = plane * 2 + fluid
            planes[target] = plane
            position[target] = relative
            reliable[target] = is_reliable
            details.append({"slot": slot, "source_prefix": prefix, "present": True,
                            "windows": count, "centres": centres.tolist(),
                            "neighbours": neighbours.tolist()})
            cursor += budget
    return row, windows, mask, series, groups, planes, position, reliable, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--ids-csv", type=Path,
                        help="CSV containing StudyInstanceUID rows; defaults to data-root/train.csv")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/train_v2"))
    parser.add_argument("--out", type=Path, default=Path("data/cache/coatnet_original224"))
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--crop-fov-mm", type=float,
                        help="Optional physical center crop applied before resizing")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if (args.resolution < 64 or args.workers < 1
            or args.crop_fov_mm is not None
            and (not np.isfinite(args.crop_fov_mm) or args.crop_fov_mm <= 0)):
        raise ValueError("resolution/workers/crop FOV must be positive and valid")
    ids_csv = args.ids_csv or (args.data_root / "train.csv")
    ids = pd.read_csv(ids_csv, dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate study UID")
    files = [args.processed_dir / f"{uid}.npz" for uid in ids]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} processed studies missing; first={missing[0]}")
    with np.load(files[0]) as first:
        signature = str(np.asarray(first["preprocess_signature"]).item())
    preprocessing = {
        "schema": "coatnet-original-neighbors-v1",
        "window_semantics": "original_consecutive_source_slices",
        "source_schema": 2,
        "source_preprocess_signature": signature,
        "source_processed_dir": str(args.processed_dir.resolve()),
        "source_train_csv_sha256": sha256(ids_csv),
        "slots": [list(item) for item in SLOTS],
        "span": list(SPAN),
        "image": args.resolution,
        "neighbor_gap_ratio": 1.8,
    }
    if args.crop_fov_mm is not None:
        preprocessing.update(
            crop_fov_mm=float(args.crop_fov_mm),
            crop_stage="after_patient_space_orientation_before_model_resize",
            source_orientation="patient_space_laterality_normalized",
        )
    shape = [len(ids), MAX_WINDOWS, 3, args.resolution, args.resolution]
    expected = {"preprocessing": preprocessing, "study_uids": ids, "shape": shape}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != expected:
            raise ValueError("corpus manifest differs; use a new output directory")
    else:
        if list(args.out.glob("*.npy")):
            raise ValueError("untracked arrays in output directory")
        manifest_path.write_text(json.dumps(expected, indent=2))
    specs = {
        "windows": (np.uint8, tuple(shape), 0),
        "masks": (np.uint8, (len(ids), MAX_WINDOWS), 0),
        "series_index": (np.int16, (len(ids), MAX_WINDOWS), -1),
        "group_id": (np.int8, (len(ids), MAX_WINDOWS), -1),
        "plane_id": (np.int8, (len(ids), MAX_WINDOWS), -1),
        "position_mm": (np.float32, (len(ids), MAX_WINDOWS), 0),
        "position_reliable": (np.uint8, (len(ids), MAX_WINDOWS), 0),
        "complete": (np.uint8, (len(ids),), 0),
    }
    arrays = {}
    for name, (dtype, array_shape, fill) in specs.items():
        path = args.out / f"{name}.npy"
        if path.exists():
            array = np.load(path, mmap_mode="r+")
            if array.shape != array_shape or array.dtype != dtype:
                raise ValueError(f"existing {name} has incompatible layout")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=array_shape)
            array[:] = fill
            array.flush()
        arrays[name] = array
    done = arrays["complete"]
    pending = iter(i for i in range(len(ids)) if not done[i])
    reports = [None] * len(ids)
    failures = []
    started = time.time()
    completed = int(done.sum())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}

        def submit_one():
            row = next(pending, None)
            if row is None:
                return False
            future = pool.submit(build_one, row, ids[row], str(files[row]), args.resolution,
                                 signature, args.crop_fov_mm)
            futures[future] = row
            return True

        for _ in range(args.workers * 2):
            if not submit_one():
                break
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                row = futures.pop(future)
                try:
                    result = future.result()
                    _, window, mask, series, group, plane, position, reliable, detail = result
                    arrays["windows"][row] = window
                    arrays["masks"][row] = mask
                    arrays["series_index"][row] = series
                    arrays["group_id"][row] = group
                    arrays["plane_id"][row] = plane
                    arrays["position_mm"][row] = position
                    arrays["position_reliable"][row] = reliable
                    for name, array in arrays.items():
                        if name != "complete":
                            array.flush()
                    done[row] = 1
                    done.flush()
                    reports[row] = detail
                    completed += 1
                except Exception as error:
                    failures.append({"study_uid": ids[row], "error": repr(error)})
                    print(f"ERROR {ids[row]}: {error}", flush=True)
                if completed % 50 == 0:
                    print(f"{completed}/{len(ids)} cached; elapsed={time.time() - started:.0f}s", flush=True)
                submit_one()
    valid_counts = np.asarray(arrays["masks"], dtype=np.uint8).sum(1)
    qa = {
        "completed": int(done.sum()),
        "total": len(ids),
        "failures": failures,
        "empty_study_uids": [ids[i] for i in np.flatnonzero(valid_counts == 0)],
        "candidate_count_min": int(valid_counts.min()),
        "candidate_count_median": float(np.median(valid_counts)),
        "candidate_count_max": int(valid_counts.max()),
        "elapsed_s": time.time() - started,
    }
    (args.out / "qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2), flush=True)
    # Empty studies are valid corpus rows and are excluded by the audited cohort
    # loader. The two known corrupt training studies intentionally have no NPZ
    # candidates; only decoding failures or unpublished rows make the build fail.
    if failures or not done.all():
        raise RuntimeError("original-neighbour corpus is incomplete")


if __name__ == "__main__":
    main()
