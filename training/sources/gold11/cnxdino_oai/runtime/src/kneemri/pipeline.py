"""Study-level preprocessing pipeline (sections 3-8 composed).

    per study:
      map series -> canonical slots            (slots.py)
      for each filled slot:
        order slices physically                (geometry.py)
        decode pixel data                      (dicom.py, injectable for tests)
        physical-mm crop -> resize             (crop.py)
        laterality normalization               (laterality.py)
        per-series intensity normalization     (crop.py)
        fixed slice sampling positions         (sampling.py)
      store one npz: per-slot volume + sampled positions + report

The npz intentionally stores the WHOLE processed volume per slot plus the
sampled positions — sampling/2.5D assembly happens dataset-side, so sampling
experiments never require re-decoding DICOM.
"""
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import asdict, replace
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from kneemri.config import PreprocessPipelineConfig
from kneemri.crop import apply_crop_resize, normalize_intensity, plan_physical_crop
from kneemri.geometry import IOP_COLS, IPP_COLS, sort_series_geometry
from kneemri.laterality import (apply_horizontal_flip, apply_stack_reversal,
                                plan_orientation, series_laterality)
from kneemri.sampling import sample_positions

# decode_fn: path -> (H, W) float32. Injectable so tests use synthetic volumes.
DecodeFn = Callable[[str], np.ndarray]


@dataclass
class SlotResult:
    slot_id: str
    series_uid: str
    valid: bool
    volume: np.ndarray | None = None          # (n_slices, H, W) uint8 or float32
    crop_mask: np.ndarray | None = None       # (H, W) or (n, H, W) uint8, 1=real px
    positions: np.ndarray | None = None       # sampled indices, (slices_per_slot,)
    z: np.ndarray | None = None               # physical z per stored slice (NaN ok)
    source_spacing_mm_per_px: tuple[float, float] | None = None
    output_spacing_mm_per_px: tuple[float, float] | None = None
    intensity_lo_hi: tuple[float, float] | None = None
    meta_codes: np.ndarray | None = None      # (3,) uint8 plane/fluid/fat-sup
    source_indices: np.ndarray | None = None
    geometry_valid: bool = False
    quality: np.ndarray | None = None
    n_warnings: int = 0
    error: str | None = None
    report: dict = field(default_factory=dict)


@dataclass
class StudyResult:
    study_uid: str
    # slot_id -> per-rank results (rank j = Top-K candidate j; rank 0 = Top-1)
    slot_ranks: dict[str, list[SlotResult]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)   # study-level metadata
    failed: bool = False
    error: str | None = None

    @property
    def slots(self) -> dict[str, SlotResult]:
        """Top-1 view (rank 0 of each slot) — the baseline interface."""
        return {sid: ranks[0] for sid, ranks in self.slot_ranks.items()
                if ranks}

    @property
    def n_valid_slots(self) -> int:
        return sum(1 for ranks in self.slot_ranks.values()
                   if any(s.valid for s in ranks))


def _decode_rows(rows: pd.DataFrame, study_root: str, decode: DecodeFn,
                 errors: list[str], return_mask: bool = False):
    """Decode slices in frame order.

    Returns (volume (n, H, W) float32, rows actually decoded, kept indices).
    Slices that fail to decode are dropped (error recorded), never silently
    shuffled into the wrong physical position.
    """
    imgs, kept = [], []
    for i, row in enumerate(rows.itertuples(index=False)):
        try:
            img = decode(os.path.join(study_root, row.rel_path))
        except Exception as exc:
            errors.append(f"slice {i} ({getattr(row, 'rel_path', '?')}): {exc}")
            continue
        imgs.append(img)
        kept.append(i)
    if not imgs:
        raise RuntimeError("no slice decoded successfully")
    if len({im.shape for im in imgs}) > 1:
        errors.append(
            f"mixed in-plane sizes {[sorted({im.shape for im in imgs})]} — slices "
            "center-padded to the largest; padded regions are recorded in the crop mask")
    rows_kept = rows.iloc[kept].reset_index(drop=True)
    if return_mask:
        if len({im.shape for im in imgs}) == 1:
            # No slice was padded, so the acquired-pixel mask is all ones and
            # callers must treat None exactly as that array: flips and stack
            # reversal leave it unchanged and `.all()` would be True.
            return _stack_same_shape(imgs), rows_kept, np.asarray(kept, dtype=int), None
        shape=(max(im.shape[0] for im in imgs),max(im.shape[1] for im in imgs))
        mask=np.stack([_pad_to(np.ones(im.shape,np.uint8),shape) for im in imgs])
        return _stack_same_shape(imgs),rows_kept,np.asarray(kept,dtype=int),mask
    return _stack_same_shape(imgs), rows_kept, np.asarray(kept, dtype=int)


def _stack_same_shape(imgs: list[np.ndarray]) -> np.ndarray:
    shapes = {im.shape for im in imgs}
    if len(shapes) > 1:  # mixed in-plane sizes inside one series
        target = (max(s[0] for s in shapes), max(s[1] for s in shapes))
        imgs = [_pad_to(im, target) for im in imgs]
    return np.stack(imgs)


def _pad_to(img: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Center-pad a slice to `shape` with zeros. Top-left anchoring would
    mis-register the anatomy relative to the later centered crop."""
    dh, dw = shape[0] - img.shape[0], shape[1] - img.shape[1]
    out = np.zeros(shape, dtype=img.dtype)
    out[(dh + 1) // 2:(dh + 1) // 2 + img.shape[0],
        (dw + 1) // 2:(dw + 1) // 2 + img.shape[1]] = img
    return out


def process_slot(slot_id: str, series_uid: str, rows: pd.DataFrame,
                 plane: str, laterality: str, laterality_source: str,
                 rank: int, fluid_code: int, fat_sup_code: int,
                 study_root: str, cfg: PreprocessPipelineConfig,
                 decode: DecodeFn) -> SlotResult:
    """Run geometry -> decode -> orientation -> crop -> intensity for one slot.

    laterality: effective laterality for this series (series tag or the
    study-level consensus — see process_study), '' when unknown.
    rank: Top-K candidate rank of this series within the slot (0 = Top-1).
    """
    errors: list[str] = []
    pcfg, gcfg = cfg.preprocessing, cfg.geometry
    result = SlotResult(slot_id=slot_id, series_uid=series_uid, valid=False)
    result.report["rank"] = rank

    ordering = sort_series_geometry(rows, gcfg, label=f"{slot_id}/{series_uid}")
    result.report["sort_source"] = ordering.source
    result.report["geometry_warnings"] = ordering.warnings
    rows_ord = ordering.ordered(rows)

    vol, rows_ord, kept_idx, source_mask = _decode_rows(rows_ord, study_root, decode, errors,return_mask=True)
    if len(vol) < gcfg.min_slices:
        raise ValueError("too few successfully decoded slices")
    if errors:
        result.report["decode_errors"] = errors
        result.n_warnings += len(errors)

    # --- orientation / laterality canonicalization ---------------------------
    # direction policy needs the series-median IOP; use the one the geometry
    # stage computed (valid even on fallback paths with partial geometry),
    # falling back to a component-wise median of what remains
    med_iop = ordering.iop if ordering.iop is not None \
        else rows_ord[IOP_COLS].median().to_numpy(dtype=float)
    decision = plan_orientation(med_iop, plane, laterality, cfg.laterality)
    if cfg.laterality.normalize:
        vol = apply_horizontal_flip(vol, decision.horizontal_flip)
        vol = apply_stack_reversal(vol, decision.reverse_stack)
        if source_mask is not None:  # an all-ones mask is invariant to both
            source_mask = apply_horizontal_flip(source_mask, decision.horizontal_flip)
            source_mask = apply_stack_reversal(source_mask, decision.reverse_stack)
    result.report["orientation"] = {
        "laterality_tag": laterality,
        "laterality_source": laterality_source,
        "plane": plane,
        "horizontal_flip": decision.horizontal_flip,
        "reverse_stack": decision.reverse_stack,
        "horizontal_axis": decision.horizontal_axis,
        "stack_axis": decision.stack_axis,
        "reason": decision.reason,
        "warnings": decision.warnings,
        "applied": cfg.laterality.normalize,
    }
    result.n_warnings += len(decision.warnings)

    spacing_row = float(rows_ord["spacing_row"].median())
    spacing_col = float(rows_ord["spacing_col"].median())
    if pcfg.crop_mode not in ("full", "physical"):
        raise ValueError(f"unknown crop_mode {pcfg.crop_mode}")
    fov = max(vol.shape[1] * spacing_row, vol.shape[2] * spacing_col) \
        if pcfg.crop_mode == "full" else pcfg.crop_fov_mm
    plan = plan_physical_crop(spacing_row, spacing_col, vol.shape[1], vol.shape[2], fov)
    result.report["crop"] = {"fov_mm": fov, "mode": pcfg.crop_mode,
                             "spacing_row": spacing_row, "spacing_col": spacing_col,
                             "padded": plan.padded,
                             "pad_rows": plan.pad_rows, "pad_cols": plan.pad_cols}
    n = vol.shape[0]
    processed = [apply_crop_resize(vol[i], plan, pcfg.output_size,
                                   pcfg.resize_interpolation, return_mask=True)
                 for i in range(n)]
    vol = np.stack([p[0] for p in processed])
    crop_mask = np.stack([p[1] for p in processed])   # (n, H, W) uint8
    if source_mask is not None and not source_mask.all():
        acquired=np.stack([apply_crop_resize(m.astype(np.float32),plan,pcfg.output_size,'nearest') for m in source_mask])
        crop_mask &= (acquired>0.5).astype(np.uint8)

    # --- per-series intensity normalization ---------------------------------
    vol, istats = normalize_intensity(vol, pcfg.intensity_percentile_low,
                                      pcfg.intensity_percentile_high,
                                      pcfg.intensity_over)
    result.report["intensity"] = istats

    out_dtype = np.dtype(cfg.output.dtype)
    vol = np.clip(vol * 255.0 + 0.5, 0, 255).astype(out_dtype) if out_dtype == np.uint8 \
        else vol.astype(out_dtype)

    result.volume = np.ascontiguousarray(vol)
    result.crop_mask = crop_mask[0] if np.all(crop_mask == crop_mask[0]) else crop_mask
    # physical z per STORED slice (drops failed decodes; reversed if the
    # orientation policy flipped the stack)
    z_stored = ordering.z[ordering.order][kept_idx]
    reversed_stack = cfg.laterality.normalize and decision.reverse_stack
    result.z = z_stored[::-1].copy() if reversed_stack else z_stored
    result.source_indices = kept_idx[::-1].copy() if reversed_stack else kept_idx
    result.geometry_valid = ordering.source == "ipp_dot_normal" and not any(
        "deviate" in w for w in ordering.warnings)
    try:
        result.positions = sample_positions(vol.shape[0], cfg.sampling, result.z)
    except ValueError as exc:
        if "uniform_mm" in str(exc) and cfg.sampling.mode == "uniform_mm":
            # physical sampling impossible without z: fall back to index
            # sampling rather than lose the slot, and say so
            fallback_cfg = replace(cfg.sampling, mode="uniform_index")
            result.positions = sample_positions(vol.shape[0], fallback_cfg, None)
            result.report["sampling_fallback"] = str(exc)
            result.n_warnings += 1
        else:
            raise
    result.source_spacing_mm_per_px = (spacing_row, spacing_col)
    plane_code = {"Sagittal": 0, "Coronal": 1, "Axial": 2}.get(plane, 255)
    result.meta_codes = np.asarray([plane_code, fluid_code, fat_sup_code],
                                   dtype=np.uint8)
    # effective mm per output pixel of the stored volume (ROI physical extent
    # mapped onto output_size). plan.* are pixel coordinates -> convert to mm.
    roi_w_mm = (plan.x1 - plan.x0) * spacing_col
    roi_h_mm = (plan.y1 - plan.y0) * spacing_row
    result.output_spacing_mm_per_px = (roi_h_mm / pcfg.output_size,
                                        roi_w_mm / pcfg.output_size)
    result.intensity_lo_hi = (float(istats.get("lo", 0.0)),
                              float(istats.get("hi", 0.0)))
    result.valid = True
    return result


def process_study(study_uid: str, cfg: PreprocessPipelineConfig,
                  slice_meta: pd.DataFrame, series_table: pd.DataFrame,
                  decode: DecodeFn) -> StudyResult:
    """Preprocess one study into per-slot results.

    slice_meta    per-slice metadata cache rows for this study
    series_table  competition series-table rows for this study
    decode        DICOM pixel decoder (path -> (H, W) float32)
    """
    from kneemri.slots import map_study_to_slots, summarize_series

    result = StudyResult(study_uid=study_uid)
    if study_uid in set(cfg.blacklist):
        result.failed, result.error = True, "blacklisted"
        return result

    rows_by_series = {uid: sub for uid, sub in slice_meta.groupby("series_uid")}
    try:
        assignment = map_study_to_slots(
            study_uid, series_table, rows_by_series, cfg.slots)
    except Exception as exc:
        result.failed, result.error = True, f"slot mapping failed: {exc}"
        return result
    result.warnings.extend(assignment.warnings)

    # --- study-level laterality consensus ------------------------------------
    # ~50% of studies carry no Laterality tag; ~25 carry conflicting L/R tags
    # across their series (vendor errors on single-knee exams). A strict
    # study-level majority fills per-series gaps and overrides lone outlier
    # tags — one consistent side per study, with every conflict recorded.
    from kneemri.laterality import normalize_laterality_tag
    all_tags = [normalize_laterality_tag(v)
                for uid in rows_by_series for v in rows_by_series[uid]["laterality"].tolist()]
    tagged = [t for t in all_tags if t]
    consensus = ""
    if tagged:
        n_l, n_r = tagged.count("L"), tagged.count("R")
        if n_l > n_r:
            consensus = "L"
        elif n_r > n_l:
            consensus = "R"
    result.report["laterality_consensus"] = consensus

    def effective_laterality(series_rows: pd.DataFrame) -> tuple[str, str]:
        series_lat = series_laterality(series_rows["laterality"].tolist())
        if consensus and series_lat and series_lat != consensus:
            result.warnings.append(
                f"laterality conflict: series tag '{series_lat}' vs study consensus "
                f"'{consensus}' — using consensus")
            return consensus, "study_consensus_overrides_series"
        if consensus:
            return consensus, "study_consensus"
        return series_lat, "series_tag" if series_lat else "none"

    for slot_id in cfg.slots.definitions:
        candidates = assignment.slot_candidates.get(slot_id, [])
        ranks: list[SlotResult] = [SlotResult(slot_id=slot_id, series_uid="",
                                              valid=False)]
        result.slot_ranks[slot_id] = ranks
        if not candidates:
            continue
        ranks = [SlotResult(slot_id=slot_id, series_uid=cand.series_uid,
                            valid=False) for cand in candidates]
        result.slot_ranks[slot_id] = ranks
        for j, (cand, slot_res) in enumerate(zip(candidates, ranks)):
            series_uid = cand.series_uid
            rows = rows_by_series[series_uid]
            if len(rows) < cfg.geometry.min_slices:
                result.warnings.append(
                    f"slot '{slot_id}' rank {j}: series has {len(rows)} slices "
                    f"(< {cfg.geometry.min_slices}) — invalid")
                continue
            try:
                plane = cand.metadata["plane"]
                lat, lat_src = effective_laterality(rows)
                ranks[j] = process_slot(slot_id, series_uid, rows, plane,
                                        lat, lat_src, j,
                                        fluid_code=int(cand.metadata["fluid_sensitive"]),
                                        fat_sup_code=int(cand.metadata["fat_suppression"]),
                                        study_root=cfg.study_root, cfg=cfg,
                                        decode=decode)
                r = ranks[j]
                r.report["selection_score"] = cand.score
                r.report["score_breakdown"] = cand.score_breakdown
                r.quality = np.asarray([cand.score, float(r.geometry_valid),
                    min(len(r.volume) / 64, 4),
                    float(np.ptp(r.z)) / 200 if np.isfinite(r.z).all() else 0,
                    r.source_spacing_mm_per_px[0], r.source_spacing_mm_per_px[1],
                    float(bool(r.report.get("decode_errors"))),
                    float(not r.geometry_valid)], np.float32)
            except Exception as exc:
                slot_res.error = f"{type(exc).__name__}: {exc}"
                result.warnings.append(
                    f"slot '{slot_id}' rank {j} failed: {slot_res.error}")

    if result.n_valid_slots == 0:
        result.failed, result.error = True, "no valid slot"
    return result


def preprocess_signature(cfg):
    fields = ("slots", "geometry", "laterality", "preprocessing", "output")
    payload = {k: asdict(getattr(cfg, k)) for k in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_study(result: StudyResult, cfg: PreprocessPipelineConfig,
               out_dir: str | None = None) -> str:
    """Write <study_uid>.npz (+ sidecar report). Returns the npz path."""
    out_dir = out_dir or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{result.study_uid}.npz")
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2),
        "slot_ids": np.asarray(list(cfg.slots.definitions)),
        "preprocess_signature": np.asarray(preprocess_signature(cfg)),
    }
    for sid in cfg.slots.definitions:
        result.slot_ranks.setdefault(sid, [SlotResult(sid, "", False)])
    report: dict[str, dict] = {"study_uid": result.study_uid,
                               "warnings": result.warnings,
                               "failed": result.failed, "error": result.error,
                               **result.report,
                               "slots": {}}
    for slot_id, ranks in result.slot_ranks.items():
        prefix = f"slot_{slot_id}"
        slot_reports = []
        n_series = 0
        for j, slot in enumerate(ranks):
            rp = f"{prefix}_s{j}"
            if slot.valid and slot.volume is not None:
                n_series += 1
                n = int(slot.volume.shape[0])
                positions = slot.positions if slot.positions is not None else \
                    sample_positions(n, cfg.sampling, slot.z)
                arrays[f"{rp}_volume"] = slot.volume
                arrays[f"{rp}_positions"] = positions
                arrays[f"{rp}_n_distinct"] = np.asarray(len(np.unique(positions)))
                arrays[f"{rp}_valid"] = np.asarray(1, dtype=np.uint8)
                arrays[f"{rp}_z"] = slot.z
                arrays[f"{rp}_source_spacing_mm_per_px"] = np.asarray(
                    slot.source_spacing_mm_per_px, dtype=np.float32)
                arrays[f"{rp}_output_spacing_mm_per_px"] = np.asarray(
                    slot.output_spacing_mm_per_px, dtype=np.float32)
                arrays[f"{rp}_intensity_lo_hi"] = np.asarray(
                    slot.intensity_lo_hi, dtype=np.float32)
                arrays[f"{rp}_crop_mask"] = slot.crop_mask
                # categorical metadata for structural embeddings (section 12):
                # [plane(0=sag,1=cor,2=ax), fluid(0/1), fat_sup(0/1)]
                arrays[f"{rp}_meta_codes"] = slot.meta_codes
                arrays[f"{rp}_source_indices"] = slot.source_indices if slot.source_indices is not None else np.arange(n)
                arrays[f"{rp}_geometry_valid"] = np.asarray(slot.geometry_valid)
                arrays[f"{rp}_quality"] = slot.quality if slot.quality is not None else np.zeros(8, np.float32)
                slot_reports.append({
                    "rank": j,
                    "series_uid": slot.series_uid, "valid": True,
                    "n_slices": n,
                    "shape": list(slot.volume.shape[1:]),
                    "n_distinct_sampled": int(len(np.unique(positions))),
                    "n_warnings": slot.n_warnings, **slot.report,
                })
            else:
                arrays[f"{rp}_valid"] = np.asarray(0, dtype=np.uint8)
                slot_reports.append({
                    "rank": j,
                    "series_uid": slot.series_uid, "valid": False,
                    "error": slot.error, **slot.report,
                })
        arrays[f"{prefix}_n_series"] = np.asarray(n_series, dtype=np.int32)
        arrays[f"{prefix}_valid"] = np.asarray(
            1 if n_series > 0 else 0, dtype=np.uint8)
        report["slots"][slot_id] = {"n_series": n_series, "ranks": slot_reports}
    with open(path + ".tmp", "wb") as f:
        (np.savez_compressed if cfg.output.compression else np.savez)(f, **arrays)
    os.replace(path + ".tmp", path)
    if cfg.output.include_metadata_json:
        with open(path.replace(".npz", ".report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
    return path
