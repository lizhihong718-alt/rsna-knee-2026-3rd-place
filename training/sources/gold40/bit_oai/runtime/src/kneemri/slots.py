"""Canonical MRI series slots (pipeline section 4).

Each study's series are mapped onto a fixed set of named slots
(sagittal_fluid_fs, coronal_fluid_fs, ...). Slot identity comes from the
competition series table (Anatomical_Plane / Fluid_Sensitive), optionally with
DICOM metadata. A slot may be empty (validity mask = 0 downstream) or
contested (up to 6 candidate series in this corpus) — contested slots are
resolved by a modular, deterministic scoring rule whose per-component scores
are recorded for every assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from kneemri.config import SelectionComponent, SlotsConfig
from kneemri.geometry import IOP_COLS, IPP_COLS, slice_normal

SERIES_TABLE_COLS = ["StudyInstanceUID", "SeriesInstanceUID",
                     "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]

# Canonical slot order used by the model's slot embeddings — the Dataset,
# feature cache and inference all iterate slots in this order.
SLOT_ORDER = ["sagittal_fluid_fs", "coronal_fluid_fs", "axial_fluid_fs",
              "sagittal_nonfluid", "coronal_nonfluid", "axial_nonfluid"]

# ---------------------------------------------------------------- components
# Each component: fn(series_summary: dict, params: dict) -> float in [0, 1].
# series_summary is built by summarize_series() below.


def _completeness(s: dict, p: dict) -> float:
    """How much of the metadata we rely on is actually present."""
    n = len(s["rows"])
    if n == 0:
        return 0.0
    score = 0.0
    geo = (~s["rows"][IOP_COLS + IPP_COLS].isna().any(axis=1)).mean()
    score += p.get("geometry", 0.5) * float(geo)
    th = (~s["rows"]["slice_thickness"].isna()).mean() if "slice_thickness" in s["rows"] else 0.0
    score += p.get("thickness", 0.2) * float(th)
    sb = (~s["rows"]["spacing_between_slices"].isna()).mean() \
        if "spacing_between_slices" in s["rows"] else 0.0
    score += p.get("spacing_between", 0.2) * float(sb)
    seq_cols = [c for c in ("sequence_name", "series_description", "echo_time")
                if c in s["rows"] and s["rows"][c].notna().any()]
    score += p.get("sequence_meta", 0.1) * (min(len(seq_cols), 3) / 3)
    total_w = p.get("geometry", 0.5) + p.get("thickness", 0.2) + \
        p.get("spacing_between", 0.2) + p.get("sequence_meta", 0.1)
    return score / total_w


def _orientation_purity(s: dict, p: dict) -> float:
    """Mean |cos| between per-slice normals and the slot plane's ideal axis.
    1.0 = perfectly orthogonal stack, lower = oblique/mixed acquisition."""
    ideal = {"Sagittal": (1.0, 0.0, 0.0), "Coronal": (0.0, 1.0, 0.0),
             "Axial": (0.0, 0.0, 1.0)}[s["plane"]]
    iops = s["rows"][IOP_COLS].to_numpy(dtype=float)
    if np.isnan(iops).any():
        return 0.0
    normals = np.stack([slice_normal(iop) for iop in iops])
    return float(np.abs(normals @ np.asarray(ideal)).mean())


def _slice_count(s: dict, p: dict) -> float:
    ref = float(p.get("reference", 30))
    return float(min(len(s["rows"]) / ref, 1.0))


def _coverage_mm(s: dict, p: dict) -> float:
    z = s.get("z")
    if z is None or np.isnan(z).all():
        return 0.0
    ref = float(p.get("reference", 150))
    return float(min((np.nanmax(z) - np.nanmin(z)) / ref, 1.0))


def _in_plane_resolution(s: dict, p: dict) -> float:
    """Finer pixel spacing scores higher; saturates at reference_mm."""
    sp = s["rows"]["spacing_row"].median()
    if pd.isna(sp) or sp <= 0:
        return 0.0
    ref = float(p.get("reference_mm", 0.4))
    return float(min(ref / sp, 1.0))


def _sequence_suitability(s: dict, p: dict) -> float:
    """TE/TR plausibility for the slot's fluid class (fluid-sensitive sequences
    are long-TE T2-weighted; non-fluid are short-TE T1/PD). Missing TE/TR is
    neutral (0.5) rather than penalized — some vendors scrub it."""
    te = s["rows"]["echo_time"].median() if "echo_time" in s["rows"] else np.nan
    if pd.isna(te):
        return 0.5
    if s["fluid_sensitive"] == 1:
        return float(np.clip((te - 20.0) / 30.0, 0.0, 1.0))  # TE>=50ms plausible
    return float(np.clip((40.0 - te) / 30.0, 0.0, 1.0))      # TE<=10ms plausible


COMPONENTS: dict[str, Callable[[dict, dict], float]] = {
    "metadata_completeness": _completeness,
    "orientation_purity": _orientation_purity,
    "slice_count": _slice_count,
    "coverage_mm": _coverage_mm,
    "in_plane_resolution": _in_plane_resolution,
    "sequence_suitability": _sequence_suitability,
}


@dataclass
class SeriesCandidate:
    """One ranked candidate series for a slot (rank 0 = the Top-1 pick)."""
    series_uid: str
    score: float
    score_breakdown: dict
    metadata: dict          # plane / fluid_sensitive / fat_suppression / n_slices


@dataclass
class SlotAssignment:
    """Result of slot mapping for ONE study."""
    study_uid: str
    # slot_id -> series_uid ("" if slot empty) — Top-1, kept for baseline use
    slot_series: dict[str, str] = field(default_factory=dict)
    # slot_id -> ranked candidates (Top-K, best first); empty list if slot empty
    slot_candidates: dict[str, list[SeriesCandidate]] = field(default_factory=dict)
    # slot_id -> None | score breakdown {component: raw score} of the Top-1
    slot_scores: dict[str, dict | None] = field(default_factory=dict)
    # per-series metadata kept alongside (plane/fluid/fs/uid/...); slot_id -> dict
    slot_metadata: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid_slots(self) -> list[str]:
        return sorted(s for s, u in self.slot_series.items() if u)


def summarize_series(rows: pd.DataFrame, plane: str, fluid_sensitive: int,
                     fat_suppression: int | None) -> dict:
    """Precompute everything the selection components need for one candidate."""
    return {
        "rows": rows,
        "plane": plane,
        "fluid_sensitive": fluid_sensitive,
        "fat_suppression": fat_suppression,
        "n_slices": len(rows),
        "z": _physical_z_if_possible(rows),
        "series_uid": str(rows["series_uid"].iloc[0]),
    }


def _physical_z_if_possible(rows: pd.DataFrame) -> np.ndarray | None:
    iops = rows[IOP_COLS].to_numpy(dtype=float)
    ipps = rows[IPP_COLS].to_numpy(dtype=float)
    if np.isnan(iops).any() or np.isnan(ipps).any():
        return None
    normals = np.stack([slice_normal(iop) for iop in iops])
    med = np.median(normals, axis=0)
    med /= (np.linalg.norm(med) + 1e-12)
    return (ipps * med).sum(axis=1)


def score_candidate(summary: dict, components: list[SelectionComponent]) -> tuple[float, dict]:
    breakdown, total, weight_sum = {}, 0.0, 0.0
    for comp in components:
        fn = COMPONENTS.get(comp.name)
        if fn is None:
            raise KeyError(f"unknown selection component '{comp.name}' "
                           f"(available: {sorted(COMPONENTS)})")
        raw = float(np.clip(fn(summary, comp.params), 0.0, 1.0))
        breakdown[comp.name] = raw
        total += comp.weight * raw
        weight_sum += comp.weight
    score = total / weight_sum if weight_sum > 0 else 0.0
    return score, breakdown


def map_study_to_slots(study_uid: str, series_table_rows: pd.DataFrame,
                       slice_rows_by_series: dict[str, pd.DataFrame],
                       cfg: SlotsConfig) -> SlotAssignment:
    """Assign every series of one study to canonical slots.

    series_table_rows   rows of the competition series table for this study
    slice_rows_by_series  series_uid -> per-slice metadata frame (may be missing)
    """
    assignment = SlotAssignment(study_uid=study_uid)
    for slot_id in cfg.definitions:
        assignment.slot_series[slot_id] = ""
        assignment.slot_candidates[slot_id] = []
        assignment.slot_scores[slot_id] = None

    # Phase 1: candidates per slot = every series whose key matches any SlotKey
    # of the slot. A series may be candidate for several slots.
    candidates: dict[str, list[str]] = {slot: [] for slot in cfg.definitions}
    usable_rows=[]
    for _,row in series_table_rows.iterrows():
        try:
            if str(row['Anatomical_Plane']) not in ('Sagittal','Coronal','Axial'):
                raise ValueError('unknown plane')
            if int(row['Fluid_Sensitive']) not in (0,1) or int(row['Fat_Suppression']) not in (0,1):
                raise ValueError('unknown sequence flags')
            usable_rows.append(row)
        except (ValueError,TypeError):
            assignment.warnings.append(f"invalid series description: {row.get('SeriesInstanceUID','?')} — skipped")
    series_table_rows=pd.DataFrame(usable_rows,columns=series_table_rows.columns).drop_duplicates('SeriesInstanceUID')
    for slot_id, keys in cfg.definitions.items():
        for _, row in series_table_rows.iterrows():
            uid = str(row["SeriesInstanceUID"])
            for k in keys:
                if (str(row["Anatomical_Plane"]) == k.plane
                        and int(row["Fluid_Sensitive"]) == k.fluid_sensitive
                        and (k.fat_suppression is None
                             or int(row["Fat_Suppression"]) == k.fat_suppression)):
                    candidates[slot_id].append(uid)
                    break

    # Phase 2: slots in definitions order (earlier = higher priority) pick the
    # best-scoring candidate NOT already taken by an earlier slot.
    claimed: set[str] = set()
    uid_to_summary: dict[str, tuple[float, str, dict, dict]] = {}

    def score_and_cache(uid: str, slot_id: str):
        if uid not in uid_to_summary:
            rows = slice_rows_by_series.get(uid)
            if rows is None or rows.empty:
                return None
            row0 = series_table_rows[series_table_rows["SeriesInstanceUID"] == uid].iloc[0]
            summary = summarize_series(rows, plane=str(row0["Anatomical_Plane"]),
                                       fluid_sensitive=int(row0["Fluid_Sensitive"]),
                                       fat_suppression=int(row0["Fat_Suppression"]))
            score, breakdown = score_candidate(summary, cfg.selection)
            breakdown["_total"] = score
            uid_to_summary[uid] = (score, uid, summary, breakdown)
        return uid_to_summary[uid]

    for slot_id, cand_uids in candidates.items():
        if not cand_uids:
            continue
        scored = []
        for uid in sorted(cand_uids):  # deterministic iteration order
            try:
                entry = score_and_cache(uid, slot_id)
            except (ValueError,TypeError,FloatingPointError) as exc:
                assignment.warnings.append(f"invalid candidate geometry {uid}: {exc} — skipped")
                continue
            if entry is None:
                assignment.warnings.append(
                    f"slot '{slot_id}': candidate series {uid} has no slice metadata — skipped")
                continue
            score, _, summary, breakdown = entry
            scored.append((score, uid, summary, breakdown))
        free = [(s, u, m, b) for s, u, m, b in scored if u not in claimed]
        if not scored:
            continue
        if not free:
            assignment.warnings.append(
                f"slot '{slot_id}': all {len(scored)} candidate(s) already claimed "
                "by higher-priority slots")
            continue
        # deterministic selection: highest score, SeriesInstanceUID tie-break;
        # keep Top-K ranks (rank 0 = Top-1), all claimed by this slot
        free.sort(key=lambda t: (-t[0], t[1]))
        keep = free[: max(1, cfg.k_series_per_slot)]
        for rank, (score, uid, summary, breakdown) in enumerate(keep):
            claimed.add(uid)
            assignment.slot_candidates[slot_id].append(SeriesCandidate(
                series_uid=uid, score=score, score_breakdown=breakdown,
                metadata={
                    "slot_id": slot_id,
                    "plane": summary["plane"],
                    "fluid_sensitive": summary["fluid_sensitive"],
                    "fat_suppression": summary["fat_suppression"],
                    "series_uid": uid,
                    "n_slices": summary["n_slices"],
                    "selection_score": score,
                    "rank": rank,
                    "n_candidates": len(scored),
                }))
        best_score, best_uid, best_summary, best_breakdown = keep[0]
        assignment.slot_series[slot_id] = best_uid
        assignment.slot_scores[slot_id] = best_breakdown
        assignment.slot_metadata[slot_id] = \
            assignment.slot_candidates[slot_id][0].metadata
        if len(free) > 1:
            losers = ", ".join(f"{u}:{s['n_slices']}sl/{sc:.3f}"
                               for sc, u, s, _ in free[len(keep):])
            assignment.warnings.append(
                f"slot '{slot_id}': {len(free)} candidates, kept {len(keep)} "
                f"(top: {best_uid} @ {best_score:.3f}); rejected: {losers}")
    return assignment
