"""Physical slice geometry (pipeline section 3).

Slice order is derived from DICOM physical coordinates — NEVER from file
names, and InstanceNumber is only ever a last-resort *logged* fallback:

    normal = cross(row_dir, col_dir)        # row_dir = IOP[0:3], col_dir = IOP[3:6]
    z      = dot(ImagePositionPatient, normal)

Series are ordered ascending by z. Direction canonicalization (which way the
stack should run for cross-study consistency) is orientation *policy* and
lives in kneemri.laterality.plan_orientation — this module is physics only.

Every fallback and every anomaly emits a warning string carried on the returned
SeriesOrdering — nothing is silently accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kneemri.config import GeometryConfig

IOP_COLS = ["iop_x", "iop_y", "iop_z", "iop_cx", "iop_cy", "iop_cz"]
IPP_COLS = ["ipp_x", "ipp_y", "ipp_z"]
# DICOM LPS patient coordinate system: +X = patient Left, +Y = Posterior, +Z = Head.


@dataclass
class SeriesOrdering:
    """Result of ordering one series' slices physically.

    order     indices into the input frame/table giving physical (ascending z) order
    z         physical z per input row, in input order (NaN where unverifiable)
    source    which key the order was built from
    normal    series-median unit slice normal, or None when no geometry exists
    iop       series-median ImageOrientationPatient (6,), triplets renormalized
              (fed to orientation policy), or None when no geometry exists
    warnings  every anomaly encountered (fallbacks, duplicates, obliquity)
    """

    order: np.ndarray
    z: np.ndarray
    source: str
    normal: np.ndarray | None = None
    iop: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    def ordered(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.iloc[self.order].reset_index(drop=True)


def slice_normal(iop: np.ndarray) -> np.ndarray:
    """Unit slice normal from a (6,) ImageOrientationPatient vector."""
    iop = np.asarray(iop, dtype=float)
    if iop.shape != (6,):
        raise ValueError(f"IOP must have 6 components, got shape {iop.shape}")
    row_dir, col_dir = iop[:3], iop[3:]
    if not (np.isfinite(iop).all() and np.linalg.norm(row_dir) > 0 and np.linalg.norm(col_dir) > 0):
        raise ValueError("IOP contains non-finite or zero direction cosines")
    normal = np.cross(row_dir, col_dir)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("degenerate IOP: row and column directions are parallel")
    return normal / norm


def physical_z(ipp: np.ndarray, normal: np.ndarray) -> float:
    """Projection of a slice position onto the slice normal."""
    return float(np.dot(np.asarray(ipp, dtype=float), normal))


def _median_iop(iops: np.ndarray) -> np.ndarray:
    """Component-wise median IOP with each direction triplet renormalized."""
    med = np.median(iops, axis=0)
    for a, b in ((0, 3), (3, 6)):
        n = np.linalg.norm(med[a:b])
        if n > 1e-9:
            med[a:b] /= n
    return med


def _median_unit_normal(iops: np.ndarray) -> np.ndarray:
    """Component-wise median of per-slice normals, renormalized (robust to a
    few deviating slices; cancellation is theoretically impossible here since
    all slices of one acquisition share an orientation within a few degrees)."""
    med = np.median(iops, axis=0)
    norm = np.linalg.norm(med)
    if norm < 1e-6:  # pragma: no cover - would need ~180-degree in-series flips
        med = iops[len(iops) // 2]
        norm = np.linalg.norm(med)
    return med / norm


def _angular_deviation_deg(iops: np.ndarray, ref: np.ndarray) -> np.ndarray:
    dots = np.clip(iops @ ref, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _has_geometry(rows: pd.DataFrame) -> bool:
    return bool(len(rows)) and not rows[IOP_COLS + IPP_COLS].isna().any().any()


def _warn_fallback(rows: pd.DataFrame, key: str,
                   warnings: list[str]) -> np.ndarray | None:
    values = rows[key].to_numpy(dtype=float) if key in rows.columns else None
    if values is None or np.isnan(values).any() or len(np.unique(values)) < len(values):
        return None
    warnings.append(
        f"geometry fallback: sorted by '{key}'; physical IOP/IPP ordering "
        f"unavailable for this series — order is NOT verified against slice geometry")
    return np.argsort(values, kind="stable")


def sort_series_geometry(rows: pd.DataFrame, cfg: GeometryConfig,
                         label: str = "") -> SeriesOrdering:
    """Order one series' slices by physical position.

    rows : per-slice metadata (columns from IOP_COLS/IPP_COLS plus
           slice_location / instance_number / sop_uid); one row per slice.

    Corpus-measured hazard this function guards against: SliceLocation rank
    order runs EXACTLY opposite to physical order in ~59% of real series
    (per-vendor sign convention, correlation -1.0 when it disagrees), so a
    fallback to slice_location preserves adjacency but its DIRECTION is
    unknowable without geometry — hence the loud warnings and the normal
    (when computable) being surfaced for downstream direction policy.
    """
    if rows.empty:
        raise ValueError(f"cannot sort empty series {label}")
    warnings: list[str] = []
    n = len(rows)
    geo_mask = ~rows[IOP_COLS + IPP_COLS].isna().any(axis=1).to_numpy()

    if geo_mask.all():
        iops = rows[IOP_COLS].to_numpy(dtype=float)
        ipps = rows[IPP_COLS].to_numpy(dtype=float)
        normals = np.stack([slice_normal(iop) for iop in iops])
        med_normal = _median_unit_normal(normals)
        dev = _angular_deviation_deg(normals, med_normal)
        n_off = int((dev > cfg.max_normal_deviation_deg).sum())
        if n_off:
            warnings.append(
                f"{n_off}/{n} slices deviate more than {cfg.max_normal_deviation_deg} deg "
                f"from the series median normal (max {dev.max():.2f} deg) — oblique or "
                "inconsistent acquisition")
        z = (ipps * med_normal).sum(axis=1)
        order = np.lexsort((rows["sop_uid"].to_numpy(), z))  # z asc, uid tie-break
        dz = np.diff(np.sort(z))
        n_dup = int((dz < cfg.duplicate_z_tol_mm).sum())
        if n_dup:
            warnings.append(
                f"{n_dup} near-duplicate slice positions (|dz| < {cfg.duplicate_z_tol_mm} mm) "
                "— repeated acquisitions at one location")
        return SeriesOrdering(order, z, "ipp_dot_normal", med_normal,
                              _median_iop(iops), warnings)

    if geo_mask.any():
        # partial geometry: the fallback keeps ADJACENCY (fallback keys are
        # monotone with physical position in this corpus) but the direction is
        # unverifiable; still surface the median normal of the geometric
        # slices so downstream direction policy can act.
        geo_iops = rows.loc[geo_mask, IOP_COLS].to_numpy(dtype=float)
        med_normal = _median_unit_normal(np.stack([slice_normal(iop) for iop in geo_iops]))
        med_iop = _median_iop(geo_iops)
        warnings.append(
            "partial geometry: missing ImageOrientationPatient/ImagePositionPatient "
            f"on {int((~geo_mask).sum())}/{n} slices — order from configured "
            "fallback preserves adjacency but physical DIRECTION is unverified")
    else:
        med_normal, med_iop = None, None
        warnings.append(
            "geometry unavailable on all slices — engaging configured fallbacks; "
            "physical order/direction NOT verified")
    z = np.full(n, np.nan)
    for fb in cfg.fallbacks:
        order = _warn_fallback(rows, fb, warnings)
        if order is not None:
            # if both fallback keys exist and disagree in rank order, direction
            # is fundamentally ambiguous — say so rather than pick silently
            if {"slice_location", "instance_number"} <= set(rows.columns):
                a = rows["slice_location"].to_numpy(dtype=float)
                b = rows["instance_number"].to_numpy(dtype=float)
                if np.isfinite(a).all() and np.isfinite(b).all():
                    rho = np.corrcoef(a, b)[0, 1]
                    if rho < 0:
                        warnings.append(
                            "slice_location and instance_number ranks disagree "
                            f"(corr={rho:.2f}) — fallback order direction ambiguous")
            return SeriesOrdering(order, z, fb, med_normal, med_iop, warnings)
    raise ValueError(
        f"series {label}: physical geometry missing and no usable fallback "
        f"({cfg.fallbacks}) — slice order undetermined; study must be skipped")
