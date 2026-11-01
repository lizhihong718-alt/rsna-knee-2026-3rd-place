"""Laterality + in-plane orientation normalization (pipeline section 5).

Goal: left and right knees land in a consistent anatomy orientation.

The key domain fact: which image axis carries the medial-lateral (M-L)
direction depends on the plane —

    sagittal   M-L = slice normal (the STACK direction)   in-plane = A-P, S-I
    coronal    M-L = in-plane horizontal                  stack = A-P
    axial      M-L = in-plane horizontal                  stack = S-I

So normalization has two mechanisms, each geometry-derivable (never pixel
guesses):

  in-plane horizontal flip — when the horizontal image axis (IOP[0:3], the
    direction of increasing column index) carries M-L, flip it to point
    toward the LATERAL side: lateral_dir = +X LPS if Laterality=='L' else -X.
    If the horizontal axis instead carries A-P or S-I (sagittal, or unusual
    coronal storage order), canonicalize it to the + direction of its
    dominant patient axis (posterior / head).
  stack direction — geometry orders slices ascending along the series median
    normal. For sagittal series that normal is ±X, so ascending order runs
    toward patient-Left for one knee and patient-Right for the other; we
    reverse when it runs toward MEDIAL, i.e. flip so the stack ascends toward
    LATERAL. Non-sagittal stacks (±Y / ±Z) are laterality-independent and
    stay ascending along their fixed +axis.

Uncertainty policy: missing/conflicting laterality or an ambiguous axis
(|alignment| below threshold) -> do NOT flip, record a warning. An uncertain
flip is worse than no flip. Every decision (including no-ops) is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kneemri.config import LateralityConfig

# DICOM LPS patient coordinates: +X = patient Left, +Y = Posterior, +Z = Head.
PATIENT_LEFT = np.array([1.0, 0.0, 0.0])
AXIS_NAMES = ("patient_left(+X)", "posterior(+Y)", "head(+Z)")


@dataclass
class OrientationDecision:
    """One series' canonicalization decision (recorded to the report)."""
    horizontal_flip: bool
    reverse_stack: bool
    laterality: str            # 'L' | 'R' | ''
    horizontal_axis: str       # which patient axis the horizontal axis carries
    stack_axis: str
    reason: str
    warnings: list[str]


def normalize_laterality_tag(value: str | None) -> str:
    """Normalize DICOM Laterality (0020,0060) to 'L' | 'R' | '' (unknown).

    The corpus contains long-form values ('RIGHT'/'LEFT'), 'B' (bilateral —
    meaningless for single-knee orientation, treated as unknown), and blanks.
    """
    if value is None:
        return ""
    v = str(value).strip().upper()
    v = {"RIGHT": "R", "LEFT": "L"}.get(v, v)
    return v if v in ("L", "R") else ""


def series_laterality(slice_laterality: list[str | None]) -> str:
    """Majority laterality over a series' slices; '' if absent/conflicting."""
    vals = [normalize_laterality_tag(v) for v in slice_laterality]
    vals = [v for v in vals if v]
    if not vals:
        return ""
    counts = {v: vals.count(v) for v in set(vals)}
    if len(counts) > 1:
        # conflicting tags inside one series: take majority only on a strict
        # margin, otherwise refuse to decide
        top2 = sorted(counts.values(), reverse=True)
        if top2[0] == top2[1]:
            return ""
    return max(counts, key=counts.get)


def _dominant_axis(vec: np.ndarray) -> tuple[int, float]:
    """(axis index, signed value) of the largest-|component| of a unit vector."""
    k = int(np.argmax(np.abs(vec)))
    return k, float(vec[k])


def plan_orientation(median_iop: np.ndarray, plane: str, laterality: str,
                     cfg: LateralityConfig) -> OrientationDecision:
    """Decide in-plane flip + stack reversal for one series.

    median_iop : series-median ImageOrientationPatient (6,) — row cosines
                 iop[:3] are the horizontal image axis, the cross product of
                 (row, col) is the slice normal (stack direction after the
                 geometry stage's ascending-z ordering).
    plane      : 'Sagittal' | 'Coronal' | 'Axial' (competition series table)
    laterality : 'L' | 'R' | ''
    """
    iop = np.asarray(median_iop, dtype=float)
    if iop.shape != (6,):
        raise ValueError(f"IOP must have 6 components, got shape {iop.shape}")
    if plane not in ("Sagittal", "Coronal", "Axial"):
        raise ValueError(f"unknown plane '{plane}'")
    if cfg.target != "lateral":
        raise ValueError(f"unknown laterality target '{cfg.target}'")
    if not np.isfinite(iop).all() or np.linalg.norm(iop[:3]) < 1e-6 \
            or np.linalg.norm(iop[3:]) < 1e-6:
        # no usable geometry (e.g. all-metadata-missing fallback path):
        # orientation policy requires physics — refuse both flips, warn
        return OrientationDecision(
            False, False, laterality, "unavailable", "unavailable",
            "geometry unavailable — orientation not normalized",
            ["median IOP is NaN/zero — no orientation normalization applied"])

    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    normal /= (np.linalg.norm(normal) + 1e-12)
    warnings: list[str] = []

    # geometry-derived plane vs the competition table's label — the table can
    # mislabel oblique acquisitions; the slice normal cannot. Policy below is
    # driven by the geometry (stack_k / h_k branches), so a mismatch only
    # degrades to a warning.
    derived_plane = {0: "Sagittal", 1: "Coronal", 2: "Axial"}[_dominant_axis(normal)[0]]
    if derived_plane != plane:
        warnings.append(
            f"plane mismatch: series table says '{plane}' but slice geometry "
            f"derives '{derived_plane}' (median normal) — using geometry")

    # ---------------- stack direction -------------------------------------
    stack_k, stack_sign = _dominant_axis(normal)
    reverse_stack = False  # python bools on purpose (no np.bool_ leakage)
    if stack_k == 0:  # stack runs along patient Left-Right -> M-L: laterality-aware
        if laterality in ("L", "R"):
            lateral_sign = +1.0 if laterality == "L" else -1.0
            if abs(normal[0]) < cfg.min_axis_alignment:
                warnings.append(
                    f"stack M-L direction ambiguous (|x|={abs(normal[0]):.2f}) — "
                    "stack left ascending +patient_left")
            else:
                reverse_stack = bool(normal[0] * lateral_sign < 0)  # runs toward medial
        else:
            warnings.append(
                "sagittal stack is the medial-lateral axis but laterality is "
                "missing/conflicting — stack left ascending +patient_left "
                "(left/right knees may be mirror images)")
    else:
        # non-sagittal stacks (±Y posterior / ±Z head) are laterality-
        # independent: canonicalize to ascending the fixed +axis
        reverse_stack = bool(stack_sign < 0)

    # ---------------- in-plane horizontal axis ----------------------------
    h_k, h_sign = _dominant_axis(row_dir)
    horizontal_flip = False
    reason_parts = []
    if abs(row_dir[h_k]) < cfg.min_axis_alignment:
        warnings.append(
            f"in-plane horizontal axis ambiguous (dominant |component|="
            f"{abs(row_dir[h_k]):.2f} < {cfg.min_axis_alignment}) — no in-plane flip")
        h_k = -1
        horizontal_flip = False
    elif stack_k == 0:
        # sagittal: horizontal axis is A-P/S-I -> geometry-only canonical +axis
        horizontal_flip = bool(h_sign < 0)
        reason_parts.append(
            f"sagittal in-plane axis canonicalized to +{AXIS_NAMES[h_k]}")
    elif h_k == 0:
        # axial/coronal with M-L horizontal axis -> flip toward LATERAL
        if laterality in ("L", "R"):
            lateral_sign = +1.0 if laterality == "L" else -1.0
            horizontal_flip = bool(row_dir[0] * lateral_sign < 0)
            reason_parts.append(
                f"horizontal axis carries medial-lateral; "
                f"{'flip' if horizontal_flip else 'no flip'} to point lateral")
        else:
            horizontal_flip = False
            warnings.append(
                "horizontal axis carries medial-lateral but laterality is "
                "missing/conflicting — no flip (left/right knees may be "
                "mirror images)")
    else:
        # axial/coronal whose horizontal axis carries A-P/S-I -> geometry-only
        horizontal_flip = bool(h_sign < 0)
        reason_parts.append(
            f"in-plane axis canonicalized to +{AXIS_NAMES[h_k]}")

    reason = "; ".join(reason_parts) if reason_parts else "no in-plane change"
    return OrientationDecision(horizontal_flip, reverse_stack, laterality,
                               AXIS_NAMES[h_k] if h_k >= 0 else "ambiguous",
                               AXIS_NAMES[stack_k], reason, warnings)


def apply_horizontal_flip(image: np.ndarray, flip: bool) -> np.ndarray:
    """Flip the last (column) axis. Works on (H, W), (n, H, W), (..., H, W, C)."""
    return np.ascontiguousarray(image[..., ::-1]) if flip else image


def apply_stack_reversal(volume: np.ndarray, reverse: bool) -> np.ndarray:
    """Reverse the slice axis (axis 0) of a physically ordered stack."""
    return np.ascontiguousarray(volume[::-1]) if reverse else volume
