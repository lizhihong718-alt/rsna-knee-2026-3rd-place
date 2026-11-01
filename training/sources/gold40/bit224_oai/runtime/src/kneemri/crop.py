"""Physical-mm crop + resize (pipeline section 6) and MRI intensity
normalization (pipeline section 7).

Crop: different scanners use different PixelSpacing, so resizing the full
image to a fixed grid changes the physical scale of anatomy. Instead we keep a
fixed physical FOV (crop_fov_mm) around the image center, converted to a pixel
window via PixelSpacing per axis, then resize that window to output_size.

Intensity: MRI has no absolute scale — normalize per SERIES (never across
studies) with percentile clipping to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kneemri.config import PreprocessConfig

# mapping config string -> cv2 interpolation (resolved per-image: AREA when
# downscaling, LINEAR when upscaling, unless overridden explicitly)
_EXPLICIT_INTERP = {
    "area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST, "cubic": cv2.INTER_CUBIC,
}


@dataclass
class CropPlan:
    """Pixel window (float, [x0, x1) x [y0, y1)) to extract from the source
    image, plus bookkeeping for the case where the FOV exceeds the image."""
    x0: float
    y0: float
    x1: float
    y1: float
    source_rows: int
    source_cols: int
    fov_mm: float
    spacing_row: float          # mm per pixel along rows (axis 0)
    spacing_col: float          # mm per pixel along cols (axis 1)
    padded: bool
    pad_rows: tuple[int, int]   # pixels added (top, bottom)
    pad_cols: tuple[int, int]   # pixels added (left, right)


def plan_physical_crop(spacing_row: float, spacing_col: float,
                       rows: int, cols: int, fov_mm: float) -> CropPlan:
    """Compute the centered physical FOV window.

    spacing_* must be > 0; fov_mm is the side length of the square physical
    window in mm. If the window exceeds the image (small FOV acquisition or
    very fine spacing), the plan pads symmetrically so the OUTPUT still
    represents exactly fov_mm of tissue.
    """
    if not np.isfinite([spacing_row,spacing_col]).all() or spacing_row <= 0 or spacing_col <= 0:
        raise ValueError(f"PixelSpacing must be positive, got ({spacing_row}, {spacing_col})")
    if not np.isfinite(fov_mm) or fov_mm <= 0:
        raise ValueError(f"crop_fov_mm must be positive, got {fov_mm}")
    crop_h_mm2px = fov_mm / spacing_row   # window height in pixels (rows axis)
    crop_w_mm2px = fov_mm / spacing_col   # window width in pixels (cols axis)

    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0   # geometric image center
    y0, y1 = cy - crop_h_mm2px / 2.0, cy + crop_h_mm2px / 2.0
    x0, x1 = cx - crop_w_mm2px / 2.0, cx + crop_w_mm2px / 2.0

    pad_top, pad_bottom = max(0, -y0), max(0, y1 - (rows - 1))
    pad_left, pad_right = max(0, -x0), max(0, x1 - (cols - 1))
    padded = any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right))
    return CropPlan(x0, y0, x1, y1, rows, cols, fov_mm, spacing_row, spacing_col,
                    padded, (int(np.ceil(pad_top)), int(np.ceil(pad_bottom))),
                    (int(np.ceil(pad_left)), int(np.ceil(pad_right))))


def apply_crop_resize(image: np.ndarray, plan: CropPlan, output_size: int,
                      interpolation: str = "auto",
                      return_mask: bool = False):
    """Extract plan's physical window and resize to (output_size, output_size).

    image: (H, W) float32. Padding (when the FOV exceeds the image) is filled
    with 0 after extraction. With return_mask=True also returns a (output,
    output) uint8 mask: 1 where the output pixel came from real acquired
    pixels, 0 where it is padding.
    """
    if image.shape[:2] != (plan.source_rows, plan.source_cols):
        raise ValueError(f"image shape {image.shape} does not match crop plan "
                         f"{(plan.source_rows, plan.source_cols)}")
    src = image
    if plan.padded:
        src = np.pad(image, pad_width=((plan.pad_rows[0], plan.pad_rows[1]),
                                       (plan.pad_cols[0], plan.pad_cols[1])),
                     mode="constant", constant_values=0)
        x0 = plan.x0 + plan.pad_cols[0]
        y0 = plan.y0 + plan.pad_rows[0]
    else:
        x0, y0 = plan.x0, plan.y0
    x1, y1 = x0 + (plan.x1 - plan.x0), y0 + (plan.y1 - plan.y0)

    # cv2.resize needs an integer ROI: expand the float window outward to the
    # covering pixel rect, then resize that rect (sub-pixel error <= 1 px at
    # source resolution, i.e. <= spacing mm; output resample averages it out)
    rx0, ry0 = int(np.floor(x0)), int(np.floor(y0))
    rx1, ry1 = int(np.ceil(x1)), int(np.ceil(y1))
    rx0, ry0 = max(rx0, 0), max(ry0, 0)
    rx1, ry1 = min(rx1, src.shape[1]), min(ry1, src.shape[0])
    if rx1 - rx0 < 2 or ry1 - ry0 < 2:
        raise ValueError(f"degenerate crop window {((rx0, ry0), (rx1, ry1))}")
    roi = src[ry0:ry1, rx0:rx1]

    interp = _resolve_interpolation(interpolation, roi, output_size)
    out = cv2.resize(roi, (output_size, output_size), interpolation=interp)
    if not return_mask:
        return np.asarray(out, dtype=np.float32)

    # real-pixel region of `src` in padded coordinates
    real_y0, real_x0 = plan.pad_rows[0], plan.pad_cols[0]
    real_y1, real_x1 = real_y0 + plan.source_rows, real_x0 + plan.source_cols
    oy0 = int(round((max(ry0, real_y0) - ry0) * output_size / (ry1 - ry0)))
    oy1 = int(round((min(ry1, real_y1) - ry0) * output_size / (ry1 - ry0)))
    ox0 = int(round((max(rx0, real_x0) - rx0) * output_size / (rx1 - rx0)))
    ox1 = int(round((min(rx1, real_x1) - rx0) * output_size / (rx1 - rx0)))
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    mask[oy0:oy1, ox0:ox1] = 1
    return np.asarray(out, dtype=np.float32), mask


def _resolve_interpolation(mode: str, roi: np.ndarray, output_size: int) -> int:
    if mode == "auto":
        return cv2.INTER_AREA if roi.shape[0] > output_size else cv2.INTER_LINEAR
    if mode not in _EXPLICIT_INTERP:
        raise ValueError(f"unknown interpolation '{mode}' (options: auto|{_EXPLICIT_INTERP})")
    return _EXPLICIT_INTERP[mode]


# ------------------------------------------------------------------ intensity

def normalize_intensity(volume: np.ndarray, p_low: float, p_high: float,
                        over: str = "foreground") -> tuple[np.ndarray, dict]:
    """Percentile-clip a per-series volume to [0, 1].

    volume: (n_slices, H, W) float32, after rescale slope/intercept.
    over='foreground' computes percentiles on nonzero voxels (MRI background
    is air, ~half the pixels); 'all' uses every pixel.
    Returns (normalized volume, stats) — stats records lo/hi in raw units.
    """
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(f"require 0 <= p_low < p_high <= 100, got ({p_low}, {p_high})")
    if over not in ('all','foreground'):
        raise ValueError('intensity_over must be all or foreground')
    if not np.isfinite(volume).all():
        raise ValueError('non-finite decoded pixels')
    vals = volume if over == "all" else volume[volume > 0]
    stats = {}
    if vals.size == 0:
        stats.update(lo=0.0, hi=0.0, empty=True)
        return np.zeros_like(volume, dtype=np.float32), stats
    # Foreground selection already produced a private copy, so let numpy
    # partition it in place: same selection algorithm, same order statistics,
    # one fewer full-size allocation. 'all' aliases the caller's volume.
    lo, hi = np.percentile(vals, [p_low, p_high], overwrite_input=vals is not volume)
    if hi <= lo:
        stats.update(lo=float(lo), hi=float(hi), degenerate=True)
        return np.zeros_like(volume, dtype=np.float32), stats
    out = np.clip(volume, lo, hi)
    out = (out - lo) / (hi - lo)
    stats.update(lo=float(lo), hi=float(hi))
    return out.astype(np.float32), stats
