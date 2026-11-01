"""Native-resolution windows: one resample from raw DICOM pixels.

The legacy corpus (``coatnet_original384_crop140``) resamples twice: the full
field of view (~160-170 mm) to 384 px, then a 140 mm centre crop of that grid
back up to 384 px. A stored pixel therefore carries ~0.42-0.44 mm of tissue,
coarser than the median acquisition (0.3125 mm). Upsampling that cache cannot
recover the lost detail.

Here the composed legacy crop rectangle is mapped back to raw pixel
coordinates (boundary rounding <= half a source pixel) and resampled exactly
once to the target resolution. Window selection, source slices, orientation,
the physical 140 mm field and the frozen per-series lo/hi are unchanged, so
only the in-plane sampling differs from the legacy corpus. At 512 px a series
keeps every acquired pixel when its spacing is >= 140/512 = 0.273 mm.

Interpolation follows the project's ``auto`` rule (kneemri.crop): INTER_AREA
when the raw rectangle is larger than the output, INTER_LINEAR otherwise.
``interpolation='area'`` reproduces the I1 ``coatnet_dicom384_crop140_uint8``
cache bit for bit at 384 px (tests guard this).
"""
from __future__ import annotations

import cv2
import numpy as np

from kneemri.crop import _resolve_interpolation, plan_physical_crop

# The crop rectangle is defined on the legacy grid; these are not tunables.
LEGACY_GRID = 384
LEGACY_CROP_MM = 140.0


def normalize_frozen(frame, lo, hi):
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    values = ((np.clip(frame, lo, hi) - lo) / (hi - lo)).astype(np.float32)
    # Float32 arithmetic can overshoot 1 by one ULP at the frozen upper bound.
    return np.clip(values, 0, 1)


def composed_bounds(shape, spacing, fov, output_spacing):
    """Map the exact legacy crop extent back to raw pixels, without resampling.

    shape: raw (H, W) of the oriented series; spacing: raw (row, col) mm/px;
    fov: legacy full-field side in mm; output_spacing: legacy grid mm/px.
    Returns ([y0, y1, x0, x1], max rounding error in source px, full plan).
    """
    plan = plan_physical_crop(*spacing, *shape, fov)
    full_start, full_extent = [], []
    for start, end, pad, size in [(plan.y0, plan.y1, plan.pad_rows, shape[0]),
                                  (plan.x0, plan.x1, plan.pad_cols, shape[1])]:
        low = max(int(np.floor(start + pad[0])), 0)
        high = min(int(np.ceil(end + pad[0])), size + sum(pad))
        full_start.append(low - pad[0])
        full_extent.append(high - low)
    floating = []
    for start, extent, step in zip(full_start, full_extent, output_spacing):
        target = max(1, int(round(LEGACY_CROP_MM / step)))
        pad = max(0, target - LEGACY_GRID)
        crop_start = (LEGACY_GRID + pad - target) // 2 - pad // 2
        floating.extend([start + crop_start * extent / LEGACY_GRID,
                         start + (crop_start + target) * extent / LEGACY_GRID])
    bounds = np.rint(floating).astype(int)
    error = float(np.max(np.abs(bounds - floating)))
    if error > .50000001:
        raise ValueError('composed crop rounding exceeded half a source pixel')
    return bounds.tolist(), error, plan


def crop_once(frame, bounds, resolution, interpolation='auto'):
    """Zero-padded raw rectangle -> one cv2.resize to [resolution, resolution]."""
    y0, y1, x0, x1 = bounds
    if min(y1 - y0, x1 - x0) < 2 or max(y1 - y0, x1 - x0) > 4 * max(frame.shape):
        raise ValueError('implausible composed raw crop')
    roi = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    sy0, sy1 = max(y0, 0), min(y1, frame.shape[0])
    sx0, sx1 = max(x0, 0), min(x1, frame.shape[1])
    if sy1 > sy0 and sx1 > sx0:
        roi[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    flag = _resolve_interpolation(interpolation, roi, resolution)
    return cv2.resize(roi, (resolution, resolution), interpolation=flag)


def native_series_windows(raw, neighbours, spacing, fov, output_spacing, lo, hi,
                          resolution, interpolation='auto'):
    """Oriented raw stack [N,H,W] -> uint8 windows [count,3,R,R].

    ``neighbours`` [count,3] indexes the stored (oriented) slices exactly as
    the legacy window builder chose them; each distinct slice is resampled once.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError('raw series must be [slice, H, W]')
    neighbours = np.asarray(neighbours, dtype=np.int64)
    bounds, error, plan = composed_bounds(raw.shape[-2:], spacing, fov, output_spacing)
    unique, inverse = np.unique(neighbours, return_inverse=True)
    frames = np.empty((len(unique), resolution, resolution), dtype=np.uint8)
    for out, index in zip(frames, unique):
        values = normalize_frozen(crop_once(raw[index], bounds, resolution, interpolation), lo, hi)
        out[:] = np.clip(values * 255 + .5, 0, 255).astype(np.uint8)
    windows = frames[inverse.reshape(-1)].reshape(len(neighbours), 3, resolution, resolution)
    return windows, dict(bounds=bounds, rounding_error=error, plan=plan,
                         raw_extent=[bounds[1] - bounds[0], bounds[3] - bounds[2]])
