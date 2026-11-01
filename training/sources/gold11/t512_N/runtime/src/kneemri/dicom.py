"""DICOM pixel decoding.

Headers are read elsewhere (scripts/scan_metadata.py builds the metadata
cache); this module only turns a .dcm file into a (H, W) float32 image in
HU-like modality units (rescale slope/intercept applied, MONOCHROME1
inverted). Known-corrupted studies (truncated pixel buffers) raise a specific
exception so the runner can blacklist and continue instead of crashing.
"""
from __future__ import annotations

import numpy as np
import pydicom


class CorruptedPixelDataError(RuntimeError):
    """Pixel buffer does not match the declared geometry (e.g. the two
    train studies whose pixel payload is exactly half the declared size)."""


def decode_pixel_array(path: str) -> np.ndarray:
    """(H, W) float32 image; applies rescale slope/intercept (defaults 1/0 —
    over half the corpus omits these tags) and inverts MONOCHROME1."""
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
    except Exception as exc:  # truncated buffer / EOF inside pixel data
        raise CorruptedPixelDataError(f"{path}: {exc}") from exc
    return modality_pixels(ds, arr, path)


def modality_pixels(ds, arr: np.ndarray, path: str = "") -> np.ndarray:
    """Apply the reference shape checks, modality transform and inversion."""

    expected = int(ds.Rows) * int(ds.Columns)
    if arr.ndim == 3 and arr.shape[0] == 1:   # degenerate single-frame stack
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] * arr.shape[1] != expected:
        raise CorruptedPixelDataError(
            f"{path}: decoded shape {arr.shape} incompatible with "
            f"({ds.Rows}, {ds.Columns})")

    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    # astype already produced a private float32 array, so the modality transform
    # runs in place. Identity slope/intercept are skipped: over half the corpus
    # omits both tags, and x * 1.0 + 0.0 returns x for every integer-derived
    # pixel value.
    if slope != 1.0:
        arr *= slope
    if intercept:
        arr += intercept
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        np.subtract(arr.max(), arr, out=arr)
    return np.ascontiguousarray(arr, dtype=np.float32)
