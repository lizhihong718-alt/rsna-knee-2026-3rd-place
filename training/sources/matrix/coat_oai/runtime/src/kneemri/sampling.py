"""Fixed slice sampling + adjacent-slice 2.5D input (pipeline sections 7-8).

Acquisitions differ in slice count / spacing / coverage, so a fixed grid of
`slices_per_slot` slice positions is sampled from the physically ordered
series. Two modes:

  uniform_index  evenly spaced indices (count-normalized coverage)
  uniform_mm     evenly spaced *physical* z positions, snapped to the nearest
                 real slice (spacing-normalized coverage)

Each sampled position becomes one 2.5D token: a stack of the slice plus its
physical neighbors (neighbor_offsets), with out-of-range neighbors clamped to
the edge slice (pad_mode='edge').
"""
from __future__ import annotations

import numpy as np

from kneemri.config import SamplingConfig


def sample_positions(n_slices: int, cfg: SamplingConfig,
                     z: np.ndarray | None = None) -> np.ndarray:
    """Indices into the physically ordered series, shape (slices_per_slot,).

    Returned indices are non-decreasing. When the series has fewer distinct
    slices than requested, indices repeat (recorded downstream as
    n_distinct_sampled).
    """
    if n_slices == 0:
        raise ValueError("cannot sample from an empty series")
    if cfg.mode == "uniform_index":
        if n_slices == 1:
            return np.zeros(cfg.slices_per_slot, dtype=int)
        pos = np.linspace(0, n_slices - 1, cfg.slices_per_slot)
        return np.round(pos).astype(int)
    if cfg.mode == "uniform_mm":
        if z is None or len(z) != n_slices:
            raise ValueError("uniform_mm sampling requires physical z per slice")
        z = np.asarray(z, dtype=float)
        if not np.isfinite(z).all():
            # fallback paths (or dropped slices) can leave z unverifiable —
            # refuse to collapse silently; caller decides the fallback
            raise ValueError("uniform_mm sampling requires finite physical z "
                             "(got NaN — order came from a metadata fallback)")
        if n_slices == 1:
            return np.zeros(cfg.slices_per_slot, dtype=int)
        order = np.argsort(z)
        zs = z[order]
        # z is normally already ascending (geometry stage); guard regardless
        grid = np.linspace(zs[0], zs[-1], cfg.slices_per_slot)
        hi = np.searchsorted(zs, grid, side="left").clip(1, n_slices - 1)
        lo = hi - 1
        choose_hi = (zs[hi] - grid) < (grid - zs[lo])   # true nearest neighbor
        return order[np.where(choose_hi, hi, lo)]
    raise ValueError(f"unknown sampling mode '{cfg.mode}' (uniform_index|uniform_mm)")


def neighbor_stack(volume: np.ndarray, positions: np.ndarray,
                   offsets: tuple[int, ...] = (-1, 0, 1),
                   pad_mode: str = "edge") -> np.ndarray:
    """Build 2.5D input: (n_positions, len(offsets), H, W).

    volume: (n_slices, H, W) in physical order. Neighbor index i+o clamps to
    [0, n_slices-1] (edge replicate) — the physically nearest available slice.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be (n, H, W), got {volume.shape}")
    if pad_mode != "edge":
        raise ValueError(f"unsupported pad_mode '{pad_mode}'")
    n = volume.shape[0]
    pos = np.asarray(positions, dtype=int)
    if (pos < 0).any() or (pos >= n).any():
        raise ValueError(f"sampled positions out of range for volume of {n} slices")
    neigh = np.clip(pos[None, :] + np.asarray(offsets, dtype=int)[:, None], 0, n - 1)
    # -> (n_positions, len(offsets), H, W)
    return np.transpose(volume[neigh], (1, 0, 2, 3))
