"""Traceable window candidates for controlled CoAtNet input ablations.

This module never changes the historical 64-slice image cache. ``legacy`` exposes
the historical concatenated-stack candidates, including gaps and series borders;
``same_series`` requires all three cached images to be valid and from one series.
Neither definition claims that sampled neighbours are original physical neighbours.
All selection strategies cap the budget at the candidate count: oversampling must
be an explicit compatibility decision in the caller, never an implicit side effect.
"""
from __future__ import annotations

import numpy as np
import torch


NORMALIZATION_STATS = {
    # The published Raptor training recipe feeds uint8 images scaled only to
    # [0, 1]. Keep that contract explicit instead of overloading a missing
    # normalization value.
    "unit": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "timm_half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def normalization_tensors(name):
    """Return CHW tensors for one explicit pretrained-backbone contract."""
    if name not in NORMALIZATION_STATS:
        raise ValueError(f"unsupported CoAtNet input normalization: {name}")
    mean, std = NORMALIZATION_STATS[name]
    return (torch.tensor(mean, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(std, dtype=torch.float32).view(3, 1, 1))


def candidate_windows(mask, series_ids, original_indices=None, mode="legacy", deduplicate=False):
    """Return candidate cached-slice index triples, shape ``[N, 3]``.

    ``series_ids`` are study-local nonnegative integer identities; -1 means absent.
    In legacy mode the support is exactly the reference evaluation lo..hi support,
    with its first-three-images fallback when fewer than three images are valid.
    Safe mode has no all-empty or short-series fallback. Depth < 3 yields no triple.
    Deduplication compares ordered (series identity, original sorted slice index)
    triples. Unknown provenance uses cached index identity to avoid collapsing
    unrelated padding. It removes duplicate triples, not repeated channels inside
    a triple; that independent design choice remains visible in provenance.
    """
    mask = np.asarray(mask)
    series_ids = np.asarray(series_ids)
    if mask.ndim != 1 or series_ids.shape != mask.shape:
        raise ValueError("mask and series_ids must be one-dimensional with equal length")
    if mode not in {"legacy", "no_cross_series", "same_series"}:
        raise ValueError(f"unknown window mode: {mode}")
    if original_indices is not None:
        original_indices = np.asarray(original_indices)
        if original_indices.shape != mask.shape:
            raise ValueError("original_indices shape must match mask")
    if deduplicate and original_indices is None:
        raise ValueError("deduplication requires original slice provenance")
    depth = len(mask)
    if depth < 3:
        return np.empty((0, 3), dtype=np.int64)
    if mode in {"legacy", "no_cross_series"}:
        valid = np.flatnonzero(mask > 0)
        if len(valid) < 3:
            valid = np.arange(min(3, depth))
        lo, hi = int(valid.min()), int(valid.max())
        centers = np.arange(lo + 1, hi, dtype=np.int64)
        if not len(centers):
            centers = np.array([max(1, min((lo + hi) // 2, depth - 2))], dtype=np.int64)
    else:
        centers = np.arange(1, depth - 1, dtype=np.int64)
    windows = centers[:, None] + np.array([-1, 0, 1], dtype=np.int64)
    if mode == "no_cross_series":
        identities = series_ids[windows]
        # Isolate UID crossings. Missing-image rejection is a separate contrast.
        crossing = np.zeros(len(windows), dtype=bool)
        for left, right in ((0, 1), (0, 2), (1, 2)):
            crossing |= ((identities[:, left] >= 0) & (identities[:, right] >= 0)
                         & (identities[:, left] != identities[:, right]))
        windows = windows[~crossing]
    if mode == "same_series":
        identities = series_ids[windows]
        keep = (mask[windows] > 0).all(1)
        keep &= (identities >= 0).all(1) & (identities == identities[:, :1]).all(1)
        windows = windows[keep]
    if deduplicate:
        seen = set()
        retained = []
        for window in windows:
            key = tuple((int(series_ids[i]), int(original_indices[i]))
                        if series_ids[i] >= 0 and original_indices[i] >= 0
                        else (-1, int(i)) for i in window)
            if key not in seen:
                seen.add(key)
                retained.append(window)
        windows = np.asarray(retained, dtype=np.int64).reshape(-1, 3)
    return windows


def _spread(indices, count):
    if not count:
        return np.empty(0, dtype=np.int64)
    return indices[np.linspace(0, len(indices) - 1, count).round().astype(np.int64)]


def _balanced_counts(groups, budget):
    """Water-fill equal group budgets; deterministic ties favour earlier groups."""
    counts = np.zeros(len(groups), dtype=np.int64)
    while int(counts.sum()) < budget:
        available = [i for i, g in enumerate(groups) if counts[i] < len(g)]
        minimum = min(counts[i] for i in available)
        for i in available:
            if counts[i] == minimum:
                counts[i] += 1
                if int(counts.sum()) == budget:
                    break
    return counts


def select_windows(candidates, k, strategy="uniform", rng=None, group_ids=None):
    """Select candidate triples without replacement, retaining their source order.

    ``group_ids`` assigns one group per candidate (not per cached image).
    ``group_balanced`` water-fills equal quotas, giving [3,3,2,2,2] for budget 12
    with five sufficiently large groups. Within each group, selection is uniform
    when rng is None and random without replacement when an RNG is supplied.
    ``stratified`` samples one item per contiguous candidate interval. With groups,
    interval counts are allocated approximately proportionally to group capacity,
    while covering every group whenever the budget permits. These are index-space
    strategies, not claimed to be millimetre-space sampling.
    """
    candidates = np.asarray(candidates)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidates must have shape [N, 3]")
    if strategy not in {"uniform", "random", "group_balanced", "stratified"}:
        raise ValueError(f"unknown sampling strategy: {strategy}")
    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != (len(candidates),):
            raise ValueError("group_ids must have one value per candidate")
    if k == "all":
        return candidates.copy()
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("window budget must be a nonnegative integer or 'all'")
    budget = min(int(k), len(candidates))
    if budget == 0:
        return candidates[:0].copy()
    if budget == len(candidates):
        return candidates.copy()
    indices = np.arange(len(candidates))
    if strategy == "uniform":
        selected = _spread(indices, budget)
    elif strategy == "random":
        rng = np.random.default_rng() if rng is None else rng
        selected = rng.choice(indices, size=budget, replace=False)
    else:
        if strategy == "group_balanced" and group_ids is None:
            raise ValueError("group_balanced requires candidate group_ids")
        if group_ids is None:
            groups = [indices]
        else:
            # Preserve first occurrence, independent of arbitrary series UID codes.
            ordered = list(dict.fromkeys(group_ids.tolist()))
            groups = [indices[group_ids == group] for group in ordered]
        if budget < len(groups):
            chosen = _spread(np.arange(len(groups)), budget)
            groups = [groups[i] for i in chosen]
        if strategy == "group_balanced":
            counts = _balanced_counts(groups, budget)
        else:
            counts = np.ones(len(groups), dtype=np.int64)
            target = np.asarray([len(g) for g in groups], dtype=float) * budget / len(candidates)
            while counts.sum() < budget:
                deficit = target - counts
                deficit[counts >= np.asarray([len(g) for g in groups])] = -np.inf
                counts[int(np.argmax(deficit))] += 1
        selected_parts = []
        for group, count in zip(groups, counts):
            if strategy == "group_balanced":
                part = (_spread(group, int(count)) if rng is None
                        else rng.choice(group, size=int(count), replace=False))
            else:
                bins = np.array_split(group, int(count))
                part = np.array([interval[(len(interval) - 1) // 2] if rng is None
                                 else rng.choice(interval) for interval in bins])
            selected_parts.append(part)
        selected = np.concatenate(selected_parts)
    return candidates[np.sort(selected)].copy()
