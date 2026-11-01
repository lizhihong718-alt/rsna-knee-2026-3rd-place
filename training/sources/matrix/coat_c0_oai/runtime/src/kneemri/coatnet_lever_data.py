"""Lever variants of the original-neighbour CoAtNet bags (2026-10-01 Fold0 experiments).

E2  CoAtNetEvidenceBags: training draws, for each Astra-positive class whose cited slices
    map to corpus windows, one window from those cited slices (random class order, at most
    ``evidence_max_windows``), then fills the rest of the budget at random as before.
    Evaluation is unchanged (deterministic, normally all windows).
E3  CoAtNetCanonicalBags: applies the preprocessing laterality canonicalization that
    untagged studies never received (horizontal mirror for coronal/axial, prev/next channel
    swap for sagittal stack reversal) and supplies a lateral-ward sagittal position in mm.
    Applied identically in training and evaluation. Artifacts come from
    build_lever_artifacts.py, whose replica of the pipeline matched all tagged decisions.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
from kneemri.labels import LABEL_COLS

from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags


class CoAtNetEvidenceBags(CoAtNetOriginalNeighborBags):
    def __init__(self, cfg, all_ids, ids, labels, train):
        super().__init__(cfg, all_ids, ids, labels, train)
        self.uid_by_row = {row: uid for uid, row in self.rows.items()}
        self.evidence = json.loads(Path(cfg['evidence_windows']).read_text()) if self.train else {}
        self.evidence_cap = int(cfg.get('evidence_max_windows', 6))
        if self.train and self.strategy != 'random':
            raise ValueError('evidence sampling extends the random training sampler only')
        self._forced = 0
        self._last_selected = None

    def _selected(self, row):
        if not self.train:
            return super()._selected(row)
        valid = np.flatnonzero(np.asarray(self.masks[row], dtype=bool))
        budget = random.choice(self.budget) if isinstance(self.budget, list) else self.budget
        rng = np.random.default_rng(random.getrandbits(64))
        self._forced = 0
        if budget == 'all' or len(valid) <= budget:
            self._last_selected = valid
            return valid, len(valid), budget
        allowed = set(valid.tolist())
        cited = {label: [w for w in windows if w in allowed]
                 for label, windows in self.evidence.get(self.uid_by_row[row], {}).items()}
        labels = [label for label, windows in cited.items() if windows]
        picked = []
        for index in rng.permutation(len(labels)):
            if len(picked) >= min(self.evidence_cap, budget):
                break
            options = [w for w in cited[labels[index]] if w not in picked]
            if options:
                picked.append(int(rng.choice(options)))
        self._forced = len(picked)
        rest = np.setdiff1d(valid, np.asarray(picked, dtype=valid.dtype))
        fill = rng.choice(rest, size=budget - len(picked), replace=False)
        selected = np.sort(np.concatenate([np.asarray(picked, dtype=valid.dtype), fill]))
        self._last_selected = selected
        return selected, len(valid), budget

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['provenance']['evidence_windows_forced'] = int(self._forced)
        selected = np.asarray(self._last_selected if self._last_selected is not None else [], dtype=np.int64)
        mask = np.zeros((len(selected), len(LABEL_COLS)), dtype=np.float32)
        cited = self.evidence.get(self.uid_by_row[row], {})
        for li, label in enumerate(LABEL_COLS):
            for window in cited.get(label, []):
                hits = np.flatnonzero(selected == int(window))
                if len(hits):
                    mask[hits, li] = 1.0
        item['evidence_mask'] = torch.from_numpy(mask)
        return item


class CoAtNetCanonicalBags(CoAtNetOriginalNeighborBags):
    def __init__(self, cfg, all_ids, ids, labels, train):
        super().__init__(cfg, all_ids, ids, labels, train)
        with np.load(cfg['orientation_artifacts']) as z:
            if z['study_uids'].tolist() != list(all_ids):
                raise ValueError('orientation artifacts are not aligned with the corpus rows')
            self.flip, self.swap = z['flip'].astype(bool), z['swap'].astype(bool)
            self.lat_pos, self.lat_pos_valid = z['lat_pos'], z['lat_pos_valid'].astype(bool)
        self._pending = None

    def _selected(self, row):
        selected, count, budget = super()._selected(row)
        self._pending = (row, np.asarray(selected, dtype=np.int64))
        return selected, count, budget

    def _augment(self, images):
        row, selected = self._pending
        images = np.array(images, copy=True)
        flip, swap = self.flip[row, selected], self.swap[row, selected]
        if flip.any():
            images[flip] = images[flip][..., ::-1]
        if swap.any():
            images[swap] = images[swap][:, ::-1]
        return super()._augment(images)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row, selected = self._pending
        if len(selected):
            reliable = np.asarray(self.metadata['position_reliable'][row, selected], dtype=bool)
            valid = self.lat_pos_valid[row, selected] & reliable
            positions = np.where(valid, self.lat_pos[row, selected], 0).astype(np.float32)[:, None]
            item['positions'] = torch.from_numpy(positions)
            item['position_mask'] = torch.from_numpy(valid.copy())
            item['provenance']['canonical_flips'] = int(self.flip[row, selected].sum())
            item['provenance']['canonical_swaps'] = int(self.swap[row, selected].sum())
        return item
