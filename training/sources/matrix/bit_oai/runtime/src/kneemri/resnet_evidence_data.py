"""E2 evidence sampling for the original-neighbour bags (training only).

Port of the 2026-10-01 CoAtNet lever (``coatnet_lever_data.CoAtNetEvidenceBags``)
onto the uint8-transfer data class used by the ResNet runtime. For each Astra
positive class whose cited slices map to corpus windows, one cited window is
forced into the training bag (random class order, at most
``evidence_max_windows``); the rest of the budget is filled at random from the
remaining valid windows. Evaluation, Gold58 scoring and DICOM inference never
construct this class, so their windows are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags


class EvidenceNeighborBags(CoAtNetOriginalNeighborBags):
    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
        if not train:
            raise ValueError('evidence sampling is a training-only sampler')
        super().__init__(cfg, all_ids, ids, labels, train, uint8_output=uint8_output)
        if self.strategy != 'random' or isinstance(self.budget, list):
            raise ValueError('evidence sampling extends the fixed-budget random sampler only')
        self.uid_by_row = {row: uid for uid, row in self.rows.items()}
        evidence = json.loads(Path(cfg['train_evidence_windows']).read_text())
        if not set(self.ids) <= set(evidence):
            raise ValueError('evidence artifact does not cover every training study')
        self.evidence = {uid: evidence[uid] for uid in self.ids}
        self.evidence_cap = int(cfg.get('evidence_max_windows', 6))
        # Cited windows forced per positive class; 1 reproduces the original E2 draws.
        self.per_class = int(cfg.get('evidence_per_class', 1))
        if self.per_class < 1:
            raise ValueError('evidence_per_class must be a positive integer')
        if not 0 < self.evidence_cap <= int(self.budget):
            raise ValueError('evidence_max_windows must lie in 1..train_windows')
        self._forced = 0

    def _selected(self, row, budget=None):
        valid = np.flatnonzero(np.asarray(self.masks[row], dtype=bool))
        budget = self.budget if budget is None else budget
        rng = np.random.default_rng(random.getrandbits(64))
        self._forced = 0
        if budget == 'all' or len(valid) <= budget:
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
            if not options:
                continue
            if self.per_class == 1:
                picked.append(int(rng.choice(options)))
            else:
                take = min(self.per_class, len(options), min(self.evidence_cap, budget) - len(picked))
                picked.extend(int(w) for w in rng.choice(options, size=take, replace=False))
        self._forced = len(picked)
        rest = np.setdiff1d(valid, np.asarray(picked, dtype=valid.dtype))
        fill = rng.choice(rest, size=budget - len(picked), replace=False)
        selected = np.sort(np.concatenate([np.asarray(picked, dtype=valid.dtype), fill]))
        return selected, len(valid), budget

    def _trace(self, uid, selected, candidate_count, budget):
        trace = super()._trace(uid, selected, candidate_count, budget)
        trace['evidence_windows_forced'] = int(self._forced)
        return trace
