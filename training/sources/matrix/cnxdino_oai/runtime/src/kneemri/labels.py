"""Validated study/target lookup, independent of any image feature cache."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from kneemri.config import abspath

LABEL_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
              "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def read_labels(path):
    frame = pd.read_csv(abspath(path))
    required = ['StudyInstanceUID', *LABEL_COLS]
    if not set(required).issubset(frame.columns):
        raise ValueError(f'label table lacks required columns: {path}')
    if frame.StudyInstanceUID.isna().any() or frame.StudyInstanceUID.duplicated().any():
        raise ValueError(f'duplicate or missing label study IDs: {path}')
    frame = frame.set_index('StudyInstanceUID')[LABEL_COLS].astype(np.float32)
    v = frame.to_numpy()
    if np.isinf(v).any() or ((v[np.isfinite(v)] < 0) | (v[np.isfinite(v)] > 1)).any():
        raise ValueError(f'labels must be probabilities in [0,1] or NaN: {path}')
    return frame


class LabelTable:
    def __init__(self, labels_csv='', gold_csv='', require_files=False):
        self.pseudo = self.gold = None
        self.fingerprints = {}
        for name, path in [('pseudo', labels_csv), ('gold', gold_csv)]:
            if not path:
                continue
            source = Path(abspath(path))
            if not source.exists():
                if require_files:
                    raise FileNotFoundError(source)
                continue
            setattr(self, name, read_labels(path))
            self.fingerprints[name] = hashlib.sha256(source.read_bytes()).hexdigest()

    def get(self, uid):
        target = np.full(len(LABEL_COLS), np.nan, np.float32)
        gold_mask = np.zeros(len(LABEL_COLS), bool)
        if self.pseudo is not None and uid in self.pseudo.index:
            target = self.pseudo.loc[uid].to_numpy(np.float32).copy()
        if self.gold is not None and uid in self.gold.index:
            gold = self.gold.loc[uid].to_numpy(np.float32)
            gold_mask = np.isfinite(gold)
            target = np.where(gold_mask, gold, target)
        return target, gold_mask

    def validate_coverage(self, uids):
        missing = [u for u in uids if not np.isfinite(self.get(u)[0]).any()]
        if missing:
            raise ValueError(f'{len(missing)} studies have no training targets; first: {missing[0]}')
