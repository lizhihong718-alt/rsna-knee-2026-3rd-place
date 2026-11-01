"""Section 21: deterministic grouped K-fold.

The corpus has leakage channels a plain random KFold would ignore:
  - identical/duplicated radiology reports (49 groups / 183 studies in train)
  - shared anonymized PatientID across studies (both knees of one patient)
so the default is GroupKFold on a configurable group column:
    report_hash  md5 of the normalized report text (train.csv)
    patient_id   majority PatientID from the DICOM header cache
Fold assignment is fully deterministic (hash-shuffle + greedy balancing) and
saved to CSV.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from kneemri.config import abspath


def report_hash_column(series_table: pd.DataFrame) -> pd.Series:
    """md5 of the report text; missing/empty reports get their own group per
    study (never share a group silently)."""
    def norm(r) -> str:
        if not isinstance(r, str) or not r.strip():
            return ""
        return " ".join(r.split()).strip().lower()

    hashes = []
    for i, r in enumerate(series_table.get("Report", pd.Series([None] * len(series_table)))):
        n = norm(r)
        h = hashlib.md5(n.encode()).hexdigest() if n else f"no_report_{i}"
        hashes.append(h)
    return pd.Series(hashes, index=series_table.index, name="report_hash")


def patient_id_column(meta_cache: pd.DataFrame,
                      study_uids: pd.Index) -> pd.Series:
    """Majority PatientID over the study's slices ('' when absent)."""
    pid = (meta_cache.dropna(subset=["patient_id"])
           .groupby("study_uid")["patient_id"]
           .agg(lambda s: s.value_counts().idxmax()))
    return pid.reindex(study_uids).fillna("").rename("patient_id")


def build_groups(study_uids: list[str], cfg) -> pd.Series:
    """Group key per study according to cfg.cv.group_column."""
    cfg_cv = cfg.cv
    if cfg_cv.group_column == "report_hash":
        # train.csv: one row per study (Report + labels). The series table is
        # per-series and would break reindex with duplicate UIDs.
        table = pd.read_csv(abspath(cfg.dataset.train_csv)).set_index(
            "StudyInstanceUID")
        table = table[~table.index.duplicated(keep="first")]
        h = report_hash_column(table.reindex(study_uids))
        h.index = pd.Index(study_uids, name="StudyInstanceUID")
        return h
    if cfg_cv.group_column == "patient_id":
        meta = pd.read_parquet(abspath(cfg.meta_cache),
                               columns=["study_uid", "patient_id"])
        ids = patient_id_column(meta, pd.Index(study_uids))
        return pd.Series([p if isinstance(p,str) and p.strip() else f"unknown_{u}" for u,p in zip(study_uids,ids)], index=study_uids)
    raise ValueError(f"unknown group_column '{cfg_cv.group_column}' "
                     "(report_hash | patient_id)")


def assign_folds(study_uids: list[str], groups: pd.Series, folds: int,
                 seed: int) -> pd.DataFrame:
    """Deterministic + balanced: groups hash-shuffled by seed, then each
    group goes to the fold with the fewest assigned studies (ties -> lowest
    fold index). Same inputs always yield the same folds."""
    if folds < 2:
        raise ValueError(f"need folds >= 2, got {folds}")
    uniq = sorted(set(groups.tolist()))
    # deterministic group order: sha1 of (seed, group)
    order = sorted(uniq, key=lambda g: hashlib.sha1(
        f"{seed}:{g}".encode()).hexdigest())
    counts = np.zeros(folds, dtype=np.int64)
    group_fold = {}
    for g in order:
        sizes = groups.tolist().count(g)
        f = int(np.argmin(counts))
        group_fold[g] = f
        counts[f] += sizes
    fold_of = groups.map(group_fold).astype(int)
    return pd.DataFrame({
        "StudyInstanceUID": list(study_uids),
        "group": groups.tolist(),
        "fold": fold_of.tolist(),
    })


def build_folds(cfg, out_csv: str | None = None) -> pd.DataFrame:
    """Full fold table for the studies present in the processed npz dir."""
    import glob
    import os

    # Build the split from the authoritative study list even if preprocessing
    # is still running; never silently freeze a partial data split.
    train_path = abspath(cfg.dataset.train_csv)
    if os.path.exists(train_path):
        uids = sorted(pd.read_csv(train_path)["StudyInstanceUID"].unique().tolist())
    else:
        files = sorted(glob.glob(os.path.join(abspath(cfg.dataset.processed_dir), "*.npz")))
        uids = [os.path.basename(p)[:-4] for p in files]
    if cfg.cv.type == "group":
        groups = build_groups(uids, cfg)
    elif cfg.cv.type == "random":
        groups = pd.Series(uids, index=uids)   # each study its own group
    else:
        raise ValueError(f"unknown cv.type '{cfg.cv.type}' (group | random)")
    df = assign_folds(uids, groups, cfg.cv.folds, cfg.cv.seed)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def fold_split(folds_df: pd.DataFrame, val_fold: int
               ) -> tuple[list[str], list[str]]:
    """(train_uids, val_uids) for one fold — deterministic order."""
    if folds_df.StudyInstanceUID.isna().any() or folds_df.StudyInstanceUID.duplicated().any():
        raise ValueError('fold table contains duplicate/missing study IDs')
    if 'group' in folds_df and folds_df.groupby('group').fold.nunique().gt(1).any():
        raise ValueError('a patient/report group crosses validation folds')
    tr = folds_df[folds_df["fold"] != val_fold]["StudyInstanceUID"].tolist()
    va = folds_df[folds_df["fold"] == val_fold]["StudyInstanceUID"].tolist()
    if not tr or not va:
        raise ValueError('training or validation fold is empty')
    return tr, va
