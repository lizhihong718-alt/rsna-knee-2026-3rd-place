"""Inference-side laterality canonicalization, matching the training artifact.

Training applied per-window flags from ``artifacts/orientation_full.npz``, built by
``coatnet_v152_levers_fold0_20261001/build_lever_artifacts.py::process``: for a
study with no Laterality consensus, the side is inferred from the median
patient-space x of the image centres (|x| <= 20 mm stays unknown), and every
window of a medial-lateral series gets the decision ``kneemri.laterality``
would have taken for a tagged study (sagittal: reverse slice order, here a
prev/next channel swap; coronal/axial: horizontal mirror). Tagged studies were
already canonicalized by the pipeline and get no flags.

``window_flags`` repeats that computation line by line from the in-memory
pipeline result and DICOM header table, so DICOM inference sees exactly the
pixels the model was trained and validated on.
"""
from __future__ import annotations

import time

import numpy as np

import coatnet_fast_runtime as cfr
from build_coatnet_original_corpus import SLOTS, _centres, _prefix, build_one

MIN_AXIS_ALIGNMENT = 0.5      # configs/preprocess.yaml laterality.min_axis_alignment
LATERALITY_MIN_ABS_X_MM = 20.0


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def series_geometry(rows):
    """kneemri geometry: median of per-slice unit normals, renormalized; z = IPP . normal."""
    iops = rows[['iop_x', 'iop_y', 'iop_z', 'iop_cx', 'iop_cy', 'iop_cz']].to_numpy(float)
    normals = np.stack([unit(np.cross(i[:3] / np.linalg.norm(i[:3]), i[3:] / np.linalg.norm(i[3:])))
                        for i in iops])
    normal = unit(np.median(normals, axis=0))
    med = np.median(iops, axis=0)
    row_dir = unit(med[:3])
    return normal, row_dir


def window_flags(result, arrays, meta, n_windows):
    """Per corpus window: (flip, swap, consensus, inferred) for one processed study."""
    flip = np.zeros(n_windows, bool)
    swap = np.zeros(n_windows, bool)
    consensus = result.report.get('laterality_consensus') or ''
    meta = meta.copy()
    for column in ['iop_x', 'iop_y', 'iop_z', 'iop_cx', 'iop_cy', 'iop_cz', 'ipp_x', 'ipp_y', 'ipp_z',
                   'rows', 'columns', 'spacing_row', 'spacing_col']:
        meta[column] = meta[column].astype(float)
    centres_x = (meta.ipp_x + meta.iop_x * meta['columns'] / 2 * meta.spacing_col
                 + meta.iop_cx * meta['rows'] / 2 * meta.spacing_row)
    cx = float(np.median(centres_x))
    inferred = '' if abs(cx) <= LATERALITY_MIN_ABS_X_MM else ('L' if cx > 0 else 'R')
    if consensus:
        return flip, swap, consensus, inferred
    lateral_sign = {'L': 1.0, 'R': -1.0}.get(inferred, 0.0)
    if not lateral_sign:
        return flip, swap, consensus, inferred
    cursor = 0
    for slot, budget, _plane in SLOTS:
        prefix = _prefix(arrays, slot)
        if prefix is None:
            cursor += budget
            continue
        slot_id, rank = prefix[len('slot_'):].rsplit('_s', 1)
        slot_result = result.slot_ranks[slot_id][int(rank)]
        orient = slot_result.report['orientation']
        rows = meta[meta.series_uid == slot_result.series_uid]
        normal, row_dir = series_geometry(rows)
        count = len(_centres(len(np.asarray(arrays[prefix + '_z'])), budget))
        stack_k = int(np.argmax(np.abs(normal)))
        h_k = int(np.argmax(np.abs(row_dir)))
        want_reverse = want_flip = False
        if stack_k == 0 and abs(normal[0]) >= MIN_AXIS_ALIGNMENT:
            want_reverse = bool(normal[0] * lateral_sign < 0)
        if stack_k != 0 and h_k == 0 and abs(row_dir[0]) >= MIN_AXIS_ALIGNMENT:
            want_flip = bool(row_dir[0] * lateral_sign < 0)
        upstream = orient['reverse_stack'] if stack_k == 0 else orient['horizontal_flip']
        if (stack_k == 0 or h_k == 0) and upstream:
            raise ValueError('untagged study was already reoriented upstream')
        swap[cursor:cursor + count] = want_reverse
        flip[cursor:cursor + count] = want_flip
        cursor += budget
    return flip, swap, consensus, inferred


def apply_flags(windows, flip, swap):
    """uint8 [K,3,H,W]: mirror the W axis and reverse the slice channels, as in training."""
    windows = np.array(windows, copy=True)
    windows[flip] = windows[flip][..., ::-1]
    windows[swap] = windows[swap][:, ::-1]
    return windows


def prepare_study_fast_canonical(uid, records, data_root, pipeline, contract, reader_pool,
                                 return_flags=False):
    """``coatnet_fast_runtime.prepare_study_fast`` plus the canonicalization step."""
    import copy
    from pathlib import Path
    started = time.monotonic()
    pipeline = copy.deepcopy(pipeline)
    pipeline.study_root = str(Path(data_root) / 'test_series' / uid)
    reader = cfr.ReadOnceStudy(pipeline.study_root, reader_pool)
    read_done = time.monotonic()
    if reader.metadata.empty:
        raise ValueError(f'no readable DICOM headers for {uid}')
    files = len(reader.datasets)
    result = cfr.process_study(uid, pipeline, reader.metadata, records, reader.decode)
    process_done = time.monotonic()
    if result.failed:
        raise ValueError(f'preprocessing failed for {uid}: {result.error}')
    serialized = cfr.result_arrays(result, pipeline)
    arrays = build_one(0, uid, serialized, contract['image'],
                       contract['source_preprocess_signature'], contract.get('crop_fov_mm'))
    flip, swap, consensus, inferred = window_flags(result, serialized, reader.metadata, len(arrays[1]))
    arrays = (arrays[0], apply_flags(arrays[1], flip, swap), *arrays[2:])
    stats = dict(read_s=read_done - started, process_s=process_done - read_done,
                 windows_s=time.monotonic() - process_done, files=files,
                 decoded_files=len(reader.decoded), canonical_flips=int(flip.sum()),
                 canonical_swaps=int(swap.sum()), laterality_consensus=consensus,
                 laterality_inferred=inferred)
    if return_flags:
        return uid, arrays, stats, (flip, swap)
    return uid, arrays, stats


def install():
    """Route every DICOM preprocessing call (parent and forked workers) through canonicalization."""
    import inference_coatnet_original_dicom as runtime
    cfr.prepare_study_fast = prepare_study_fast_canonical
    runtime.prepare_study_fast = prepare_study_fast_canonical
