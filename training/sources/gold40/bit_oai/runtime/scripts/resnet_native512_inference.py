#!/usr/bin/env python3
"""DICOM inference for native-resolution depth checkpoints (single model, no TTA).

Training windows came from build_native.py: the legacy full-field pipeline fixes
the window selection, source slices, orientation, frozen lo/hi and metadata, and
the pixels are the composed legacy 140 mm rectangle resampled ONCE from the raw
DICOM to the corpus resolution (kneemri.native_crop). This entry reproduces
exactly that for test studies: the legacy ``build_one`` supplies metadata and
window centres/neighbours, its pixels are discarded, and each window is rebuilt
from the same raw frames the pipeline decoded (served from the study's decode
cache). Canonical pixel flags and lateral positions are then applied exactly
as in resnet_depth_inference.install_positions.

The entry refuses a checkpoint whose training corpus was not built this way,
so a native model can never silently receive cache-style upsampled windows.
"""
import copy
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import resnet_depth_inference as rdi

NATIVE_STAGE = 'raw_dicom_composed_legacy_fov_single_resample'


def native_windows(result, serialized, details, metadata, decode, study_root, pipeline, resolution):
    """uint8 [64,3,R,R] windows for one study, mirroring build_native.study_windows."""
    from build_coatnet_original_corpus import MAX_WINDOWS, SLOTS
    from kneemri.geometry import sort_series_geometry
    from kneemri.native_crop import native_series_windows
    from kneemri.pipeline import _stack_same_shape
    windows = np.zeros((MAX_WINDOWS, 3, resolution, resolution), np.uint8)
    cursor = 0
    for (slot, budget, _), detail in zip(SLOTS, details):
        if detail['slot'] != slot:
            raise ValueError('window details are out of slot order')
        if not detail['present']:
            cursor += budget
            continue
        prefix = detail['source_prefix']
        slot_id, rank = prefix[len('slot_'):].rsplit('_s', 1)
        series = result.slot_ranks[slot_id][int(rank)]
        report = series.report
        rows = metadata[metadata.series_uid == series.series_uid]
        ordering = sort_series_geometry(rows, pipeline.geometry, label=prefix)
        if ordering.source != report['sort_source']:
            raise ValueError(f'{prefix}: slice ordering differs from the pipeline')
        ordered = ordering.ordered(rows)
        source = np.asarray(serialized[prefix + '_source_indices'], dtype=np.int64)
        raw = _stack_same_shape([decode(os.path.join(study_root, ordered.iloc[s].rel_path)) for s in source])
        orientation = report['orientation']
        if orientation['applied'] and orientation['horizontal_flip']:
            raw = raw[..., ::-1].copy()
        spacing = [report['crop']['spacing_row'], report['crop']['spacing_col']]
        lo, hi = report['intensity']['lo'], report['intensity']['hi']
        image, _ = native_series_windows(raw, np.asarray(detail['neighbours']), spacing, report['crop']['fov_mm'],
                                         serialized[prefix + '_output_spacing_mm_per_px'], lo, hi, resolution)
        if len(image) != detail['windows']:
            raise ValueError(f'{prefix}: native window count differs')
        windows[cursor:cursor + len(image)] = image
        cursor += budget
    return windows


def install_native(canonical_pixels, resolution):
    """Route DICOM preprocessing through native windows, canonical flags and lateral positions."""
    import coatnet_fast_runtime as cfr
    import inference_coatnet_original_dicom as runtime
    import resnet_canonical as rc
    from build_coatnet_original_corpus import build_one

    def prepare(uid, records, data_root, pipeline, contract, reader_pool, return_flags=False):
        started = time.monotonic()
        if contract.get('crop_stage') != NATIVE_STAGE or int(contract['image']) != resolution:
            raise ValueError('training corpus is not a native single-resample corpus of this resolution')
        pipeline = copy.deepcopy(pipeline)
        pipeline.study_root = str(Path(data_root) / 'test_series' / uid)
        reader = cfr.ReadOnceStudy(pipeline.study_root, reader_pool)
        if reader.metadata.empty:
            raise ValueError(f'no readable DICOM headers for {uid}')
        result = cfr.process_study(uid, pipeline, reader.metadata, records, reader.decode)
        if result.failed:
            raise ValueError(f'preprocessing failed for {uid}: {result.error}')
        serialized = cfr.result_arrays(result, pipeline)
        # Legacy builder at its own 384 grid: metadata, centres and neighbours only.
        legacy = build_one(0, uid, serialized, 384, contract['source_preprocess_signature'],
                           contract.get('crop_fov_mm'))
        windows = native_windows(result, serialized, legacy[8], reader.metadata, reader.decode,
                                 pipeline.study_root, pipeline, resolution)
        flip = swap = np.zeros(len(windows), bool)
        if canonical_pixels:
            flip, swap, _, _ = rc.window_flags(result, serialized, reader.metadata, len(windows))
            windows = rc.apply_flags(windows, flip, swap)
        multiplier = rdi.lateral_multipliers(result, serialized, reader.metadata, len(windows))
        positions = np.where(multiplier != 0, legacy[6] * multiplier, legacy[6]).astype(np.float32)
        arrays = (legacy[0], windows, *legacy[2:6], positions, *legacy[7:])
        stats = dict(total_s=time.monotonic() - started, canonical_flips=int(np.sum(flip)),
                     canonical_swaps=int(np.sum(swap)), lateral_windows=int((multiplier != 0).sum()),
                     native_resolution=resolution)
        if return_flags:
            return uid, arrays, stats, (flip, swap, multiplier)
        return uid, arrays, stats

    cfr.prepare_study_fast = prepare
    runtime.prepare_study_fast = prepare
    return prepare


def main():
    config = torch.load(rdi.checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    if not config['depth_stage'].get('canonical_positions'):
        raise ValueError('native entry expects the canonical lateral-position E2E recipe')
    install_native(bool(config.get('canonical_orientation')), int(config['resolution']))
    rdi.resnet.runtime.build_model = rdi.build_depth_model
    rdi.resnet.runtime.checkpoint_config = rdi.resnet.checkpoint_config
    rdi.resnet.runtime.main()


if __name__ == '__main__':
    main()
