#!/usr/bin/env python3
"""Single-checkpoint DICOM inference for native384 ConvNeXt / DINOv3-ViT checkpoints.

Pixels follow reports/native_corpora_20261009/build_native.py: the legacy pipeline
fixes window selection, source slices, orientation, frozen lo/hi and metadata; each
window is the composed legacy 140 mm rectangle resampled ONCE from the raw DICOM
(kneemri.native_crop, sha e7512520..., identical to the corpus builder). native_windows
is copied verbatim from resnet_native512_20261009/runtime_e2e (89/89 train-DICOM parity
there). Canonical flags are applied as in resnet_canonical; positions stay the legacy
values (these heads do not consume positions). No TTA, one checkpoint.
"""
import copy
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import inference_coatnet_original_dicom as runtime  # noqa: E402
import train_coatnet_ablation as trainer  # noqa: E402
from kneemri.gold_cohort import GOLD_KEYS  # noqa: E402
from resnet_fullstrength import CONFIG_KEYS  # noqa: E402

# Training-only fields (objective, cohort, augmentation, optimizer) never alter inference.
trainer.DATA_KEYS.update(CONFIG_KEYS | GOLD_KEYS | {'canonical_orientation', 'warp_interpolation',
                                                    'geometric_prob', 'layer_decay', 'layer_decay_coarse',
                                                    'blur_prob', 'blur_sigma'})
trainer.MODEL_KEYS.update({'drop_path_rate', 'output_stride', 'dino_gap', 'dino_layers'})
NATIVE_STAGE = 'raw_dicom_composed_legacy_fov_single_resample'
SCHEMA = 'cnx-dino-native384-study-mil-v1'


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


# Copied verbatim from resnet_native512_20261009/runtime_e2e/scripts/resnet_depth_inference.py
# (native512 parity 89/89 incl. positions): canonical lateral mm for E2E_D checkpoints.
def lateral_multipliers(result, serialized, meta, n_windows):
    """Per corpus window: sign that turns the physical position into lateral mm (0 = keep)."""
    import resnet_canonical as rc
    from build_coatnet_original_corpus import SLOTS, _centres, _prefix
    multiplier = np.zeros(n_windows, np.float32)
    consensus = result.report.get('laterality_consensus') or ''
    meta = meta.copy()
    for column in ['iop_x', 'iop_y', 'iop_z', 'iop_cx', 'iop_cy', 'iop_cz', 'ipp_x', 'ipp_y', 'ipp_z',
                   'rows', 'columns', 'spacing_row', 'spacing_col']:
        meta[column] = meta[column].astype(float)
    centres_x = (meta.ipp_x + meta.iop_x * meta['columns'] / 2 * meta.spacing_col
                 + meta.iop_cx * meta['rows'] / 2 * meta.spacing_row)
    cx = float(np.median(centres_x))
    inferred = '' if abs(cx) <= rc.LATERALITY_MIN_ABS_X_MM else ('L' if cx > 0 else 'R')
    lateral_sign = {'L': 1.0, 'R': -1.0}.get(consensus or inferred, 0.0)
    if not lateral_sign:
        return multiplier
    cursor = 0
    for slot, budget, plane in SLOTS:
        prefix = _prefix(serialized, slot)
        if prefix is None:
            cursor += budget
            continue
        z = np.asarray(serialized[prefix + '_z'], dtype=np.float64)
        count = len(_centres(len(z), budget))
        if plane == 0 and np.isfinite(z).all():
            slot_id, rank = prefix[len('slot_'):].rsplit('_s', 1)
            series_uid = result.slot_ranks[slot_id][int(rank)].series_uid
            normal, _ = rc.series_geometry(meta[meta.series_uid == series_uid])
            multiplier[cursor:cursor + count] = np.sign(normal[0] * lateral_sign)
        cursor += budget
    return multiplier


def install_native(canonical_pixels, resolution, lateral_positions=False):
    """Route every DICOM preprocessing call (parent and forked workers) through native windows."""
    import coatnet_fast_runtime as cfr
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
        positions = legacy[6]
        lateral = 0
        if lateral_positions:
            multiplier = lateral_multipliers(result, serialized, reader.metadata, len(windows))
            positions = np.where(multiplier != 0, legacy[6] * multiplier, legacy[6]).astype(np.float32)
            lateral = int((multiplier != 0).sum())
        arrays = (legacy[0], windows, *legacy[2:6], positions, *legacy[7:])
        stats = dict(total_s=time.monotonic() - started, canonical_flips=int(np.sum(flip)),
                     canonical_swaps=int(np.sum(swap)), native_resolution=resolution, lateral_windows=lateral)
        if return_flags:
            return uid, arrays, stats, (flip, swap)
        return uid, arrays, stats

    cfr.prepare_study_fast = prepare
    runtime.prepare_study_fast = prepare
    return prepare


_checkpoint_config = runtime.checkpoint_config


def checkpoint_config(checkpoint, config=None):
    if checkpoint['receipt'].get('schema') != SCHEMA:
        raise ValueError('this entrypoint requires an audited native384 ConvNeXt/DINO checkpoint')
    return _checkpoint_config(checkpoint, config)


def checkpoint_path(argv):
    for index, value in enumerate(argv):
        if value == '--checkpoint':
            return argv[index + 1]
        if value.startswith('--checkpoint='):
            return value.split('=', 1)[1]
    raise ValueError('--checkpoint is required')


def main():
    config = torch.load(checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    depth = bool(config.get('depth_e2e'))
    if depth:
        # runtime_d E2E_D: depth model builder and canonical lateral positions, as in training.
        from kneemri import depth384
        trainer.MODEL_KEYS.update(depth384.DEPTH_MODEL_KEYS)
        trainer.DATA_KEYS.update(depth384.DEPTH_DATA_KEYS)
        runtime.build_model = depth384.build_model
        if not config.get('depth_canonical_positions'):
            raise ValueError('E2E_D inference expects canonical lateral positions')
    install_native(bool(config.get('canonical_orientation')), int(config['resolution']),
                   lateral_positions=depth)
    print(f"native{config['resolution']} windows; laterality canonicalization: "
          f"{'enabled' if config.get('canonical_orientation') else 'disabled'}", flush=True)
    runtime.checkpoint_config = checkpoint_config
    runtime.main()


if __name__ == '__main__':
    main()
