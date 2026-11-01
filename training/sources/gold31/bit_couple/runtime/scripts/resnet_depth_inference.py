#!/usr/bin/env python3
"""DICOM inference for a depth-stage ResNet checkpoint (single model, no TTA).

The checkpoint holds the complete state of ``DepthStageModel`` (BiT backbone,
spatial queries/MIL and the depth module) and its config carries a
``depth_stage`` block. Inference reuses the audited streaming engine unchanged:
``encode`` returns the final feature maps of each window chunk and ``head``
applies depth context, spatial decoding and the study MIL.

When the model was trained on canonical lateral positions, sagittal window
positions are multiplied by sign(normal_x * lateral_sign) exactly as
``build_lever_artifacts.process`` computed ``lat_pos`` for training (side =
Laterality consensus, else inferred from the median image-centre x).
"""
import sys

import numpy as np
import torch

import inference_resnet_dicom as resnet
from resnet_fullstrength import CONFIG_KEYS
from kneemri.resnet_depth import DepthContext, DepthStageModel

DEPTH_KEYS = {'depth_stage', 'depth_canonical_positions', 'depth_e2e', 'series_block_cap'}
resnet.trainer.DATA_KEYS.update(CONFIG_KEYS | DEPTH_KEYS)


class DepthInferenceModel(DepthStageModel):
    @property
    def n(self):
        return self.base.n

    def encode(self, x, window_mask=None, group_ids=None):
        b, k = x.shape[:2]
        valid = (torch.ones((b, k), dtype=torch.bool, device=x.device) if window_mask is None
                 else window_mask).flatten()
        if not valid.any():
            raise ValueError('a chunk without valid windows reached the depth encoder')
        maps = self.suffix(self.prefix(x.flatten(0, 1)[valid]))
        out = maps.new_zeros((b * k, *maps.shape[1:]))
        out[valid] = maps
        return out.view(b, k, *maps.shape[1:])

    def head(self, maps, window_mask=None, group_ids=None, positions=None, series_ids=None,
             position_mask=None, plane_ids=None):
        metadata = dict(window_mask=window_mask, group_ids=group_ids, positions=positions,
                        series_ids=series_ids, position_mask=position_mask, plane_ids=plane_ids)
        return self.study_logits(maps, metadata)


def build_depth_model(cfg, pretrained=False):
    from kneemri.resnet_model import build_model
    stage = cfg['depth_stage']
    plain = {k: v for k, v in cfg.items() if k not in DEPTH_KEYS}
    if cfg.get('depth_e2e'):
        from kneemri.resnet_depth_e2e import build_model as build_e2e
        base = build_e2e(cfg, pretrained=False)
    else:
        base = build_model(plain, pretrained=False)
    depth = None
    if stage['use_depth']:
        d = stage['depth']
        depth = DepthContext(base.feature_dim, width=d['width'], heads=d['heads'], regions=d['regions'],
                             absolute=d['absolute'], cross_series=d['cross_series'])
    return DepthInferenceModel(base, depth, stage['trainable_stages'])


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


def install_positions(canonical_pixels):
    """Route DICOM preprocessing through pixel canonicalization (optional) and lateral positions."""
    import copy
    import time
    from pathlib import Path
    import coatnet_fast_runtime as cfr
    import inference_coatnet_original_dicom as runtime
    import resnet_canonical as rc
    from build_coatnet_original_corpus import build_one

    def prepare(uid, records, data_root, pipeline, contract, reader_pool, return_flags=False):
        started = time.monotonic()
        pipeline = copy.deepcopy(pipeline)
        pipeline.study_root = str(Path(data_root) / 'test_series' / uid)
        reader = cfr.ReadOnceStudy(pipeline.study_root, reader_pool)
        if reader.metadata.empty:
            raise ValueError(f'no readable DICOM headers for {uid}')
        result = cfr.process_study(uid, pipeline, reader.metadata, records, reader.decode)
        if result.failed:
            raise ValueError(f'preprocessing failed for {uid}: {result.error}')
        serialized = cfr.result_arrays(result, pipeline)
        arrays = build_one(0, uid, serialized, contract['image'],
                           contract['source_preprocess_signature'], contract.get('crop_fov_mm'))
        windows = arrays[1]
        flip = swap = np.zeros(len(windows), bool)
        if canonical_pixels:
            flip, swap, _, _ = rc.window_flags(result, serialized, reader.metadata, len(windows))
            windows = rc.apply_flags(windows, flip, swap)
        multiplier = lateral_multipliers(result, serialized, reader.metadata, len(windows))
        positions = np.where(multiplier != 0, arrays[6] * multiplier, arrays[6]).astype(np.float32)
        arrays = (arrays[0], windows, *arrays[2:6], positions, *arrays[7:])
        stats = dict(total_s=time.monotonic() - started, canonical_flips=int(np.sum(flip)),
                     canonical_swaps=int(np.sum(swap)), lateral_windows=int((multiplier != 0).sum()))
        if return_flags:
            return uid, arrays, stats, (flip, swap, multiplier)
        return uid, arrays, stats

    cfr.prepare_study_fast = prepare
    runtime.prepare_study_fast = prepare
    return prepare


def checkpoint_path(argv):
    for index, value in enumerate(argv):
        if value == '--checkpoint':
            return argv[index + 1]
        if value.startswith('--checkpoint='):
            return value.split('=', 1)[1]
    raise ValueError('--checkpoint is required')


def main():
    config = torch.load(checkpoint_path(sys.argv[1:]), map_location='cpu', weights_only=True)['config']
    if config['depth_stage'].get('canonical_positions'):
        install_positions(bool(config.get('canonical_orientation')))
    elif config.get('canonical_orientation'):
        import resnet_canonical
        resnet_canonical.install()
    resnet.runtime.build_model = build_depth_model
    resnet.runtime.checkpoint_config = resnet.checkpoint_config
    resnet.runtime.main()


if __name__ == '__main__':
    main()
