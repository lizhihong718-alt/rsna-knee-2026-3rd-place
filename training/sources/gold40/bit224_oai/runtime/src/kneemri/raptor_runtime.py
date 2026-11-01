"""Checkpoint-driven CoAtNet inference with the same window contract as validation."""
from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from kneemri import raptor_reference as raptor
from kneemri.coatnet_data import candidate_windows, normalization_tensors, select_windows
from kneemri.coatnet_model import build_model


def _budget(value):
    if value == 'all':
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("window budget must be a positive integer or 'all'")
    return value


def checkpoint_input_contract(config):
    historical = 'window_mode' not in config
    return {
        'window_mode': config.get('window_mode', 'legacy'),
        'eval_windows': _budget(config.get('eval_windows', 42)),
        'eval_sampling': config.get('eval_sampling', 'uniform'),
        'deduplicate_windows': bool(config.get('deduplicate_windows', False)),
        # Pre-ablation checkpoints have no window_mode field and used the
        # duplicate-capable linspace->round evaluator. Explicit legacy configs
        # may intentionally select without replacement.
        'legacy_sampling': bool(config.get('legacy_sampling', historical)),
        'normalization': config.get('normalization', 'imagenet'),
        'resolution': int(config.get('resolution', 384)),
        'eval_encode_chunk': int(config.get('eval_encode_chunk', 32)),
        'eval_amp': config.get('eval_amp', 'fp16'),
        'preprocess_workers': int(config.get('preprocess_workers', 2)),
        'cudnn_benchmark': False,
    }


def load_checkpoint(path, device='cpu', expected_sha256=None):
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('checkpoint SHA256 differs from the pinned bundle manifest')
    checkpoint = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    del payload
    if checkpoint.get('lab', raptor.LAB) != raptor.LAB:
        raise ValueError('checkpoint target order differs')
    arch, resolution = checkpoint['arch'], int(checkpoint['res'])
    config = dict(checkpoint.get('config', {}))
    if config and (config.get('normalization') not in {'unit', 'imagenet', 'timm_half'}
                   or int(config.get('resolution', resolution)) != resolution):
        raise ValueError('checkpoint normalization or resolution differs')
    config.update(arch=arch, resolution=resolution, num_classes=12,
                  gradient_checkpointing=False)
    model = build_model(config, pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().to(device)
    cohort = checkpoint.get('cohort', {})
    metadata = {
        'checkpoint': str(path), 'weights_sha256': digest,
        'model_origin': cohort.get('model_origin', 'external task-trained checkpoint'),
        'source_tag': checkpoint.get('src'), 'epoch_one_based': checkpoint['epoch'] + 1,
        'cohort': cohort, 'arch': arch, 'resolution': resolution,
        'config': config, 'input_contract': checkpoint_input_contract(config),
    }
    return model, metadata


def evaluation_role(metadata, study_uids):
    """Refuse an in-training evaluation being presented as held-out evidence."""
    cohort = metadata['cohort']
    if not cohort:
        return 'publisher selection set; image training membership not independently verified'
    train, held = set(cohort['trained_study_uids']), set(study_uids)
    if train & held:
        raise ValueError('evaluation studies overlap checkpoint image training')
    monitor = set(cohort.get('gold_monitor_study_uids', []))
    declared = monitor if held == monitor else set(cohort.get('validation_study_uids', []))
    if held != declared:
        raise ValueError('evaluation cohort differs from declared checkpoint validation or monitor set')
    return 'declared held-out checkpoint evaluation cohort; development evidence, not an independent final test'


def _actual_groups(plane, fluid):
    groups = np.full(plane.shape, -1, np.int64)
    for plane_id, fluid_id, group_id in [(0, 1, 0), (0, 0, 1), (1, 1, 2), (1, 0, 3)]:
        groups[(plane == plane_id) & (fluid == fluid_id)] = group_id
    groups[plane == 2] = 4
    return groups


def _window_metadata(windows, mask, source):
    series = np.asarray(source['series_index'])
    plane = np.asarray(source['actual_plane'])
    fluid = np.asarray(source['fluid'])
    identities = series[windows]
    same = ((identities >= 0).all(1) & (identities == identities[:, :1]).all(1)
            & np.asarray(mask, bool)[windows].all(1))
    series_ids = np.where(same, identities[:, 1], -1).astype(np.int64)
    groups = _actual_groups(plane, fluid)
    same_group = (groups[windows] >= 0).all(1) & (groups[windows] == groups[windows[:, :1]]).all(1)
    group_ids = np.where(same & same_group, groups[windows[:, 1]], -1).astype(np.int64)
    same_plane = (plane[windows] >= 0).all(1) & (plane[windows] == plane[windows[:, :1]]).all(1)
    plane_ids = np.where(same & same_plane, plane[windows[:, 1]], -1).astype(np.int64)
    physical = np.asarray(source.get('physical_position_mm', np.full(len(mask), np.nan)))
    reliable = np.asarray(source.get('geometry_reliable', np.zeros(len(mask))), bool) & np.isfinite(physical)
    known = same & reliable[windows].all(1)
    positions = np.zeros((len(windows), 1), np.float32)
    for identity in np.unique(series_ids[known]):
        extent = physical[(series == identity) & reliable & np.asarray(mask, bool)]
        midpoint = (float(extent.min()) + float(extent.max())) / 2
        selected = known & (series_ids == identity)
        positions[selected, 0] = physical[windows[selected, 1]] - midpoint
    return dict(group_ids=group_ids, series_ids=series_ids, plane_ids=plane_ids,
                positions=positions, position_mask=known)


def prepare_study_inputs(volume, mask, source, config, windows=None, window_mode=None):
    """Construct one deterministic study bag using the validation sampling contract."""
    contract = checkpoint_input_contract(config)
    budget = contract['eval_windows'] if windows in (None, 'checkpoint') else _budget(windows)
    mode = contract['window_mode'] if window_mode in (None, 'checkpoint') else window_mode
    if mode not in {'legacy', 'no_cross_series', 'same_series'}:
        raise ValueError('window_mode must be legacy, no_cross_series, same_series, or checkpoint')
    mean, std = normalization_tensors(contract['normalization'])
    candidates = candidate_windows(mask, source['series_index'], source['original_index'],
                                   mode=mode, deduplicate=contract['deduplicate_windows'])
    candidate_meta = _window_metadata(candidates, mask, source) if len(candidates) else None
    if budget == 'all' or not len(candidates):
        selected = candidates.copy()
    elif contract['legacy_sampling']:
        if mode != 'legacy' or contract['eval_sampling'] != 'uniform' or contract['deduplicate_windows']:
            raise ValueError('legacy_sampling requires legacy, uniform, non-deduplicated validation inputs')
        indices = np.linspace(0, len(candidates) - 1, budget).round().astype(np.int64)
        selected = candidates[indices]
    else:
        selected = select_windows(candidates, budget, strategy=contract['eval_sampling'],
                                  group_ids=candidate_meta['group_ids'] if candidate_meta else None)
    if len(selected):
        unique, inverse = np.unique(selected, return_inverse=True)
        slices = torch.from_numpy(np.asarray(volume[unique], dtype=np.float32) / 255.0).unsqueeze(1)
        resolution = contract['resolution']
        if slices.shape[-2:] != (resolution, resolution):
            slices = F.interpolate(slices, size=(resolution, resolution), mode='bilinear', align_corners=False)
        images = slices[:, 0][torch.from_numpy(inverse)].reshape(len(selected), 3, resolution, resolution)
        images = (images - mean) / std
        valid = torch.full((len(selected),), bool(np.asarray(mask).any()), dtype=torch.bool)
        meta = _window_metadata(selected, mask, source)
    else:
        resolution = contract['resolution']
        images = torch.zeros((1, 3, resolution, resolution), dtype=torch.float32)
        valid = torch.zeros(1, dtype=torch.bool)
        meta = dict(group_ids=np.full(1, -1, np.int64), series_ids=np.full(1, -1, np.int64),
                    plane_ids=np.full(1, -1, np.int64), positions=np.zeros((1, 1), np.float32),
                    position_mask=np.zeros(1, bool))
    batch = dict(images=images.unsqueeze(0), window_mask=valid.unsqueeze(0))
    batch.update({key: torch.from_numpy(value).unsqueeze(0) for key, value in meta.items()})
    details = dict(window_mode=mode, requested_windows=budget, candidate_windows=len(candidates),
                   selected_windows=len(selected), valid_windows=int(valid.sum()),
                   padded_empty_bag=not bool(valid.any()),
                   eval_sampling=contract['eval_sampling'],
                   deduplicate_windows=contract['deduplicate_windows'],
                   legacy_sampling=contract['legacy_sampling'])
    return batch, details


def _autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()


def infer_prepared(model, batch, device='cpu', encode_chunk=32, amp_dtype=None):
    device = torch.device(device)
    if encode_chunk < 1:
        raise ValueError('encode_chunk must be positive')
    if amp_dtype is None and device.type == 'cuda':
        amp_dtype = torch.float16
    metadata = {key: value.to(device) for key, value in batch.items() if key != 'images'}
    features = []
    with torch.inference_mode(), _autocast(device, amp_dtype):
        for start in range(0, batch['images'].shape[1], encode_chunk):
            images = batch['images'][:, start:start + encode_chunk]
            images = images.to(device=device, dtype=amp_dtype or images.dtype, non_blocking=True)
            valid = metadata['window_mask'][:, start:start + encode_chunk]
            features.append(model.encode(images, valid))
        logits = model.head(torch.cat(features, dim=1), **metadata).float()
    probabilities = logits.sigmoid()[0].cpu().numpy()
    if probabilities.shape != (len(raptor.LAB),) or not np.isfinite(probabilities).all():
        raise ValueError('model produced invalid probabilities')
    return probabilities


def contract_amp_dtype(contract, device):
    device = torch.device(device)
    if device.type != 'cuda' or contract.get('eval_amp', 'fp16') == 'off':
        return None
    return torch.bfloat16 if contract['eval_amp'] == 'bf16' else torch.float16


def predict_studies(model, data_root, study_uids, series_records, device='cpu', windows=None,
                    window_mode=None, config=None, encode_chunk=None, preprocess_workers=None):
    if len(set(study_uids)) != len(study_uids) or not study_uids:
        raise ValueError('study UIDs must be nonempty and unique')
    config = dict(config or {})
    contract = checkpoint_input_contract(config)
    encode_chunk = encode_chunk or contract['eval_encode_chunk']
    preprocess_workers = (contract['preprocess_workers'] if preprocess_workers is None
                          else int(preprocess_workers))
    if preprocess_workers < 1:
        raise ValueError('preprocess_workers must be positive')
    directory = Path(data_root) / 'test_series'
    if not directory.is_dir():
        directory = Path(data_root) / 'test_images'
    if not directory.is_dir():
        raise FileNotFoundError('test DICOM directory missing')
    probabilities, empty, valid_counts, input_records = [], [], [], []
    timings = {'preprocess_wait_s': 0.0, 'inference_s': 0.0}

    def preprocess(uid):
        # The reader stores per-series header metadata on its closure, so every
        # worker needs an independent reader instance.
        reader = raptor._make_reader()
        volume, mask, source = raptor.build_study(
            uid, series_records, str(directory), reader, return_metadata=True)
        batch, details = prepare_study_inputs(volume, mask, source, config, windows, window_mode)
        return batch, details, int(mask.sum())

    def prepared_in_order():
        if preprocess_workers == 1:
            for uid in study_uids:
                yield preprocess(uid)
            return
        with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
            pending = {}
            next_submit = 0
            while next_submit < min(preprocess_workers, len(study_uids)):
                pending[next_submit] = executor.submit(preprocess, study_uids[next_submit])
                next_submit += 1
            for index in range(len(study_uids)):
                yield pending.pop(index).result()
                if next_submit < len(study_uids):
                    pending[next_submit] = executor.submit(preprocess, study_uids[next_submit])
                    next_submit += 1

    iterator = iter(prepared_in_order())
    for index, uid in enumerate(study_uids):
        started = time.perf_counter()
        batch, details, valid_count = next(iterator)
        timings['preprocess_wait_s'] += time.perf_counter() - started
        valid_counts.append(valid_count)
        details['study_uid'] = uid
        input_records.append(details)
        if not batch['window_mask'].any():
            empty.append(uid)
            probability = np.full(len(raptor.LAB), .5, np.float32)
        else:
            started = time.perf_counter()
            probability = infer_prepared(
                model, batch, device, encode_chunk, contract_amp_dtype(contract, device))
            timings['inference_s'] += time.perf_counter() - started
        if (probability.shape != (len(raptor.LAB),) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise ValueError(f'invalid model prediction for {uid}')
        probabilities.append(probability)
        if (index + 1) % 100 == 0 or index + 1 == len(study_uids):
            print(f'{index + 1}/{len(study_uids)} studies; empty={len(empty)}', flush=True)
    if len(empty) == len(study_uids):
        raise ValueError('all studies are empty; refusing an invalid all-neutral submission')
    resolved = input_records[0] if input_records else {}
    return np.stack(probabilities), {
        'empty_study_uids': empty, 'valid_slice_counts': valid_counts,
        'window_mode': resolved.get('window_mode'), 'requested_windows': resolved.get('requested_windows'),
        'selected_window_counts': [item['selected_windows'] for item in input_records],
        'candidate_window_counts': [item['candidate_windows'] for item in input_records],
        'eval_sampling': resolved.get('eval_sampling'),
        'deduplicate_windows': resolved.get('deduplicate_windows'),
        'legacy_sampling': resolved.get('legacy_sampling'), 'encode_chunk': encode_chunk,
        'preprocess_workers': preprocess_workers, 'timings': timings,
        'normalization': contract['normalization'], 'resolution': contract['resolution'],
        'output_transform': 'sigmoid probabilities; no rank transform or rounding',
    }
