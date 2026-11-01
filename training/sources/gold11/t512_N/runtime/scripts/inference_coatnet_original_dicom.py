#!/usr/bin/env python3
"""Stream raw DICOM studies through one original-neighbour CoAtNet checkpoint."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import train_coatnet_ablation as trainer
from build_coatnet_original_corpus import SLOTS, SPAN
from inference_coatnet_ablation import checkpoint_config, training_manifest, preprocessing_contract
from coatnet_fast_runtime import (compact_windows, init_preprocess_worker, prepare_study_fast,
                                  preprocess_worker, process_reader_pool)
from kneemri.coatnet_data import normalization_tensors
from kneemri.coatnet_model import build_model
from kneemri.coatnet_research import build_research_model
from kneemri.coatnet_training_data import coatnet_collate
from kneemri.config import load_config
from kneemri.labels import LABEL_COLS
from kneemri.pipeline import preprocess_signature


def prepare_study(uid, records, data_root, pipeline, contract, reader_threads=8):
    """Single-read DICOM path: headers and pixels share one pass over each file.

    Window arrays are identical to the reference ``study_frame`` +
    ``decode_pixel_array`` path; only the number of file reads and DICOM parses
    differs. ``tests/test_coatnet_fast_runtime.py`` guards both equalities.
    """
    uid, arrays, _ = prepare_study_fast(uid, records, Path(data_root), pipeline, contract,
                                        process_reader_pool(reader_threads))
    return uid, arrays


def arrays_to_batch(uid, arrays, cfg):
    _, windows, mask, series, groups, planes, positions, reliable, _ = arrays
    selected = np.flatnonzero(mask)
    if cfg['eval_windows'] != 'all':
        raise ValueError('streaming runtime requires the all-window selection contract')
    if len(selected):
        images = torch.from_numpy(windows[selected].astype(np.float32) / 255.0)
        if images.shape[-1] != cfg['resolution']:
            images = torch.nn.functional.interpolate(images, size=(cfg['resolution'], cfg['resolution']),
                                                     mode='bilinear', align_corners=False)
        mean, std = normalization_tensors(cfg['normalization'])
        images = (images - mean) / std
        valid = torch.ones(len(selected), dtype=torch.bool)
        groups = groups[selected].astype(np.int64)
        series = series[selected].astype(np.int64)
        planes = planes[selected].astype(np.int64)
        positions = positions[selected, None]
        reliable = reliable[selected].astype(bool)
    else:
        images = torch.zeros(1, 3, cfg['resolution'], cfg['resolution'])
        valid = torch.zeros(1, dtype=torch.bool)
        groups = series = planes = np.full(1, -1, dtype=np.int64)
        positions = np.zeros((1, 1), dtype=np.float32)
        reliable = np.zeros(1, dtype=bool)
    return coatnet_collate([dict(
        images=images, targets=torch.zeros(12), window_mask=valid,
        group_ids=torch.from_numpy(groups.copy()), series_ids=torch.from_numpy(series.copy()),
        plane_ids=torch.from_numpy(planes.copy()), positions=torch.from_numpy(positions.copy()),
        position_mask=torch.from_numpy(reliable.copy()), study_id=uid,
        label_source='inference', provenance={'study_id': uid, 'candidate_count': len(selected)},
    )])


class ExactUint8Inference:
    """Keep the serialized uint8 contract and normalize on the active device.

    ``(v / 255 - mean) / std`` is evaluated directly in FP32 instead of gathering
    from a 256-entry table. Both forms apply the same three FP32 operations to
    the same inputs, so every value is bit-identical, while the direct form drops
    the int64 index tensor that was eight bytes per subpixel.
    """

    def __init__(self, model, cfg, device, chunk_size):
        if cfg['eval_windows'] != 'all' or chunk_size < 1:
            raise ValueError('optimized inference requires all windows and a positive chunk')
        self.model = model
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.resolution = int(cfg['resolution'])
        mean, std = normalization_tensors(cfg['normalization'])
        self.mean, self.std = mean.to(self.device), std.to(self.device)

    def _normalize(self, pixels):
        images = pixels.to(torch.float32)
        images.div_(255.0).sub_(self.mean).div_(self.std)
        return images.unsqueeze(0)

    def _normalize_batch(self, pixels):
        images = pixels.to(torch.float32)
        images.div_(255.0).sub_(self.mean).div_(self.std)
        return images

    def _forward(self, pixels, metadata, count):
        features = []
        for start in range(0, count, self.chunk_size):
            stop = start + self.chunk_size
            images = self._normalize(pixels[start:stop])
            features.append(self.model.encode(images, metadata['window_mask'][:, start:stop],
                                              metadata['group_ids'][:, start:stop]))
        return self.model.head(torch.cat(features, dim=1), **metadata).float()

    def _to_device(self, windows):
        host = torch.from_numpy(windows)
        if self.device.type == 'cuda':
            host = host.pin_memory()
        return host.to(self.device, non_blocking=True)

    def _metadata(self, count, groups, series, planes, positions, reliable):
        as_tensor = lambda value: torch.as_tensor(value[None], device=self.device)
        return {
            'window_mask': torch.ones((1, count), device=self.device, dtype=torch.bool),
            'group_ids': as_tensor(groups), 'series_ids': as_tensor(series),
            'plane_ids': as_tensor(planes), 'position_mask': as_tensor(reliable),
            'positions': torch.as_tensor(positions[None, :, None], device=self.device),
        }

    def logits_compact(self, payload):
        """Score a worker payload that already holds only the valid windows.

        Returns raw logits and never synchronises: the caller checks finiteness
        once per device after the loop, so both GPUs stay queued.
        """
        count = payload['count']
        windows = payload['windows']
        if count and (windows.dtype != np.uint8
                      or windows.shape[-2:] != (self.resolution, self.resolution)):
            raise ValueError('optimized inference requires native-resolution uint8 windows')
        if not count:
            return torch.zeros(1, self.model.n, device=self.device)
        metadata = self._metadata(count, payload['group_ids'], payload['series_ids'],
                                  payload['plane_ids'], payload['positions'],
                                  payload['position_mask'])
        return self._forward(self._to_device(windows), metadata, count)

    def logits_compact_batch(self, payloads):
        """Score several studies in one backbone stream.

        The old loop launched the CoAtNet for every study separately.  That
        leaves a 384px T4 under-filled and pays the launch/transfer overhead
        once per study.  This path concatenates valid windows from a small
        study batch, encodes them together, then applies the existing head to
        each study's feature slice.  No window is dropped or mixed across
        studies, and the head still receives exactly the same metadata and
        window order as the single-study path.

        A batch is padded only in the inexpensive window dimension.  The mask
        keeps padding out of the backbone, while the leading study dimension
        gives the convolution kernels a larger image batch. ``model.eval()``
        makes the per-window backbone independent of the other studies in the
        batch (including BatchNorm).
        """
        if not payloads:
            return torch.empty((0, self.model.n), device=self.device)
        counts = [int(payload['count']) for payload in payloads]
        if any(count < 0 for count in counts):
            raise ValueError('compact payload count must be nonnegative')
        total = sum(counts)
        if total == 0:
            return torch.zeros(len(payloads), self.model.n, device=self.device)

        slots = ((max(counts) + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        batch_size = len(payloads)
        windows = np.zeros((batch_size, slots, 3, self.resolution, self.resolution), dtype=np.uint8)
        valid_host = np.zeros((batch_size, slots), dtype=bool)
        groups_host = np.full((batch_size, slots), -1, dtype=np.int64)
        for row, (payload, count) in enumerate(zip(payloads, counts)):
            if not count:
                continue
            study_windows = np.ascontiguousarray(payload['windows'])
            if (study_windows.dtype != np.uint8
                    or study_windows.shape != (count, 3, self.resolution, self.resolution)):
                raise ValueError('optimized inference requires native-resolution uint8 windows')
            windows[row, :count] = study_windows
            valid_host[row, :count] = True
            groups_host[row, :count] = np.asarray(payload['group_ids'], dtype=np.int64)
        pixels = self._to_device(windows)
        encoded = []
        for start in range(0, slots, self.chunk_size):
            stop = start + self.chunk_size
            images = self._normalize_batch(pixels[:, start:stop])
            valid = torch.as_tensor(valid_host[:, start:stop], device=self.device)
            group_ids = torch.as_tensor(groups_host[:, start:stop], device=self.device)
            encoded.append(self.model.encode(images, valid, group_ids))
        features = torch.cat(encoded, dim=1)

        outputs = []
        for row, (payload, count) in enumerate(zip(payloads, counts)):
            if count == 0:
                outputs.append(torch.zeros(1, self.model.n, device=self.device))
                continue
            metadata = self._metadata(
                count, payload['group_ids'], payload['series_ids'], payload['plane_ids'],
                payload['positions'], payload['position_mask'])
            outputs.append(self.model.head(features[row:row + 1, :count], **metadata).float())
        return torch.cat(outputs, dim=0)

    def logits(self, arrays):
        payload = compact_windows(arrays)
        logits = self.logits_compact(payload)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite optimized inference logits')
        return logits


def usable_cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def inference_devices(device, requested=None):
    """Every visible GPU that is identical to the primary one.

    Sharding studies over replicas of one checkpoint keeps a single model and a
    single set of weights. Devices that differ in name or capability are excluded
    because only identical hardware guarantees identical arithmetic.
    """
    if device.type != 'cuda':
        if requested not in (None, 1):
            raise ValueError('multiple devices require CUDA')
        return [device]
    primary = device.index or 0
    reference = (torch.cuda.get_device_name(primary), torch.cuda.get_device_capability(primary))
    usable = [primary] + [index for index in range(torch.cuda.device_count())
                          if index != primary
                          and (torch.cuda.get_device_name(index),
                               torch.cuda.get_device_capability(index)) == reference]
    if requested is not None:
        if not 1 <= requested <= len(usable):
            raise ValueError(f'requested {requested} devices, {len(usable)} identical GPUs visible')
        usable = usable[:requested]
    return [torch.device('cuda', index) for index in usable]


def replica(model, device):
    return (model if next(model.parameters()).device == device
            else copy.deepcopy(model)).to(device).eval()


class PreparedStudies:
    """Bounded prefetch over a forked process pool, with in-parent recovery.

    A worker killed by the host (out of memory, for instance) breaks the whole
    pool, which would otherwise lose every remaining study. The failed study is
    retried in this process and the pool is rebuilt for the rest; retries run the
    same function on the same files, so recovered studies are not approximations.
    """

    def __init__(self, ids, records, pipeline, contract, data_root, *, workers,
                 reader_threads, depth, max_restarts=4):
        self.ids, self.records = ids, records
        self.initargs = (pipeline, contract, str(data_root), reader_threads)
        self.workers, self.reader_threads = workers, reader_threads
        self.depth, self.max_restarts = depth, max_restarts
        self.pipeline, self.contract, self.data_root = pipeline, contract, data_root
        self.recovered, self.restarts = [], 0
        self.pending = {}
        self.pool = self._pool()
        for index in range(depth):
            self._submit(index)

    def _pool(self):
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context('fork'),
                                   initializer=init_preprocess_worker, initargs=self.initargs)

    def _submit(self, index):
        if 0 <= index < len(self.ids) and index not in self.pending:
            uid = self.ids[index]
            self.pending[index] = self.pool.submit(preprocess_worker, uid, self.records[uid])

    def _in_parent(self, index):
        uid, arrays = prepare_study(self.ids[index], self.records[self.ids[index]],
                                    self.data_root, self.pipeline, self.contract,
                                    self.reader_threads)
        return uid, compact_windows(arrays)

    def get(self, index):
        try:
            uid, payload, _ = self.pending.pop(index).result()
        except Exception as error:
            if self.restarts >= self.max_restarts:
                raise
            self.restarts += 1
            self.recovered.append(dict(study_index=index, study_id=self.ids[index],
                                       error=f'{type(error).__name__}: {error}'))
            outstanding = sorted(self.pending)
            self.pending.clear()
            self.pool.shutdown(cancel_futures=True)
            self.pool = self._pool()
            uid, payload = self._in_parent(index)
            for other in outstanding:
                self._submit(other)
        self._submit(index + self.depth)
        if uid != self.ids[index]:
            raise ValueError('study ordering changed during preprocessing')
        return payload

    def shutdown(self):
        self.pool.shutdown(cancel_futures=True)


def collect_probabilities(logits, engines, assignments=None):
    """Sigmoid per device, then reassemble in submission order.

    Finiteness is checked on the raw logits, as the per-study path did, but once
    per device so the scoring loop never blocks on a device synchronisation.
    """
    if assignments is None:
        # Preserve the original single-study round-robin contract for callers
        # that do not pack studies into batches.
        assignments = [index % len(engines) for index in range(len(logits))]
    if len(assignments) != len(logits) or any(
            not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(engines)
            for index in assignments):
        raise ValueError('engine assignments must align with logits')
    values = np.empty((len(logits), len(LABEL_COLS)), dtype=np.float32)
    for offset, engine in enumerate(engines):
        rows = [row for row, assignment in enumerate(assignments) if int(assignment) == offset]
        if not rows:
            continue
        block = torch.cat([logits[row] for row in rows], dim=0)
        if not torch.isfinite(block).all():
            raise FloatingPointError('nonfinite optimized inference logits')
        values[rows] = block.sigmoid().cpu().numpy()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-manifest', type=Path)
    parser.add_argument('--preprocess-config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--encode-chunk', type=int, default=16)
    parser.add_argument('--study-batch-size', type=int, default=2,
                        help='studies packed into one backbone stream; 2 fits a 16GB T4')
    parser.add_argument('--eval-amp', choices=['bf16', 'fp16', 'off'])
    parser.add_argument('--cudnn-benchmark', choices=['on', 'off'], default='off',
                        help='autotune fixed-shape convolutions; off avoids cold-start search and is the deterministic baseline')
    parser.add_argument('--workers', type=int,
                        help='preprocessing processes; defaults to the usable CPU count')
    parser.add_argument('--reader-threads', type=int, default=1,
                        help='concurrent DICOM reads inside each preprocessing process; '
                             'the process pool already reads from several studies at once, '
                             'and extra threads measured slower on a warm page cache')
    parser.add_argument('--devices', type=int,
                        help='CUDA devices to score on; defaults to every identical visible GPU')
    # Resolved after the preprocessing pool is forked so that argument parsing
    # cannot initialise CUDA in the parent before the children exist.
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    # Kaggle's T4 instance exposes four usable cores; never fan out further than
    # that by default, so a large shared host is not oversubscribed either.
    workers = args.workers if args.workers else min(4, usable_cpu_count())
    if (not 1 <= workers <= 32 or args.encode_chunk < 1
            or not 1 <= args.study_batch_size <= 16
            or not 1 <= args.reader_threads <= 64):
        raise ValueError('workers 1..32, study batch 1..16, reader threads 1..64 and encode-chunk must be positive')
    started = time.monotonic()
    import cv2
    cv2.setNumThreads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    research = 'scripts/train_coatnet_research.py' in checkpoint['receipt']['sources']
    runtime_trainer = trainer
    if research:
        import train_coatnet_research as runtime_trainer
        cfg = runtime_trainer.normalize_config(checkpoint['config'])
        if (checkpoint.get('lab') != LABEL_COLS or checkpoint.get('arch') != cfg['arch']
                or checkpoint.get('res') != cfg['resolution']
                or cfg['corpus_format'] != 'original_neighbors'):
            raise ValueError('research checkpoint model contract differs from its config')
    else:
        cfg = checkpoint_config(checkpoint)
    if cfg['eval_windows'] != 'all' or cfg['eval_sampling'] != 'uniform':
        raise ValueError('expected a checkpoint selected with all windows')
    torch.set_num_threads(min(4, cfg['num_threads']))
    # Resolve the device before selecting CUDA-only math flags.  TF32 is
    # ignored on pre-Ampere GPUs (including Kaggle's T4), while it speeds the
    # same kernels on local Ampere/Hopper validation hardware.
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == 'on'
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    runtime_trainer.seed_all(cfg['seed'])
    contract = preprocessing_contract(training_manifest(checkpoint, args.training_manifest))
    pipeline = load_config(str(args.preprocess_config))
    pipeline.blacklist = []
    if preprocess_signature(pipeline) != contract['source_preprocess_signature']:
        raise ValueError('DICOM preprocessing configuration differs from training')
    if (contract['slots'] != [list(slot) for slot in SLOTS] or contract['span'] != list(SPAN)
            or contract['neighbor_gap_ratio'] != 1.8 or contract['source_schema'] != 2):
        raise ValueError('window builder differs from training')
    ids = pd.read_csv(args.data_root / 'test.csv').StudyInstanceUID.astype(str).tolist()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('submission IDs must be nonempty and unique')
    table = pd.read_csv(args.data_root / 'test_series.csv')
    records = {uid: rows for uid, rows in table.groupby('StudyInstanceUID')}
    if not set(ids) <= set(records):
        raise ValueError('test series table misses requested studies')
    sample = args.data_root / 'sample_submission.csv'
    if sample.exists() and pd.read_csv(sample, nrows=0).columns.tolist() != ['StudyInstanceUID', *LABEL_COLS]:
        raise ValueError('submission label order differs from sample')
    # Fork the preprocessing processes before the parent creates a CUDA context.
    # Children inherit the parsed config and never touch a device.
    prepared = PreparedStudies(ids, records, pipeline, contract, args.data_root,
                               workers=workers, reader_threads=args.reader_threads,
                               depth=workers + 2)
    try:
        if args.eval_amp:
            cfg['eval_amp'] = args.eval_amp
        elif device.type == 'cuda' and cfg['eval_amp'] == 'bf16' and torch.cuda.get_device_capability(device)[0] < 8:
            # T4 deployment uses the same weights at an explicitly recorded precision.
            cfg['eval_amp'] = 'fp16'
        model = (build_research_model(cfg, pretrained=False) if research else
                 build_model(cfg, pretrained=False)).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        epoch = checkpoint['epoch'] + 1
        del checkpoint
        devices = inference_devices(device, args.devices)
        dtype = runtime_trainer.select_eval_amp(cfg, device)
        engines = [ExactUint8Inference(replica(model, target), cfg, target, args.encode_chunk)
                   for target in devices]
        logits, counts, assignments = [None] * len(ids), [], []
        for batch_start in range(0, len(ids), args.study_batch_size):
            batch_indices = list(range(batch_start, min(len(ids), batch_start + args.study_batch_size)))
            payloads = [prepared.get(i) for i in batch_indices]
            engine_index = (batch_start // args.study_batch_size) % len(engines)
            engine = engines[engine_index]
            with torch.inference_mode(), runtime_trainer.amp_context(engine.device, dtype):
                batch_logits = engine.logits_compact_batch(payloads)
            for row, index in enumerate(batch_indices):
                logits[index] = batch_logits[row:row + 1]
                counts.append(payloads[row]['count'])
                assignments.append(engine_index)
            done = batch_indices[-1] + 1
            if done % 25 < args.study_batch_size or done == len(ids):
                print(json.dumps({'studies': done, 'total': len(ids), 'elapsed_s': time.monotonic() - started}), flush=True)
        values = collect_probabilities(logits, engines, assignments)
    finally:
        prepared.shutdown()
    result = pd.DataFrame(values, columns=LABEL_COLS)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('nonfinite submission probabilities')
    result.insert(0, 'StudyInstanceUID', ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.csv.tmp')
    result.to_csv(temporary, index=False)
    temporary.replace(args.out)
    receipt = dict(checkpoint_sha256=runtime_trainer.hash_file(args.checkpoint), checkpoint_epoch=epoch,
                   input=contract, normalization=cfg['normalization'], eval_amp=cfg['eval_amp'],
                   inference_input='uint8_device_fp32_exact', persistent_autocast=True,
                   eval_windows='all', encode_chunk=args.encode_chunk,
                   study_batch_size=args.study_batch_size, workers=workers,
                   reader_threads=args.reader_threads, preprocess_executor='process',
                   dicom_reads_per_file=1, recovered_studies=prepared.recovered,
                   preprocess_pool_restarts=prepared.restarts,
                   studies=len(ids), candidate_counts=counts,
                   ensemble=False, tta=False, model_count=1, checkpoint_count=1,
                   submission_sha256=runtime_trainer.hash_file(args.out), elapsed_s=time.monotonic() - started)
    receipt['hardware'] = dict(device=str(device), torch_version=str(torch.__version__),
                               cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                               cudnn_benchmark=torch.backends.cudnn.benchmark,
                               tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                               cpu_count=os.cpu_count(), usable_cpu_count=usable_cpu_count(),
                               visible_cuda_devices=(torch.cuda.device_count()
                                                     if device.type == 'cuda' else 0),
                               scoring_devices=[str(engine.device) for engine in engines])
    if device.type == 'cuda':
        receipt['hardware'].update(name=torch.cuda.get_device_name(device),
                                    capability=list(torch.cuda.get_device_capability(device)))
    args.out.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
