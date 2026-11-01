"""Conservative inference acceleration with unchanged preprocessing arithmetic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from io import BytesIO
import os
from pathlib import Path
import struct
import time
import warnings
import sys

import numpy as np
import pandas as pd
import pydicom
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from build_coatnet_original_corpus import build_one
from kneemri.coatnet_data import normalization_tensors
from kneemri.dicom import CorruptedPixelDataError, modality_pixels
from kneemri.headers import COLUMNS, _TAG_MAP, dataset_meta, read_slice_meta
from kneemri.pipeline import preprocess_signature, process_study


READ_TAGS = list(_TAG_MAP) + [
    'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'ImageOrientationPatient',
    'ImagePositionPatient', 'PixelSpacing', 'ImageLaterality', 'PixelData',
    'FloatPixelData', 'DoubleFloatPixelData', 'PlanarConfiguration',
    'ExtendedOffsetTable', 'ExtendedOffsetTableLengths',
]
READ_TAG_NUMBERS = {pydicom.datadict.tag_for_keyword(keyword) for keyword in READ_TAGS}
READ_TAG_NUMBERS.add(0x00080005)
LONG_VRS = {'OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'SQ', 'UC', 'UR', 'UT', 'UN', 'SV', 'UV'}
VALID_VRS = {str(value.value) for value in pydicom.valuerep.VR}


PIXEL_DATA_TAG = 0x7FE00010
HEADER_PREFIX_BYTES = 1 << 16


def scan_explicit_vr(payload, stop_at_pixels=False):
    """Explicit-VR little-endian element scan; raises on anything unexpected.

    With ``stop_at_pixels`` the walk ends at the PixelData element header, so a
    bounded prefix of the file is enough to build the metadata row and to learn
    where the pixels start. Values are handed to pydicom's own raw-element
    conversion, so every decoded tag matches ``dcmread``.
    Returns ``(dataset, pixel_offset, pixel_length)``; the offsets are None when
    the scan never reached PixelData.
    """
    if payload[128:132] != b'DICM':
        raise ValueError('no Part 10 preamble')
    offset, elements, syntax = 132, {}, None
    pixel_offset = pixel_length = None
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError('truncated element header')
        group, element = struct.unpack_from('<HH', payload, offset)
        tag = (group << 16) | element
        vr = payload[offset + 4:offset + 6].decode('ascii')
        if vr not in VALID_VRS:
            raise ValueError('unsupported explicit VR')
        if vr in LONG_VRS:
            if offset + 12 > len(payload) or payload[offset + 6:offset + 8] != b'\0\0':
                raise ValueError('invalid long VR header')
            length = struct.unpack_from('<I', payload, offset + 8)[0]
            value_start = offset + 12
        else:
            length = struct.unpack_from('<H', payload, offset + 6)[0]
            value_start = offset + 8
        if length == 0xFFFFFFFF:
            raise ValueError('undefined element length')
        if tag == PIXEL_DATA_TAG and stop_at_pixels:
            if syntax != str(pydicom.uid.ExplicitVRLittleEndian):
                raise ValueError('non-native transfer syntax')
            pixel_offset, pixel_length = value_start, length
            break
        if value_start + length > len(payload):
            raise ValueError('truncated element value')
        if tag == 0x00020010:
            syntax = payload[value_start:value_start + length].rstrip(b'\0 ').decode('ascii')
        if group != 2 and syntax != str(pydicom.uid.ExplicitVRLittleEndian):
            raise ValueError('non-native transfer syntax')
        if tag in READ_TAG_NUMBERS:
            elements[pydicom.tag.Tag(tag)] = pydicom.dataelem.RawDataElement(
                pydicom.tag.Tag(tag), vr, length, payload[value_start:value_start + length],
                value_start, False, True)
        if tag == PIXEL_DATA_TAG:
            pixel_offset, pixel_length = value_start, length
        offset = value_start + length
    if syntax != str(pydicom.uid.ExplicitVRLittleEndian):
        raise ValueError('missing native transfer syntax')
    if stop_at_pixels and pixel_offset is None:
        raise ValueError('prefix ended before PixelData')
    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = syntax
    dataset = pydicom.dataset.FileDataset(None, elements, preamble=payload[:128],
                                           file_meta=meta, is_implicit_VR=False, is_little_endian=True)
    dataset.set_original_encoding(False, True)
    return dataset, pixel_offset, pixel_length


def read_dataset_bytes(payload):
    """Fast explicit-VR little-endian scan, retaining pydicom value conversion."""
    try:
        return scan_explicit_vr(payload)[0]
    except (ValueError, UnicodeError, struct.error):
        return pydicom.dcmread(BytesIO(payload))


def _native_geometry(dataset, length):
    """Shared native-buffer predicate for both the element and offset decoders."""
    syntax = str(getattr(dataset.file_meta, 'TransferSyntaxUID', ''))
    allocated = int(getattr(dataset, 'BitsAllocated', 0))
    stored = int(getattr(dataset, 'BitsStored', 0))
    representation = int(getattr(dataset, 'PixelRepresentation', -1))
    rows, columns = int(dataset.Rows), int(dataset.Columns)
    native = (
        syntax in {'1.2.840.10008.1.2', '1.2.840.10008.1.2.1'}
        and allocated in (8, 16, 32) and 0 < stored <= allocated
        and int(getattr(dataset, 'HighBit', -1)) == stored - 1
        and representation in (0, 1)
        and int(getattr(dataset, 'SamplesPerPixel', 0)) == 1
        and int(getattr(dataset, 'NumberOfFrames', 1) or 1) == 1
        and str(getattr(dataset, 'PhotometricInterpretation', '')) in {'MONOCHROME1', 'MONOCHROME2'}
        and rows > 0 and columns > 0
    )
    expected = rows * columns * (allocated // 8)
    if not native or length not in {expected, expected + expected % 2}:
        return None
    return np.dtype(f"<{'i' if representation else 'u'}{allocated // 8}"), rows, columns, allocated, stored


def _from_buffer(buffer, geometry, offset=0):
    dtype, rows, columns, allocated, stored = geometry
    pixels = np.frombuffer(buffer, dtype=dtype, count=rows * columns, offset=offset)
    if stored < allocated:
        pixels = pixels.copy()
        shift = allocated - stored
        np.left_shift(pixels, shift, out=pixels)
        np.right_shift(pixels, shift, out=pixels)
    return pixels.reshape(rows, columns)


def decode_dataset(dataset, path=''):
    """Use native little-endian monochrome buffers, otherwise pydicom's decoder."""
    try:
        geometry = (_native_geometry(dataset, len(dataset.PixelData))
                    if 'PixelData' in dataset else None)
        pixels = (_from_buffer(dataset.PixelData, geometry) if geometry is not None
                  else dataset.pixel_array)
    except Exception as exc:
        raise CorruptedPixelDataError(f'{path}: {exc}') from exc
    return modality_pixels(dataset, pixels, path)


def decode_at_offset(dataset, payload, offset, length, path=''):
    """Decode straight out of the file buffer using the scanned PixelData offset.

    Identical arithmetic to ``decode_dataset``; it only skips re-parsing the file
    to recover bytes whose position the header scan already recorded.
    """
    try:
        geometry = _native_geometry(dataset, length) if offset is not None else None
        if geometry is None:
            return decode_dataset(read_dataset_bytes(payload), path)
        pixels = _from_buffer(payload, geometry, offset)
    except CorruptedPixelDataError:
        raise
    except Exception as exc:
        raise CorruptedPixelDataError(f'{path}: {exc}') from exc
    return modality_pixels(dataset, pixels, path)


def _read_bytes(path, limit=-1):
    with open(path, 'rb') as handle:
        return handle.read(limit)


class ReadOnceStudy:
    """Bounded, study-local DICOM datasets shared by headers and pixel decoding.

    Each file leaves the filesystem exactly once. One ``read_bytes`` feeds the
    explicit-VR scan, and the resulting dataset already owns its PixelData copy,
    so the pixel pass needs no second ``dcmread``. Header values still come from
    pydicom's own raw-element conversion, so ``metadata`` stays identical to
    ``study_frame``; only the number of reads and parses changes.
    """

    def __init__(self, root, executor):
        self.root = Path(root)
        self.executor = executor
        self.datasets = {}
        self.decoded = {}
        self.payloads = {}
        self.series = {}
        paths = sorted(self.root.glob('*/*.dcm'))
        for path in paths:
            self.series.setdefault(str(path.parent), []).append(str(path))
        # Threads read; this thread parses. Parsing is GIL-bound, so doing it in
        # the reader threads only makes them contend: measured 0.96 ms/file
        # serial against 3.15 ms/file across eight threads on four cores.
        prefixes = [executor.submit(_read_bytes, str(path), HEADER_PREFIX_BYTES)
                    for path in paths]
        rows = []
        for path, prefix in zip(paths, prefixes):
            path, row, entry = self._read_header(path, prefix.result())
            if row is not None:
                rows.append(row)
                self.datasets[str(path)] = entry
        self.metadata = pd.DataFrame(rows, columns=COLUMNS)

    def _read_header(self, path, prefix):
        try:
            try:
                entry = scan_explicit_vr(prefix, stop_at_pixels=True)
            except (ValueError, UnicodeError, struct.error):
                # Anything unusual (compressed pixels, a header longer than the
                # prefix) falls back to reading and parsing the whole file.
                entry = (read_dataset_bytes(Path(path).read_bytes()), None, None)
            try:
                row = dataset_meta(entry[0], str(path), str(self.root))
            except Exception as exc:
                row = read_slice_meta(str(path), str(self.root))
                entry = exc
            if not row['study_uid'] or not row['series_uid']:
                raise ValueError('missing study/series UID')
            return path, row, entry
        except Exception as exc:
            warnings.warn(f'skipped unreadable DICOM header {path}: {type(exc).__name__}: {exc}',
                          RuntimeWarning, stacklevel=2)
            return path, None, None

    def _payload(self, key):
        """Read one file, prefetching the rest of its series concurrently.

        ``_decode_rows`` walks a whole series in order, so the first slice of a
        series is the point where the remaining reads can be overlapped.
        """
        if key not in self.payloads:
            try:
                for sibling in self.series.get(str(Path(key).parent), [key]):
                    if sibling not in self.payloads:
                        self.payloads[sibling] = self.executor.submit(_read_bytes, sibling)
            except RuntimeError:
                # The pool is gone; a serial read still returns the same bytes.
                return _read_bytes(key)
        return self.payloads.pop(key).result()

    def decode(self, path):
        key = str(path)
        if key not in self.decoded:
            entry = self.datasets.get(key)
            if not isinstance(entry, tuple):
                # Header parsing already failed for this slice; re-read it the
                # reference way rather than guessing at the retained object.
                self.payloads.pop(key, None)
                self.decoded[key] = decode_dataset(
                    pydicom.dcmread(key, specific_tags=READ_TAGS), key)
                return self.decoded[key]
            dataset, offset, length = entry
            self.decoded[key] = decode_at_offset(dataset, self._payload(key), offset, length, key)
            self.datasets.pop(key, None)
        return self.decoded[key]


def result_arrays(result, pipeline):
    """Expose precisely the serialized fields consumed by the window builder."""
    arrays = {'schema_version': np.asarray(2),
              'preprocess_signature': np.asarray(preprocess_signature(pipeline))}
    for slot_id, ranks in result.slot_ranks.items():
        for rank, slot in enumerate(ranks):
            prefix = f'slot_{slot_id}_s{rank}'
            valid = slot.valid and slot.volume is not None
            arrays[prefix + '_valid'] = np.asarray(int(valid), dtype=np.uint8)
            if not valid:
                continue
            arrays.update({
                prefix + '_volume': slot.volume,
                prefix + '_source_indices': slot.source_indices if slot.source_indices is not None else np.arange(len(slot.volume)),
                prefix + '_z': slot.z,
                prefix + '_geometry_valid': np.asarray(slot.geometry_valid),
                prefix + '_output_spacing_mm_per_px': np.asarray(slot.output_spacing_mm_per_px, dtype=np.float32),
                prefix + '_meta_codes': slot.meta_codes,
                prefix + '_quality': slot.quality if slot.quality is not None else np.zeros(8, np.float32),
            })
    return arrays


_READER_POOLS = {}


def process_reader_pool(threads):
    """One reader thread pool per process and size.

    The raw ``read_bytes`` releases the GIL, so threads overlap the competition
    input's per-reader bandwidth limit; the explicit-VR scan that follows is
    GIL-bound and gains nothing, which is why the pool only wraps the reads.
    """
    threads = int(threads)
    if threads < 1:
        raise ValueError('reader threads must be positive')
    # Key on the pid: a forked child inherits the parent's executor object, but
    # not its worker threads, so reusing it would queue work nothing can run.
    key = (os.getpid(), threads)
    if key not in _READER_POOLS:
        _READER_POOLS.clear()
        _READER_POOLS[key] = ThreadPoolExecutor(max_workers=threads,
                                                thread_name_prefix='dicom-read')
    return _READER_POOLS[key]


def prepare_study_fast(uid, records, data_root, pipeline, contract, reader_pool):
    started = time.monotonic()
    pipeline = copy.deepcopy(pipeline)
    pipeline.study_root = str(Path(data_root) / 'test_series' / uid)
    reader = ReadOnceStudy(pipeline.study_root, reader_pool)
    read_done = time.monotonic()
    if reader.metadata.empty:
        raise ValueError(f'no readable DICOM headers for {uid}')
    files = len(reader.datasets)
    result = process_study(uid, pipeline, reader.metadata, records, reader.decode)
    process_done = time.monotonic()
    if result.failed:
        raise ValueError(f'preprocessing failed for {uid}: {result.error}')
    arrays = build_one(0, uid, result_arrays(result, pipeline), contract['image'],
                       contract['source_preprocess_signature'], contract.get('crop_fov_mm'))
    stats = dict(read_s=read_done - started, process_s=process_done - read_done,
                 windows_s=time.monotonic() - process_done, files=files,
                 decoded_files=len(reader.decoded))
    return uid, arrays, stats


def compact_windows(arrays):
    """Keep only the valid windows, so the worker sends no padding rows.

    The selection is exactly ``np.flatnonzero(mask)``, the same rows the
    inference engine would have taken from the padded corpus arrays.
    """
    _, windows, mask, series, groups, planes, positions, reliable, _ = arrays
    selected = np.flatnonzero(mask)
    return {
        'count': int(selected.size),
        'windows': np.ascontiguousarray(windows[selected]),
        'series_ids': series[selected].astype(np.int64),
        'group_ids': groups[selected].astype(np.int64),
        'plane_ids': planes[selected].astype(np.int64),
        'positions': positions[selected].astype(np.float32),
        'position_mask': reliable[selected].astype(bool),
    }


_WORKER = {}


def init_preprocess_worker(pipeline, contract, data_root, reader_threads):
    """Pin one preprocessing process to a single OpenCV thread and cache config.

    Preprocessing is GIL-bound Python, so the deployed runtime fans studies out
    over processes; each of them owns the reader thread pool that overlaps IO.
    Workers never touch CUDA, which is why the pool is forked before the parent
    initialises a device.
    """
    import cv2
    cv2.setNumThreads(1)
    _WORKER.update(pipeline=pipeline, contract=contract, data_root=Path(data_root),
                   reader_threads=int(reader_threads))


def preprocess_worker(uid, records):
    uid, arrays, stats = prepare_study_fast(
        uid, records, _WORKER['data_root'], _WORKER['pipeline'], _WORKER['contract'],
        process_reader_pool(_WORKER['reader_threads']))
    return uid, compact_windows(arrays), stats


class Uint8Inference:
    """Transfer compact pixels and expand through an exact FP32 normalization LUT."""

    def __init__(self, model, cfg, device, chunk_size=16):
        if cfg['eval_windows'] != 'all' or chunk_size < 1:
            raise ValueError('uint8 inference requires all windows and a positive chunk')
        self.model, self.cfg = model, cfg
        self.device, self.chunk_size = torch.device(device), chunk_size
        values = torch.from_numpy(np.arange(256, dtype=np.float32) / 255.0).view(1, 1, 1, 256)
        mean, std = normalization_tensors(cfg['normalization'])
        self.lookup = ((values - mean) / std)[0, :, 0].to(self.device)
        self.channels = torch.arange(3, device=self.device).view(1, 3, 1, 1)

    def logits(self, arrays):
        _, windows, mask, series, groups, planes, positions, reliable, _ = arrays
        if windows.dtype != np.uint8 or windows.shape[-2:] != (self.cfg['resolution'],) * 2:
            raise ValueError('uint8 inference requires native-resolution uint8 windows')
        selected = np.flatnonzero(mask)
        count = len(selected)
        if count == 0:
            return torch.zeros(1, self.model.n)
        host = torch.from_numpy(np.ascontiguousarray(windows[selected]))
        if self.device.type == 'cuda':
            host = host.pin_memory()
        pixels = host.to(self.device, non_blocking=True)
        metadata = dict(
            window_mask=torch.ones((1, count), device=self.device, dtype=torch.bool),
            group_ids=torch.as_tensor(groups[selected].astype(np.int64)[None], device=self.device),
            series_ids=torch.as_tensor(series[selected].astype(np.int64)[None], device=self.device),
            plane_ids=torch.as_tensor(planes[selected].astype(np.int64)[None], device=self.device),
            positions=torch.as_tensor(positions[selected][None, :, None], device=self.device),
            position_mask=torch.as_tensor(reliable[selected].astype(bool)[None], device=self.device),
        )
        features = []
        for start in range(0, count, self.chunk_size):
            stop = start + self.chunk_size
            images = self.lookup[self.channels, pixels[start:stop].long()].unsqueeze(0)
            features.append(self.model.encode(images, metadata['window_mask'][:, start:stop],
                                              metadata['group_ids'][:, start:stop]))
        logits = self.model.head(torch.cat(features, dim=1), **metadata).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite optimized inference logits')
        return logits.cpu()
