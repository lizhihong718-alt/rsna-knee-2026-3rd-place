"""KMAR image-only representation learning; clinical checkmarks are audited separately."""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def normalized_name(value):
    value = Path(str(value).strip().strip("'\"")).name.lower()
    value = value.replace('transection', 'transverse').replace('axial', 'transverse')
    return re.sub(r'_(\d+)\.0(?=\.nii\.gz$)', r'_\1', value)


def infer_laterality(description):
    match = re.search(r'_([RL])$', str(description).strip().strip("'\""), re.I)
    return match.group(1).upper() if match else ''


def orient_nifti(image, laterality=''):
    """NIfTI x/y/z -> slice/row/column; reproduce RSNA IOP/LPS flip policy."""
    from kneemri.config import LateralityConfig
    from kneemri.laterality import plan_orientation
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim != 3 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError('expected a finite 3D MRI with at least two slices')
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('physical crop requires explicit millimetre units')
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or not (spacing > 0).all():
        raise ValueError('invalid NIfTI affine spacing')
    directions = np.diag([-1., -1., 1.]) @ image.affine[:3, :3] / spacing
    normal = np.cross(directions[:, 0], directions[:, 1])
    normal /= np.linalg.norm(normal)
    plane = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}[int(np.abs(normal).argmax())]
    volume = np.transpose(array, (2, 1, 0))
    # First establish the same ascending IPP-dot-IOP-normal order as RSNA.
    source_reverse = float(np.dot(normal, directions[:, 2])) < 0
    if source_reverse:
        volume = volume[::-1]
    decision = plan_orientation(np.r_[directions[:, 0], directions[:, 1]],
                                plane, laterality, LateralityConfig())
    if decision.horizontal_flip:
        volume = volume[..., ::-1]
    if decision.reverse_stack:
        volume = volume[::-1]
    return np.ascontiguousarray(volume), spacing, {
        'plane': plane, 'laterality': laterality,
        'source_stack_reversed': source_reverse,
        'horizontal_flip': decision.horizontal_flip,
        'canonical_stack_reversed': decision.reverse_stack,
        'warnings': decision.warnings,
    }


def preprocess_volume(image, laterality='', resolution=384, fov_mm=140.):
    """Match v152's full-FOV cache -> uint8 -> physical140mm -> resize stages."""
    import cv2
    from kneemri.crop import plan_physical_crop, apply_crop_resize, normalize_intensity
    cv2.setNumThreads(1)
    volume, spacing, orientation = orient_nifti(image, laterality)
    height, width = volume.shape[1:]
    field = max(height * spacing[1], width * spacing[0])
    plan = plan_physical_crop(spacing[1], spacing[0], height, width, field)
    full = np.stack([apply_crop_resize(frame, plan, resolution, 'auto') for frame in volume])
    full, intensity = normalize_intensity(full, 1., 99., 'foreground')
    if intensity.get('empty') or intensity.get('degenerate'):
        raise ValueError('empty or degenerate MRI intensity distribution')
    full = np.clip(full * 255. + .5, 0, 255).astype(np.uint8)
    output_spacing = field / resolution
    target = max(1, round(fov_mm / output_spacing))
    if target > 4 * resolution:
        raise ValueError('implausible physical field')
    pad = max(0, target - resolution)
    full = np.pad(full, ((0, 0), (pad // 2, pad - pad // 2), (pad // 2, pad - pad // 2)))
    top = (full.shape[1] - target) // 2
    full = full[:, top:top + target, top:top + target]
    output = np.stack([cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
                       for frame in full])
    return output, dict(orientation=orientation, intensity=intensity,
                       source_shape=list(image.shape), source_spacing_mm=spacing.tolist(),
                       full_field_mm=float(field), physical_crop_mm=fov_mm)


class KMARWindows(Dataset):
    def __init__(self, root, split, seed=2026, windows_per_volume=8):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.records = [r for r in self.manifest['records'] if r['split'] == split]
        self.seed = seed
        self.windows_per_volume = windows_per_volume
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.records) * self.windows_per_volume

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        volume_index = index % len(self.records)
        record = self.records[volume_index]
        path = record['cache_path']
        if path not in self.cache:
            self.cache[path] = np.load(self.root / path, mmap_mode='r')
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        volume = self.cache[path]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
        count = len(volume)
        lo, hi = round(.06 * (count - 1)), round(.94 * (count - 1))
        center = int(rng.integers(lo, hi + 1))
        indices = np.clip([center - 1, center, center + 1], 0, count - 1)
        return torch.from_numpy(np.array(volume[indices], copy=True)), volume_index


class VolumeBatchSampler(Sampler):
    """Each minibatch contains distinct volumes; all volumes get equal draws."""
    def __init__(self, dataset, batch_size, seed=2026):
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
        if batch_size < 2 or len(dataset.records) < batch_size:
            raise ValueError('VICReg requires at least two distinct volumes per batch')

    def __len__(self):
        return self.dataset.windows_per_volume * math.ceil(len(self.dataset.records) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset.records)
        rng = np.random.default_rng(self.seed + self.epoch)
        for repetition in range(self.dataset.windows_per_volume):
            order = rng.permutation(n)
            batches = [order[start:start+self.batch_size] for start in range(0, n, self.batch_size)]
            if len(batches[-1]) == 1:
                batches[-1] = np.r_[batches[-2][-1], batches[-1]]
                batches[-2] = batches[-2][:-1]
            for batch in batches:
                yield [(self.epoch, repetition * n + int(i)) for i in batch]


def augment_views(images):
    """Shared geometry across true neighbouring channels; no RGB colour jitter."""
    x = images.float() / 255.
    n = len(x)
    angle = torch.empty(n, device=x.device).uniform_(-7., 7.) * math.pi / 180
    scale = torch.empty(n, device=x.device).uniform_(.9, 1.05)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = angle.cos() * scale
    theta[:, 0, 1] = -angle.sin() * scale
    theta[:, 1, 0] = angle.sin() * scale
    theta[:, 1, 1] = angle.cos() * scale
    theta[:, :, 2] = torch.empty(n, 2, device=x.device).uniform_(-.04, .04)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                      align_corners=False, padding_mode='zeros')
    gamma = torch.empty(n, 1, 1, 1, device=x.device).uniform_(.85, 1.15)
    gain = torch.empty_like(gamma).uniform_(.9, 1.1)
    x = (x.clamp(0, 1).pow(gamma) * gain).clamp(0, 1)
    return x * 2 - 1


def vicreg_loss(a, b):
    """Single-device VICReg, actual batch denominator, all statistics in FP32."""
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError('VICReg expects paired [batch>=2, features] tensors')
    a, b = a.float(), b.float()
    invariance = F.mse_loss(a, b)
    x, y = a - a.mean(0), b - b.mean(0)
    std_a = torch.sqrt(x.var(0, unbiased=True) + 1e-4)
    std_b = torch.sqrt(y.var(0, unbiased=True) + 1e-4)
    variance = (F.relu(1 - std_a).mean() + F.relu(1 - std_b).mean()) / 2
    cx, cy = x.T @ x / (len(x) - 1), y.T @ y / (len(y) - 1)
    eye = torch.eye(cx.shape[0], dtype=torch.bool, device=cx.device)
    covariance = (cx[~eye].square().sum() + cy[~eye].square().sum()) / a.shape[1]
    loss = 25 * invariance + 25 * variance + covariance
    return loss, {'invariance': invariance, 'variance': variance, 'covariance': covariance,
                  'projection_std': (std_a.mean() + std_b.mean()) / 2}


class VICRegCoAtNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from kneemri.coatnet_model import build_model
        self.backbone = build_model(cfg, pretrained=True).backbone
        d = self.backbone.num_features
        self.projector = nn.Sequential(nn.Linear(d, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                       nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                       nn.Linear(2048, 1024, bias=False))

    def forward(self, x):
        feature = self.backbone(x)
        return self.projector(feature), feature
