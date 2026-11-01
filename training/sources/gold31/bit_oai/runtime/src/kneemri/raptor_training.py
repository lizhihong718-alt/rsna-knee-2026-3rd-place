"""Local, reproducible training of the Raptor-compatible attention MIL baseline.

The model and sampled-stack input contract come from the credited reference module.
This trainer uses our label table and generic ImageNet initialization. It does not
use the public competition-trained checkpoint as initialization or as OOF evidence.
"""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kneemri import raptor_reference as raptor
from kneemri.labels import read_labels


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cohort(cfg):
    root = Path(cfg['corpus'])
    manifest = json.loads((root / 'manifest.json').read_text())
    pre = manifest['preprocessing']
    corpus_format = cfg.get('corpus_format', 'raptor64')
    if corpus_format == 'raptor64':
        if (pre['schema'] != 'raptor64-published-v7-v1' or pre['image'] != raptor.IMG
                or pre['crop_mm'] != raptor.CROP_MM
                or pre['slots'] != [list(s) for s in raptor.SLOTS]
                or pre['span'] != [.06, .94] or pre['percentiles'] != [2, 98]
                or pre['window_semantics'] != 'adjacent_in_sampled_concatenated_stack'):
            raise ValueError('incompatible corpus; the reference weights require the 64-slice recipe')
        image_path = root / 'volumes.npy'
        mask_path = root / 'masks.npy'
        expected_shape = tuple(manifest['shape'])
        mask_width = expected_shape[1]
    elif corpus_format == 'original_neighbors':
        if (pre.get('schema') != 'coatnet-original-neighbors-v1'
                or pre.get('window_semantics') != 'original_consecutive_source_slices'
                or len(manifest.get('shape', [])) != 5 or manifest['shape'][2] != 3):
            raise ValueError('incompatible original-neighbour CoAtNet corpus')
        image_path = root / 'windows.npy'
        mask_path = root / 'masks.npy'
        expected_shape = tuple(manifest['shape'])
        mask_width = expected_shape[1]
    elif corpus_format == 'public224_triplets':
        expected_shape = tuple(manifest.get('volume_shape', []))
        triplet_shape = manifest.get('triplet_shape')
        if (pre.get('schema') != 'coatnet-public224-triplets-v1'
                or pre.get('window_semantics') != 'public_gap_gated_physical_triplets'
                or pre.get('source_dataset_version') != 'RSNA_KNEE_RESNET224_V1'
                or pre.get('source_algorithm_version') != 'resnet224-v1.0.0'
                or pre.get('fov_mm') != 140 or pre.get('span') != [.02, .98]
                or pre.get('percentiles') != [2, 98] or pre.get('image') != 224
                or pre.get('slot_offsets') != [0, 24, 44, 64, 76, 88, 96]
                or len(expected_shape) != 4 or expected_shape[1:] != (96, 224, 224)
                or triplet_shape != [84, 3]):
            raise ValueError('incompatible public physical-triplet CoAtNet corpus')
        image_path = root / 'volumes.npy'
        image_paths = [image_path]
        mask_path = root / 'triplet_valid_mask.npy'
        mask_width = triplet_shape[0]
    elif corpus_format == 'published_raptor44':
        expected_shape = tuple(manifest.get('shape', []))
        parts = manifest.get('parts', [])
        if (pre.get('schema') != 'coatnet-published-raptor44-v1'
                or pre.get('window_semantics') != 'released_concatenated_stack_neighbors'
                or pre.get('source_dataset_versions') != [2, 1]
                or pre.get('crop_mm') != 140 or pre.get('span') != [.15, .85]
                or pre.get('image') != 336
                or pre.get('slot_offsets') != [0, 12, 22, 30, 36, 44]
                or expected_shape[1:] != (44, 336, 336) or len(parts) != 2):
            raise ValueError('incompatible released Raptor 44-slice corpus')
        image_paths = [(root / part['volume_path']).resolve() for part in parts]
        part_shapes = [tuple(part.get('shape', [])) for part in parts]
        if (any(shape[1:] != expected_shape[1:] for shape in part_shapes)
                or sum(shape[0] for shape in part_shapes) != expected_shape[0]):
            raise ValueError('released Raptor part shapes differ from the combined manifest')
        mask_path = root / 'masks.npy'
        mask_width = 44
    else:
        raise ValueError('unsupported corpus_format')
    if corpus_format != 'published_raptor44':
        image_paths = [image_path]
        part_shapes = [expected_shape]
    if cfg['normalization'] not in {'unit', 'imagenet', 'timm_half'}:
        raise ValueError('a supported pretrained-backbone normalization must be explicit')
    ids = manifest['study_uids']
    done = np.load(root / 'complete.npy', mmap_mode='r')
    masks = np.load(mask_path, mmap_mode='r')
    images = [np.load(path, mmap_mode='r') for path in image_paths]
    if (expected_shape[0] != len(ids) or len(ids) != len(set(ids))
            or done.shape != (len(ids),) or not done.all()
            or masks.shape != (len(ids), mask_width)
            or masks.dtype != np.uint8
            or any(image.shape != shape or image.dtype != np.uint8
                   for image, shape in zip(images, part_shapes))):
        raise ValueError('incomplete or invalid corpus layout')
    pseudo = read_labels(cfg['labels_csv']); gold = read_labels(cfg['gold_csv'])
    gold_ids = gold.dropna().index.tolist()
    if not gold_ids or not set(gold_ids) <= set(ids): raise ValueError('gold studies missing')
    groups = pd.read_csv(cfg['groups_csv']).set_index('StudyInstanceUID')['group']
    if groups.index.duplicated().any() or not set(ids) <= set(groups.index):
        raise ValueError('incomplete/duplicate study grouping table')
    # Identical normalized reports stay together even when one member is in gold.
    held_groups = set(groups.loc[gold_ids])
    empty = {uid for i, uid in enumerate(ids) if not masks[i].any()}
    if empty & set(gold_ids): raise ValueError('an expert-labeled study has no valid images')
    eligible = [uid for uid in ids if uid not in empty and groups.loc[uid] not in held_groups]
    unlabeled = [uid for uid in eligible if uid not in pseudo.index]
    train_ids = [uid for uid in eligible if uid in pseudo.index]
    if not train_ids:
        raise ValueError('missing training labels')
    values = pseudo.loc[train_ids].to_numpy(np.float32)
    if not np.isfinite(values).all(): raise ValueError('this complete-label baseline requires finite soft labels')
    if not np.isin(gold.loc[gold_ids].to_numpy(), [0, 1]).all(): raise ValueError('gold must be binary')
    labels = {uid: row.to_numpy(np.float32) for uid, row in pseudo.iterrows()}
    labels.update({uid: gold.loc[uid].to_numpy(np.float32) for uid in gold_ids})
    audit = {'validation_role': cfg['validation_role'], 'trained_study_uids': train_ids,
             'validation_study_uids': gold_ids, 'excluded_empty_study_uids': sorted(empty),
             'excluded_unlabeled_study_uids': sorted(unlabeled),
             'excluded_report_group_peers': [u for u in ids if u not in gold_ids and groups.loc[u] in held_groups],
             'corpus_manifest_sha256': sha256(root / 'manifest.json'),
             'labels_sha256': sha256(cfg['labels_csv']), 'gold_sha256': sha256(cfg['gold_csv']),
             'groups_sha256': sha256(cfg['groups_csv'])}
    return ids, train_ids, gold_ids, labels, audit


class StudyBags(Dataset):
    def __init__(self, cfg, all_ids, ids, labels, train):
        self.cfg = cfg; self.ids = list(ids); self.labels = labels; self.train = train
        self.rows = {u: i for i, u in enumerate(all_ids)}
        self.volumes = self.masks = None

    def __len__(self): return len(self.ids)

    def __getitem__(self, index):
        if self.volumes is None:
            root = Path(self.cfg['corpus'])
            self.volumes = np.load(root / 'volumes.npy', mmap_mode='r')
            self.masks = np.load(root / 'masks.npy', mmap_mode='r')
        uid = self.ids[index]; row = self.rows[uid]
        vol = self.volumes[row]; mask = self.masks[row]
        if self.train:
            # Compatibility with the released training: adjacency is in the sampled
            # 64-image stack. These are deliberately NOT advertised as original neighbours.
            valid = np.flatnonzero(mask)
            lo, hi = int(valid.min()), int(valid.max())
            centers = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, len(vol) - 2))]
            k = self.cfg['train_windows']
            pool = centers * (k // len(centers) + 1); random.shuffle(pool)
            x = torch.from_numpy(np.stack([vol[c-1:c+2] for c in pool[:k]]).astype(np.float32) / 255.0)
            res = self.cfg['resolution']
            if x.shape[-2:] != (res, res):
                x = torch.nn.functional.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
            gain = 1 + random.uniform(-self.cfg['intensity_gain'], self.cfg['intensity_gain'])
            x = (x * gain).clamp(0, 1)
            x = (x - raptor._MEAN) / raptor._STD
        else:
            x = raptor.eval_windows(vol, mask, self.cfg['eval_windows'], self.cfg['resolution'])
        return x, torch.from_numpy(self.labels[uid].copy())


def initialize_model(cfg):
    from safetensors.torch import load_file
    path = Path(cfg['initial_weights'])
    if path.suffix != '.safetensors': raise ValueError('use the generic timm safetensors initialization')
    state = load_file(str(path)); bb = raptor.build_backbone(cfg['arch'], pretrained=False)
    extra = set(state) - set(bb.state_dict())
    if extra != {'head.fc.weight', 'head.fc.bias'}:
        raise ValueError(f'initialization is not the expected generic ImageNet classifier: {extra}')
    for key in extra: del state[key]
    bb.load_state_dict(state, strict=True)
    model = raptor.RaptorClassifier(bb, F_dim=bb.num_features)
    if cfg['gradient_checkpointing']: bb.set_grad_checkpointing(True)
    return model
