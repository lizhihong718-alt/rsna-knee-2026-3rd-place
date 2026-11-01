"""OAI teacher pseudo-labels for the ResNet E2E_D native384 trainer: pretrain, mix, fine-tune.

Identity when disabled: with none of CONFIG_KEYS set nothing is patched and training is
bit-for-bit the parent recipe (E1 = R2 + EMA).

* OAI knees are read from a *virtual* original-neighbour corpus (``oai_index``): the
  competition 64-slot metadata arrays plus a per-knee table of lizhihong's teacher-cache
  ``native384.npy`` files, so pixels, slots, series, groups and lateral positions are
  exactly what the depth teachers were fed. No canonical flip/swap (laterality known).
* ``oai_labels_csv`` holds the equal-weight mean of the teachers' sigmoid probabilities.
  The teachers were trained with the competition positive weights w, so their
  probabilities sit at the pos-weighted fixed point p = w t / (w t + 1 - t). With
  ``oai_target_transform: deweight`` the stored target is t = p / (p + w (1 - p)) for the
  competition w; the unchanged mass-preserved pos-weighted BCE then has its optimum at the
  teacher probability p and is algebraically BCE(student, p) / mass_normalizer per class.
  Competition rows and their loss are untouched.
* Each epoch draws ``rsna_per_epoch`` competition studies ('all' or an int, 0 = OAI-only
  pretraining) and ``oai_per_epoch`` OAI knees ('all' or an int), without replacement,
  shuffled together from the trainer's per-epoch generator.
* Positive weights and the mass normalizer are always those of the competition training
  cohort (the full-strength check runs on that cohort before OAI rows are added).
* ``init_checkpoint`` loads a complete model state (strict) after the model is built, for
  the fine-tuning stage after OAI pretraining.

Host-aligned expert labels (10-16, reports/oai_host_20261016), all off by default:

* NaN OAI label cells are masked (zero weight); ``oai_weights_csv`` gives per-cell weights
  (same rows/columns), multiplied by ``oai_loss_weight``.
* ``oai_fixed_denominator``: the global loss denominator of an effective batch is 12 x the
  number of studies instead of the summed cell weights, so masked OAI cells remove gradient
  without rescaling everyone else. Identity on competition-only groups (all 12 weights 1).
* ``oai_logit_adjust``: OAI rows are scored on logits + delta_c with
  delta_c = logit(OAI supervised prevalence) - logit(competition training prevalence), the
  label-shift correction for the selected OAI cells; 0 for classes with < 20 positive or
  negative OAI mass. Inference never sees it.
* ``oai_sampling: cells``: the per-epoch OAI draw is without replacement with probability
  proportional to the number of supervised cells of each knee (default uniform);
  ``weights`` (10-16, soft-label arms): proportional to the summed cell weights instead, so
  half-weight filled cells count half. Identical to ``cells`` when every weight is 0 or 1.
* ``oai_importance_weight`` (10-16, label-distribution matching; excludes ``oai_logit_adjust``):
  per class, OAI positive terms are weighted a_c = pi_RSNA / pi_OAI and negative terms
  b_c = (1 - pi_RSNA) / (1 - pi_OAI), pi_OAI being the OAI prevalence of the cells actually
  drawn (cell weights x draw probabilities), so each class's OAI supervision has the
  competition class mix. ``label``: plain importance weighting, the label-shift correction for
  OAI ground truth (BCE optimum moves by log(a/b) = -delta, same optimum as the logit offset).
  ``teacher``: also adds log(a/b) to OAI logits, cancelling that move, so the optimum stays the
  teacher probability (teachers are already calibrated on the competition prior) and only the
  gradient class mix changes. Classes with < 20 OAI positive or negative mass get a = b = 1.
* ``oai_until_epoch: N`` (10-18 user: joint until the LR tail, then competition only): OAI knees
  are drawn in epochs 1..N only; later epochs draw the competition studies alone, so they are
  shorter. The LR schedule is kept in epoch units: an update in a shorter epoch advances the
  parent scheduler by (reference updates per epoch) / (updates this epoch), so the warmup and the
  ``lr_tail_*`` epochs land where the parent recipe puts them, and the tail anneals to its final
  ratio by the last epoch. Epochs 1..N are bit-identical to the run without the key. The epoch is
  read from the per-epoch loader seed (seed + epoch), which the trainer sets before each epoch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

CONFIG_KEYS = {'oai_index', 'oai_labels_csv', 'oai_per_epoch', 'rsna_per_epoch', 'oai_loss_weight',
               'oai_target_transform', 'init_checkpoint', 'oai_weights_csv', 'oai_fixed_denominator',
               'oai_logit_adjust', 'oai_sampling', 'oai_importance_weight', 'oai_until_epoch'}
MIN_LABEL_MASS = 20.
PREFIX = 'OAI.'
ARRAYS = ('series_index', 'group_id', 'plane_id', 'position_mm', 'position_reliable')


def enabled(cfg):
    return bool(cfg.get('oai_index'))


class KneeWindows:
    """``windows[row, candidates]`` over per-knee [n_selected,3,H,W] uint8 files."""

    def __init__(self, files):
        self.paths = [f['path'] for f in files]
        self.inverse = []
        for f in files:
            inv = np.full(64, -1, np.int64)
            inv[np.asarray(f['selected'], np.int64)] = np.arange(len(f['selected']))
            self.inverse.append(inv)

    def __getitem__(self, key):
        row, candidates = key
        index = self.inverse[row][np.asarray(candidates, np.int64)]
        if (index < 0).any():
            raise IndexError('requested an empty OAI candidate slot')
        return np.load(self.paths[row], mmap_mode='r')[index]


def oai_class(base_cls):
    """The training bag class reading the virtual OAI corpus (same augmentation path)."""

    class OaiKneeBags(base_cls):
        def _open(self):
            if self.windows is not None:
                return
            self.masks = np.load(self.root / 'masks.npy', mmap_mode='r')
            self.metadata = {name: np.load(self.root / f'{name}.npy', mmap_mode='r') for name in ARRAYS}
            self.windows = KneeWindows(json.loads((self.root / 'knee_files.json').read_text()))

    OaiKneeBags.__name__ = 'OaiKneeBags_' + base_cls.__name__
    return OaiKneeBags


class MixedTrainBags(Dataset):
    """Competition and OAI training bags under one index space (any id order)."""

    def __init__(self, base_cls, cfg, all_ids, ids, labels, uint8_output=False):
        self.ids = list(ids)
        rsna_ids = [u for u in self.ids if not u.startswith(PREFIX)]
        oai_ids = [u for u in self.ids if u.startswith(PREFIX)]
        self.parts = {}
        self.rsna = base_cls(cfg, all_ids, rsna_ids, labels, train=True, uint8_output=uint8_output) \
            if rsna_ids else None
        oai_cfg = dict(cfg, corpus=cfg['oai_index'], canonical_orientation=None,
                       depth_canonical_positions=None)
        manifest = json.loads((Path(cfg['oai_index']) / 'manifest.json').read_text())
        self.oai = oai_class(base_cls)(oai_cfg, manifest['study_uids'], oai_ids, labels, train=True,
                                       uint8_output=uint8_output) if oai_ids else None
        for part, sub in (('rsna', self.rsna), ('oai', self.oai)):
            if sub is not None:
                for local, uid in enumerate(sub.ids):
                    self.parts[uid] = (part, local)
        if self.rsna is not None and self.oai is not None and (
                self.oai.uint8_output != self.rsna.uint8_output
                or self.oai.source_resolution != self.rsna.source_resolution):
            raise ValueError('OAI index must share the competition corpus resolution/transfer')
        self.uint8_output = (self.rsna or self.oai).uint8_output
        self.rsna_index = [i for i, u in enumerate(self.ids) if not u.startswith(PREFIX)]
        self.oai_index = [i for i, u in enumerate(self.ids) if u.startswith(PREFIX)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        part, local = self.parts[self.ids[index]]
        return (self.rsna if part == 'rsna' else self.oai)[local]


def _count(value, n):
    if value in (None, 'all'):
        return n
    value = int(value)
    if value < 0:
        raise ValueError('per-epoch counts must be nonnegative')
    return min(value, n)


class EpochMixSampler(Sampler):
    def __init__(self, rsna_index, oai_index, rsna_per_epoch, oai_per_epoch, generator, oai_draw_weights=None,
                 oai_until_epoch=None, seed=None):
        self.rsna_index = torch.as_tensor(rsna_index, dtype=torch.long)
        self.oai_index = torch.as_tensor(oai_index, dtype=torch.long)
        self.k_rsna = _count(rsna_per_epoch, len(rsna_index))
        self.k_oai_joint = _count(oai_per_epoch, len(oai_index))
        if not self.k_rsna + self.k_oai_joint:
            raise ValueError('an epoch must draw at least one study')
        if oai_until_epoch is not None and (seed is None or not self.k_rsna):
            raise ValueError('oai_until_epoch needs the run seed and competition studies in every epoch')
        self.oai_until_epoch, self.seed = oai_until_epoch, seed
        self.generator = generator
        self.oai_draw_weights = None
        if oai_draw_weights is not None:
            w = torch.as_tensor(oai_draw_weights, dtype=torch.double)
            if w.shape != self.oai_index.shape or not torch.isfinite(w).all() or (w <= 0).any():
                raise ValueError('OAI draw weights must be positive, one per OAI knee')
            self.oai_draw_weights = w

    def epoch(self):
        """0-based epoch from the loader seed; 0 before the first epoch seeds it (sizing the schedule)."""
        if self.seed is None:
            return 0
        epoch = self.generator.initial_seed() - self.seed
        return epoch if 0 <= epoch < 100000 else 0

    @property
    def k_oai(self):
        if self.oai_until_epoch is not None and self.epoch() >= self.oai_until_epoch:
            return 0
        return self.k_oai_joint

    def __len__(self):
        return self.k_rsna + self.k_oai

    def __iter__(self):
        g = self.generator
        parts = []
        for index, k in ((self.rsna_index, self.k_rsna), (self.oai_index, self.k_oai)):
            if k == len(index):
                parts.append(index)
            elif k and index is self.oai_index and self.oai_draw_weights is not None:
                parts.append(index[torch.multinomial(self.oai_draw_weights, k, replacement=False, generator=g)])
            elif k:
                parts.append(index[torch.randperm(len(index), generator=g)[:k]])
        order = torch.cat(parts)
        order = order[torch.randperm(len(order), generator=g)]
        return iter(order.tolist())


class EpochUnitScheduler:
    """Step a per-update scheduler in epoch units when epochs differ in update count (oai_until_epoch)."""

    def __init__(self, inner, sampler, updates_ref, batch_size, accumulation, drop_last):
        self.inner, self.sampler, self.updates_ref = inner, sampler, updates_ref
        self.batch_size, self.accumulation, self.drop_last = batch_size, accumulation, drop_last
        self.epoch, self.updates_in_epoch = None, 0

    def epoch_updates(self):
        items = len(self.sampler)
        batches = items // self.batch_size if self.drop_last else -(-items // self.batch_size)
        steps = min(batches, self.updates_ref * self.accumulation)  # a smoke run caps every epoch alike
        return -(-steps // self.accumulation)

    def step(self):
        epoch = self.sampler.epoch()
        if epoch != self.epoch:
            self.epoch, self.updates_in_epoch = epoch, 0
        self.updates_in_epoch += 1
        position = epoch * self.updates_ref + round(self.updates_in_epoch * self.updates_ref / self.epoch_updates())
        self.inner.last_epoch = position - 1
        self.inner.step()

    def state_dict(self):
        return dict(self.inner.state_dict(), epoch_unit=dict(epoch=self.epoch, updates_in_epoch=self.updates_in_epoch))

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        unit = state_dict.pop('epoch_unit')
        self.epoch, self.updates_in_epoch = unit['epoch'], unit['updates_in_epoch']
        self.inner.load_state_dict(state_dict)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def read_oai_labels(path, label_cols):
    frame = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    if list(frame.columns) != list(label_cols):
        raise ValueError('OAI label columns differ from the competition label order')
    if not frame.index.str.startswith(PREFIX).all() or frame.index.duplicated().any():
        raise ValueError('OAI label rows must use unique OAI.<ID>.<side> study ids')
    values = frame.to_numpy(np.float64)
    finite = np.isfinite(values)
    if np.isinf(values).any() or (values[finite] < 0).any() or (values[finite] > 1).any():
        raise ValueError('OAI labels must be probabilities or NaN (masked)')
    if not finite.any(axis=1).all():
        raise ValueError('every OAI row needs at least one supervised cell')
    return frame.index.tolist(), values


def read_oai_weights(path, oai_ids, label_cols, values):
    frame = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    if list(frame.columns) != list(label_cols) or frame.index.duplicated().any():
        raise ValueError('OAI weight columns/rows differ from the label file')
    if set(frame.index) != set(oai_ids):
        raise ValueError('OAI weight rows differ from the label rows')
    weights = frame.loc[oai_ids].to_numpy(np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('OAI weights must be finite and nonnegative')
    if (weights[~np.isfinite(values)] != 0).any():
        raise ValueError('masked (NaN) OAI cells must have weight 0')
    if not (weights > 0).any(axis=1).all():
        raise ValueError('every OAI row needs a positive weight')
    return weights


def logit_offsets(oai_targets, oai_weights, rsna_targets):
    """delta_c = logit(pi_OAI) - logit(pi_competition); 0 where OAI lacks either label."""
    y = np.where(oai_weights > 0, np.nan_to_num(oai_targets), 0.)
    pos, neg = (oai_weights * y).sum(0), (oai_weights * (1 - y)).sum(0)
    pi_oai = np.clip(pos / np.maximum(pos + neg, 1e-12), .01, .99)
    pi_rsna = np.clip(np.asarray(rsna_targets, np.float64).mean(0), .01, .99)
    delta = np.log(pi_oai / (1 - pi_oai)) - np.log(pi_rsna / (1 - pi_rsna))
    delta[(pos < MIN_LABEL_MASS) | (neg < MIN_LABEL_MASS)] = 0.
    return delta, pi_oai, pi_rsna, pos, neg


def deweight(p, w):
    """Inverse of the pos-weighted fixed point p = w t / (w t + 1 - t)."""
    p = np.clip(np.asarray(p, np.float64), 0, 1)
    return p / (p + np.asarray(w, np.float64) * (1 - p))


def install(trainer):
    """Install after the full-strength hook (its mass check must see the competition cohort)."""
    if getattr(trainer, '_oai_pseudo_installed', False):
        return
    trainer._oai_pseudo_installed = True
    trainer.DATA_KEYS.update(CONFIG_KEYS)
    import kneemri.coatnet_original_data as data_module

    base_cls = data_module.CoAtNetOriginalNeighborBags
    original_cohort = trainer.load_cohort
    original_weights = trainer.supervision_weights
    original_positive = trainer.positive_weights
    original_loader = trainer.make_loader
    original_receipt = trainer.source_receipt
    original_build = trainer.build_model
    original_group = trainer.group_supervision
    original_inputs = trainer.model_inputs
    original_loss = trainer.weighted_loss
    original_scheduler = trainer.create_scheduler
    original_groups = trainer.accumulation_groups
    state = {}

    def load_cohort(cfg):
        all_ids, train_ids, val_ids, labels, cohort = original_cohort(cfg)
        if cfg.get('init_checkpoint'):
            cohort = dict(cohort, init_checkpoint=str(cfg['init_checkpoint']),
                          init_checkpoint_sha256=trainer.hash_file(cfg['init_checkpoint']))
        if not enabled(cfg):
            return all_ids, train_ids, val_ids, labels, cohort
        comp = np.stack([labels[u] for u in train_ids])
        pw = original_positive(comp, np.ones_like(comp), cfg)
        state['pw'] = pw
        oai_ids, values = read_oai_labels(cfg['oai_labels_csv'], trainer.LABEL_COLS)
        manifest = json.loads((Path(cfg['oai_index']) / 'manifest.json').read_text())
        if set(oai_ids) - set(manifest['study_uids']):
            raise ValueError('OAI labels name knees missing from the OAI index')
        if set(oai_ids) & set(all_ids):
            raise ValueError('OAI ids collide with competition ids')
        transform = cfg.get('oai_target_transform', 'deweight')
        if transform == 'deweight':
            targets = deweight(values, pw)
        elif transform == 'raw':
            targets = values
        else:
            raise ValueError('oai_target_transform must be deweight or raw')
        if cfg.get('oai_weights_csv'):
            cell_weights = read_oai_weights(cfg['oai_weights_csv'], oai_ids, trainer.LABEL_COLS, values)
        else:
            cell_weights = np.isfinite(values).astype(np.float64)
        state['cell_weights'] = dict(zip(oai_ids, cell_weights.astype(np.float32)))
        state['cells'] = dict(zip(oai_ids, (cell_weights > 0).sum(1).astype(np.float64)))
        state['weight_sums'] = dict(zip(oai_ids, cell_weights.sum(1).astype(np.float64)))
        state['cfg'] = cfg
        extra = {}
        until = cfg.get('oai_until_epoch')
        if until is not None:
            if isinstance(until, bool) or not isinstance(until, int) or not 1 <= until < cfg['epochs']:
                raise ValueError('oai_until_epoch must be an integer in [1, epochs)')
            extra['oai_until_epoch'] = until
        if cfg.get('oai_weights_csv'):
            extra['weights_csv'] = str(cfg['oai_weights_csv'])
            extra['weights_sha256'] = trainer.hash_file(cfg['oai_weights_csv'])
        if cfg.get('oai_logit_adjust'):
            delta, pi_oai, pi_rsna, pos, neg = logit_offsets(targets, cell_weights, comp)
            state['delta'] = torch.as_tensor(delta, dtype=torch.float32)
            extra['logit_adjust'] = dict(delta=delta.tolist(), pi_oai=pi_oai.tolist(), pi_rsna=pi_rsna.tolist(),
                                         oai_positive_mass=pos.tolist(), oai_negative_mass=neg.tolist())
        mode = cfg.get('oai_importance_weight')
        if mode:
            if mode not in ('label', 'teacher') or cfg.get('oai_logit_adjust'):
                raise ValueError('oai_importance_weight must be label or teacher, without oai_logit_adjust')
            sampling = cfg.get('oai_sampling', 'uniform')
            draw = {'uniform': np.ones(len(oai_ids)), 'cells': (cell_weights > 0).sum(1),
                    'weights': cell_weights.sum(1)}[sampling].astype(np.float64)
            q = (draw / draw.sum())[:, None]
            y = np.where(cell_weights > 0, np.nan_to_num(targets), 0.)
            pos, neg = (q * cell_weights * y).sum(0), (q * cell_weights * (1 - y)).sum(0)
            pool_pos, pool_neg = (cell_weights * y).sum(0), (cell_weights * (1 - y)).sum(0)
            pi_oai = np.clip(pos / np.maximum(pos + neg, 1e-12), .01, .99)
            pi_rsna = np.clip(comp.mean(0), .01, .99)
            a, b = pi_rsna / pi_oai, (1 - pi_rsna) / (1 - pi_oai)
            off = (pool_pos < MIN_LABEL_MASS) | (pool_neg < MIN_LABEL_MASS)
            a[off], b[off] = 1., 1.
            state['iw_a'] = torch.as_tensor(a, dtype=torch.float32)
            state['iw_b'] = torch.as_tensor(b, dtype=torch.float32)
            state['iw_offset'] = torch.as_tensor(np.log(a / b), dtype=torch.float32) if mode == 'teacher' else None
            extra['importance_weight'] = dict(mode=mode, a=a.tolist(), b=b.tolist(), pi_oai_drawn=pi_oai.tolist(),
                                              pi_rsna=pi_rsna.tolist())
        extra.update(fixed_denominator=bool(cfg.get('oai_fixed_denominator', False)),
                     sampling=cfg.get('oai_sampling', 'uniform'),
                     supervised_cells=(cell_weights > 0).sum(0).tolist())
        if cfg.get('oai_sampling', 'uniform') not in ('uniform', 'cells', 'weights'):
            raise ValueError('oai_sampling must be uniform, cells or weights')
        labels = dict(labels, **{u: t.astype(np.float32) for u, t in zip(oai_ids, targets)})
        k_rsna = cfg.get('rsna_per_epoch', 'all')
        ordered = oai_ids + list(train_ids) if k_rsna == 0 else list(train_ids) + oai_ids
        if os.environ.get('KNEEMRI_OAI_INTERLEAVE') == '1':  # smoke tests only: mixed first 16 studies
            ordered = [u for pair in zip(oai_ids, train_ids) for u in pair]
        with np.errstate(invalid='ignore'):
            target_mean = [None if not np.isfinite(m) else float(m) for m in np.nanmean(
                np.where(cell_weights > 0, targets, np.nan), 0)]
        cohort = dict(cohort, trained_study_uids=ordered,
                      oai=dict(index=str(cfg['oai_index']), labels_csv=str(cfg['oai_labels_csv']),
                               labels_sha256=trainer.hash_file(cfg['oai_labels_csv']),
                               index_manifest_sha256=trainer.hash_file(Path(cfg['oai_index']) / 'manifest.json'),
                               knee_files_sha256=trainer.hash_file(Path(cfg['oai_index']) / 'knee_files.json'),
                               knees=len(oai_ids), oai_per_epoch=cfg.get('oai_per_epoch', 'all'),
                               rsna_per_epoch=k_rsna, loss_weight=float(cfg.get('oai_loss_weight', 1.0)),
                               target_transform=transform, competition_positive_weights=pw.tolist(),
                               target_mean=target_mean, **extra))
        return all_ids, ordered, val_ids, labels, cohort

    def supervision_weights(cfg, train_ids, cohort):
        weights = original_weights(cfg, [u for u in train_ids if not u.startswith(PREFIX)], cohort)
        scale = float(cfg.get('oai_loss_weight', 1.0))
        for uid in train_ids:
            if uid.startswith(PREFIX):
                weights[uid] = (scale * state['cell_weights'][uid]).astype(np.float32)
        return {uid: weights[uid] for uid in train_ids}

    def group_supervision(batches, weights):
        denominators = original_group(batches, weights)
        cfg = state.get('cfg')
        if not (cfg and cfg.get('oai_fixed_denominator')):
            return denominators
        studies = sum(int(batch['window_mask'].any(1).sum()) for batch in batches)
        return torch.full((12,), float(studies))

    def model_inputs(batch, device):
        state['current_ids'] = list(batch['study_ids'])
        return original_inputs(batch, device)

    def weighted_loss(logits, targets, weights, pos_weight, cfg, group_denominators=None):
        delta = state.get('delta')
        ids = state.get('current_ids')
        if delta is not None and ids is not None:
            if len(ids) != logits.shape[0]:
                raise RuntimeError('logit adjustment lost track of the batch study ids')
            rows = torch.tensor([u.startswith(PREFIX) for u in ids], device=logits.device)
            if rows.any():
                logits = logits.float() + rows[:, None].float() * delta.to(logits.device)[None]
        if state.get('iw_a') is not None and ids is not None:
            if len(ids) != logits.shape[0]:
                raise RuntimeError('importance weighting lost track of the batch study ids')
            rows = torch.tensor([u.startswith(PREFIX) for u in ids], device=logits.device)[:, None]
            if rows.any():
                a, b = state['iw_a'].to(logits.device)[None], state['iw_b'].to(logits.device)[None]
                ones = torch.ones_like(a)
                pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=logits.device).reshape(1, -1)
                pos_weight = pw * torch.where(rows, a / b, ones)        # [B,12]: positive term a*w, negative term b
                weights = weights.float() * torch.where(rows, b, ones)
                if state.get('iw_offset') is not None:
                    logits = logits.float() + rows.float() * state['iw_offset'].to(logits.device)[None]
        return original_loss(logits, targets, weights, pos_weight, cfg, group_denominators)

    def positive_weights(targets, weights, cfg):
        # Before OAI rows exist (the full-strength mass check) the original applies.
        if enabled(cfg) and 'pw' in state:
            return np.asarray(state['pw'], np.float32)
        return original_positive(targets, weights, cfg)

    def factory(cfg, all_ids, ids, labels, train, uint8_output=False):
        if train and enabled(cfg):
            return MixedTrainBags(base_cls, cfg, all_ids, ids, labels, uint8_output=uint8_output)
        return base_cls(cfg, all_ids, ids, labels, train=train, uint8_output=uint8_output)

    def make_loader(dataset, cfg, device, train=False, generator=None, pin_memory=True):
        if not (train and isinstance(dataset, MixedTrainBags)):
            return original_loader(dataset, cfg, device, train, generator, pin_memory)
        from kneemri.coatnet_training_data import coatnet_collate
        draw = None
        if cfg.get('oai_sampling', 'uniform') == 'cells':
            draw = [state['cells'][dataset.ids[i]] for i in dataset.oai_index]
        elif cfg.get('oai_sampling', 'uniform') == 'weights':
            draw = [state['weight_sums'][dataset.ids[i]] for i in dataset.oai_index]
        sampler = EpochMixSampler(dataset.rsna_index, dataset.oai_index, cfg.get('rsna_per_epoch', 'all'),
                                  cfg.get('oai_per_epoch', 'all'), generator, draw,
                                  oai_until_epoch=cfg.get('oai_until_epoch'), seed=cfg['seed'])
        state['sampler'] = sampler
        kwargs = dict(batch_size=cfg['batch_size'], sampler=sampler, num_workers=cfg['workers'],
                      pin_memory=pin_memory and device.type == 'cuda',
                      worker_init_fn=trainer.seed_worker, collate_fn=coatnet_collate,
                      generator=generator, drop_last=cfg['drop_last_train'])
        if cfg['workers']:
            kwargs['prefetch_factor'] = 2
        return DataLoader(dataset, **kwargs)

    def build_model(cfg, pretrained=False):
        model = original_build(cfg, pretrained)
        if pretrained and cfg.get('init_checkpoint'):
            checkpoint = torch.load(cfg['init_checkpoint'], map_location='cpu', weights_only=True)
            if checkpoint.get('arch') != cfg['arch'] or checkpoint.get('res') != cfg['resolution'] \
                    or checkpoint.get('lab') != trainer.LABEL_COLS:
                raise ValueError('init_checkpoint architecture, resolution or label order differs')
            model.load_state_dict(checkpoint['model'], strict=True)
        return model

    def receipt(output, cfg, cohort, synthetic=False):
        result = original_receipt(output, cfg, cohort, synthetic)
        source = Path(__file__)
        target = output / 'source/src/kneemri' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        result['sources']['src/kneemri/oai_pseudo.py'] = trainer.hash_file(source)
        entry = source.parents[2] / 'scripts/train_oai_pseudo.py'
        result['sources']['scripts/train_oai_pseudo.py'] = trainer.hash_file(entry)
        copy = output / 'source/scripts' / entry.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        if not copy.exists():
            copy.write_bytes(entry.read_bytes())
        result['oai_pseudo'] = dict(oai=cohort.get('oai'), init_checkpoint=cohort.get('init_checkpoint'),
                                    init_checkpoint_sha256=cohort.get('init_checkpoint_sha256'))
        return result

    def create_scheduler(optimizer, cfg, updates_per_epoch):
        scheduler = original_scheduler(optimizer, cfg, updates_per_epoch)
        if not enabled(cfg) or cfg.get('oai_until_epoch') is None:
            return scheduler
        return EpochUnitScheduler(scheduler, state['sampler'], updates_per_epoch, cfg['batch_size'],
                                  cfg['gradient_accumulation'], cfg['drop_last_train'])

    def accumulation_groups(loader, steps, accumulation):
        sampler = getattr(loader, 'sampler', None)
        if isinstance(sampler, EpochMixSampler) and sampler.oai_until_epoch is not None:
            steps = min(steps, len(loader))  # competition-only epochs are shorter than the sized first one
        return original_groups(loader, steps, accumulation)

    trainer.load_cohort = load_cohort
    trainer.create_scheduler = create_scheduler
    trainer.accumulation_groups = accumulation_groups
    trainer.supervision_weights = supervision_weights
    trainer.positive_weights = positive_weights
    trainer.make_loader = make_loader
    trainer.build_model = build_model
    trainer.source_receipt = receipt
    trainer.group_supervision = group_supervision
    trainer.model_inputs = model_inputs
    trainer.weighted_loss = weighted_loss
    data_module.CoAtNetOriginalNeighborBags = factory
