"""One image/metadata contract for cached features, training and inference."""
from __future__ import annotations
import glob
import hashlib
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from kneemri.config import abspath, SamplingConfig
from kneemri.sampling import sample_positions
from kneemri.labels import LABEL_COLS, LabelTable


def select_candidates(valid, quality, budget, rng=None):
    """Coverage first, then extra candidates; rank IDs are never reassigned."""
    selected = np.zeros_like(valid, dtype=bool)
    available = np.argwhere(valid)
    if budget is None or len(available) <= budget:
        return valid.copy()
    if budget < 1:
        raise ValueError("series budget must be positive")
    slots = np.flatnonzero(valid.any(1)).tolist()
    if rng is not None:
        rng.shuffle(slots)
    else:
        slots.sort(key=lambda s: -float(quality[s, valid[s], 0].max()))
    for s in slots[:budget]:
        ranks = np.flatnonzero(valid[s])
        scores = quality[s, ranks, 0]
        if rng is None:
            rank = ranks[np.argmax(scores)]
        else:
            reasonable = ranks[scores >= scores.max() - 0.2]
            rank = rng.choice(reasonable)
        selected[s, rank] = True
    remaining = np.argwhere(valid & ~selected)
    if rng is not None:
        rng.shuffle(remaining)
    else:
        remaining = sorted(remaining, key=lambda sk: -float(quality[tuple(sk)][0]))
    for s, k in remaining[:max(0, budget - int(selected.sum()))]:
        selected[s, k] = True
    return selected


class KneeMRIDataset(Dataset):
    def __init__(self, processed_dir, cfg=None, labels_csv=None, resize_to=None,
                 k_series=None, training=False, files=None, apply_budget=True,
                 slot_ids=None):
        # Parallelism belongs to DataLoader workers; OpenCV's per-process
        # default thread pool otherwise oversubscribes the shared server.
        cv2.setNumThreads(1)
        self.files = list(files) if files is not None else sorted(glob.glob(os.path.join(abspath(processed_dir), '*.npz')))
        if not self.files:
            raise FileNotFoundError(f'no npz under {processed_dir}')
        self.cfg, self.training, self.epoch = cfg, training, 0
        self.apply_budget = apply_budget
        self.resize_to = resize_to or (cfg.dataset.resize_to if cfg else None)
        with np.load(self.files[0]) as z:
            found = sorted(k[len('slot_'):-len('_n_series')] for k in z.files if k.endswith('_n_series'))
            self.slot_ids = list(slot_ids or (list(cfg.slots.definitions) if cfg else z['slot_ids'].tolist() if 'slot_ids' in z else found))
            if set(found) != set(self.slot_ids):
                raise ValueError('configured slot identities differ from NPZ')
            self.k_series = k_series or (cfg.slots.k_series_per_slot if cfg else 1)
            keys = [k for k in z.files if k.endswith('_positions')]
            stored_n = len(z[keys[0]]) if keys else 32
            self.slices_per_slot = (cfg.dataset.centers_per_series or cfg.sampling.slices_per_slot) if cfg else stored_n
            if cfg and training and cfg.dataset.train_centers_per_series:
                self.slices_per_slot = cfg.dataset.train_centers_per_series
            volumes = [k for k in z.files if k.endswith('_volume')]
            self.resize_to = self.resize_to or (int(z[volumes[0]].shape[-1]) if volumes else 224)
        self.n_slots = len(self.slot_ids)
        self.neighbor_offsets = tuple(cfg.sampling.neighbor_offsets) if cfg else (-1,0,1)
        if self.neighbor_offsets != (-1,0,1):
            raise ValueError('2.5D reader currently requires [-1,0,1] original neighbors')
        path = labels_csv if labels_csv is not None else cfg.dataset.labels_csv if cfg else None
        self.label_table = LabelTable(path, cfg.dataset.train_csv if cfg else '',
                                     require_files=bool(cfg and cfg.dataset.require_targets))
        self.labels, self.gold = self.label_table.pseudo, self.label_table.gold
        if cfg and cfg.dataset.require_targets:
            self.label_table.validate_coverage([self.study_uid(i) for i in range(len(self))])
        self.channel_mode = cfg.dataset.channel_mode if cfg else 'adjacent'
        if self.channel_mode not in ('adjacent', 'repeat'):
            raise ValueError('channel_mode must be adjacent or repeat')
        if cfg and cfg.dataset.position_frame not in ('physical','canonical'):
            raise ValueError('position_frame must be physical or canonical')

    def __len__(self):
        return len(self.files)

    def study_uid(self, idx):
        return os.path.basename(self.files[idx])[:-4]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        uid = self.study_uid(idx)
        seed = int(hashlib.sha256(f'{self.cfg.seed if self.cfg else 0}:{self.epoch}:{uid}'.encode()).hexdigest()[:16],16)
        rng = np.random.default_rng(seed)
        S,K,N,H = self.n_slots,self.k_series,self.slices_per_slot,self.resize_to
        images = np.zeros((S,K,N,3,H,H),np.uint8)
        pad = np.ones((S,K,N),bool)
        valid = np.zeros((S,K),bool)
        positions = np.zeros((S,K,N),np.int32)
        z_norm = np.full((S,K,N),0.5,np.float32)
        physical = np.zeros((S,K,N,6),np.float32)  # relative mm/100, extent/200, dz-/10, dz+/10, geom, direction
        quality = np.zeros((S,K,8),np.float32)
        meta = np.full((S,K,3),255,np.int64)
        pixel_valid = np.zeros((S,K,N,H,H),bool)
        with np.load(self.files[idx]) as z:
            if self.cfg and self.cfg.dataset.require_schema and int(z.get('schema_version',0)) != self.cfg.dataset.require_schema:
                raise ValueError('NPZ schema is stale; regenerate with full-FOV preprocessing')
            if self.cfg and self.cfg.dataset.require_schema:
                from kneemri.pipeline import preprocess_signature
                if str(z.get('preprocess_signature','')) != preprocess_signature(self.cfg):
                    raise ValueError('NPZ preprocessing fingerprint differs from configuration')
            for s,slot in enumerate(self.slot_ids):
                for k in range(K):
                    p=f'slot_{slot}_s{k}'
                    valid[s,k]=bool(z.get(p+'_valid',0))
                    if valid[s,k]:
                        quality[s,k]=z.get(p+'_quality',np.array([1/(k+1),0,0,0,0,0,0,1]))
                        meta[s,k]=z.get(p+'_meta_codes',np.full(3,255))
            chosen=select_candidates(valid,quality,self.cfg.dataset.max_series_per_study if self.cfg and self.apply_budget else None,rng if self.training else None)
            for s,k in np.argwhere(chosen):
                p=f'slot_{self.slot_ids[s]}_s{k}'
                vol=z[p+'_volume']; zp=np.asarray(z[p+'_z'],float); count=len(vol)
                if vol.dtype != np.uint8:
                    if not np.isfinite(vol).all() or vol.min()<0 or vol.max()>1:
                        raise ValueError('float volume must contain finite intensities in [0,1]')
                    vol=np.round(vol*255).astype(np.uint8)
                if len(zp)!=count: zp=np.full(count,np.nan)
                coordinates_finite=bool(np.isfinite(zp).all())
                finite=bool(coordinates_finite and np.ptp(zp)>0)
                geom=finite and bool(z.get(p+'_geometry_valid',finite))
                source=z.get(p+'_source_indices',np.arange(count))
                if self.cfg:
                    sc=SamplingConfig(mode=self.cfg.sampling.mode,slices_per_slot=N)
                    pos=sample_positions(count,sc,zp) if sc.mode!='uniform_mm' or finite else sample_positions(count,SamplingConfig(slices_per_slot=N))
                    # Sample one center from each index bin during training.
                    if self.training and count>N:
                        if sc.mode=='uniform_mm' and finite:
                            bounds=np.linspace(zp.min(),zp.max(),N+1)
                            locations=rng.uniform(bounds[:-1],bounds[1:])
                            pos=np.sort(np.abs(zp[:,None]-locations[None,:]).argmin(0))
                        else:
                            bins=np.linspace(0,count,N+1).astype(int)
                            pos=np.array([rng.integers(bins[i],max(bins[i]+1,bins[i+1])) for i in range(N)])
                else:
                    pos=z[p+'_positions'].astype(int)
                positions[s,k]=pos
                neigh=np.clip(pos[:,None]+np.array([-1,0,1]),0,count-1)
                dz=np.abs(np.diff(zp)); positive=dz[np.isfinite(dz)&(dz>0.05)]
                typical=float(np.median(positive)) if len(positive) else 0
                gap_ratio=self.cfg.dataset.max_neighbor_gap_ratio if self.cfg else 1.8
                neighbor_ok=np.ones((N,3),bool)
                for channel in (0,2):
                    j=neigh[:,channel]
                    dist=np.abs(zp[j]-zp[pos])
                    ok=geom & (np.abs(source[j]-source[pos])==1) & (dist>0.05) & (dist<=max(typical*gap_ratio,0.05))
                    neighbor_ok[:,channel]=ok
                    neigh[:,channel]=np.where(ok,j,pos)
                if self.channel_mode == 'repeat':
                    neigh[:]=pos[:,None]
                unique,inverse=np.unique(neigh,return_inverse=True)
                resized=np.stack([cv2.resize(vol[j],(H,H),interpolation=cv2.INTER_AREA) for j in unique])
                groups=resized[inverse].reshape(N,3,H,H)
                cm=z.get(p+'_crop_mask',np.ones(vol.shape[1:],np.uint8))
                if cm.ndim == 2:
                    resized_mask = cv2.resize(cm,(H,H),interpolation=cv2.INTER_NEAREST)
                    masks = np.broadcast_to(resized_mask,(N,H,H))
                else:
                    masks=np.stack([cv2.resize(cm[j],(H,H),interpolation=cv2.INTER_NEAREST) for j in pos])
                if self.training and self.cfg and self.cfg.dataset.augment:
                    d=self.cfg.dataset
                    affine=cv2.getRotationMatrix2D(((H-1)/2,(H-1)/2),rng.uniform(-d.rotation_deg,d.rotation_deg),rng.uniform(1-d.scale_jitter,1+d.scale_jitter))
                    groups=np.stack([cv2.warpAffine(im,affine,(H,H),flags=cv2.INTER_LINEAR) for im in groups.reshape(-1,H,H)]).reshape(N,3,H,H)
                    masks=np.stack([cv2.warpAffine(im,affine,(H,H),flags=cv2.INTER_NEAREST) for im in masks])
                    gain=rng.uniform(1-d.intensity_jitter,1+d.intensity_jitter)
                    gamma=rng.uniform(1-d.intensity_jitter,1+d.intensity_jitter)
                    groups=np.clip((groups.astype(np.float32)/255)**gamma*gain*255,0,255).astype(np.uint8)
                images[s,k]=groups
                pixel_valid[s,k]=masks.astype(bool)
                seen=[]
                for t,j in enumerate(pos):
                    location=float(zp[j]) if coordinates_finite else int(j)
                    duplicate=any(abs(location-prior)<=0.05 for prior in seen) if coordinates_finite else location in seen
                    pad[s,k,t]=duplicate or not masks[t].any()
                    seen.append(location)
                if finite:
                    z_norm[s,k]=(zp[pos]-zp.min())/np.ptp(zp)
                    physical[s,k,:,0]=(zp[pos]-zp[0])/100
                    physical[s,k,:,1]=np.ptp(zp)/200
                    physical[s,k,:,2]=np.abs(zp[neigh[:,0]]-zp[pos])/10
                    physical[s,k,:,3]=np.abs(zp[neigh[:,2]]-zp[pos])/10
                    physical[s,k,:,4]=geom
                    physical[s,k,:,5]=np.sign(zp[-1]-zp[0])
                    if self.cfg and self.cfg.dataset.position_frame=='canonical':
                        direction=np.sign(zp[-1]-zp[0])
                        if direction:
                            z_norm[s,k]=(zp[pos]-zp[0])/(zp[-1]-zp[0])
                            physical[s,k,:,0]=(zp[pos]-zp[0])*direction/100
                elif self.cfg and self.cfg.dataset.position_frame=='canonical':
                    z_norm[s,k]=pos/max(1,count-1)
                quality[s,k,1]=geom
        series_pad=~chosen | pad.all(-1)
        targets,gold_mask=self.label_table.get(uid)
        target_mask = np.isfinite(targets).astype(np.float32)
        if self.training and series_pad.all():
            target_mask[:] = 0
        values={'images':images,'slice_pad':pad,'series_pad':series_pad,'positions':positions,
                'z_norm':z_norm,'physical':physical,'quality':quality,'pixel_valid':pixel_valid,
                'plane_codes':meta[:,:,0],'fluid_codes':meta[:,:,1],'fat_sup_codes':meta[:,:,2],
                'meta_codes':meta,'targets':targets,'target_mask':target_mask,
                'is_gold':np.asarray(gold_mask.all()),'gold_mask':gold_mask}
        return {**{key:torch.from_numpy(np.asarray(value)) for key,value in values.items()},'study_uid':uid}


def iter_study_uids(processed_dir):
    for path in sorted(glob.glob(os.path.join(processed_dir,'*.npz'))):
        yield os.path.basename(path)[:-4]
