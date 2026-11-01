"""Conservative, fold-isolated MRI adaptation of an ImageNet CoAtNet.

Both objectives use a fixed ImageNet teacher only during pretraining. No
teacher, projector, or prediction head is exported for disease inference.
"""
from collections import OrderedDict, defaultdict
import json
import math
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


def validate_training_boundary(trained, allowed, protected):
    trained, allowed, protected = set(trained), set(allowed), set(protected)
    if not trained or not trained <= allowed or trained & protected:
        raise ValueError('pretraining images cross the downstream training boundary')


class DomainWindows(Dataset):
    """No disease labels; deterministic study/group, series, and depth sampling."""
    def __init__(self, cfg, all_ids, train_ids, cohort):
        self.cfg, self.source = cfg, cfg['source_dataset']
        self.seed = cfg['seed']; self.arrays = None; self.cache = OrderedDict()
        self.group_units = defaultdict(list)
        if self.source == 'KMAR-50K':
            self.root = Path(cfg['kmar_cache'])
            manifest = json.loads((self.root/'manifest.json').read_text())
            self.records = [r for r in manifest['records'] if r['split'] == 'train']
            for i,r in enumerate(self.records):
                # This is a filename-based patient proxy, not verified cross-dataset identity.
                name=Path(r['source_path']).name
                match=re.match(r'(\d{4}_\d+)_',name)
                self.group_units[match.group(1) if match else name].append(i)
            self.training_study_uids=[]
        elif self.source == 'RSNA-train-fold':
            import pandas as pd
            self.root=Path(cfg['corpus']); manifest=json.loads((self.root/'manifest.json').read_text())
            if manifest['study_uids'] != list(all_ids): raise ValueError('cache UID ordering differs')
            validate_training_boundary(train_ids,cohort['trained_study_uids'],cohort['protected_validation_study_uids'])
            self.training_study_uids=list(train_ids)
            rows={u:i for i,u in enumerate(all_ids)}
            folds=pd.read_csv(cfg['folds_csv']).set_index('StudyInstanceUID')
            self.records=[{'uid':u,'row':rows[u]} for u in train_ids]
            for i,r in enumerate(self.records):self.group_units[str(folds.loc[r['uid'],'group'])].append(i)
        elif self.source in {'MRNet', 'OAI-3DESS', 'fastMRI-knee', 'KneeMRI'}:
            self.root = Path(cfg['external_root'])
            self.records = self._external_records()
            for i, record in enumerate(self.records):
                self.group_units[str(record.get('group', i))].append(i)
            self.training_study_uids = []
        else: raise ValueError('unknown domain dataset')
        self.groups=sorted(self.group_units)

    def _external_records(self):
        if self.source == 'MRNet':
            files = sorted((self.root / 'MRNet-v1.0' / 'train').glob('*/*.npy'))
            if not files:
                files = sorted((self.root / 'train').glob('*/*.npy'))
            return [{'path': str(path.relative_to(self.root)), 'group': path.stem,
                     'kind': 'mrnet'} for path in files]
        if self.source == 'OAI-3DESS':
            files = sorted(self.root.glob('*.npy'))
            if not files:
                raise ValueError(f'no OAI 3D-DESS arrays found under {self.root}')
            records = []
            # Kaggle's OAI export stores all studies in one array per class.
            # Materialize virtual study records so minibatches draw independent
            # patients instead of treating the two class files as two groups.
            for path in files:
                array = np.load(path, mmap_mode='r')
                count = int(array.shape[0]) if array.ndim >= 3 else 1
                for index in range(count):
                    records.append({'path': str(path.relative_to(self.root)),
                                    'array_index': index if array.ndim >= 3 else None,
                                    'group': f'{path.stem}_{index:06d}',
                                    'kind': 'oai_array'})
            return records
        if self.source == 'KneeMRI':
            files = sorted(self.root.rglob('*.pck'))
            if not files:
                raise ValueError(f'no KneeMRI pickle volumes found under {self.root}')
            return [{'path': str(path.relative_to(self.root)), 'group': path.stem,
                     'kind': 'kneemri'} for path in files]
        files = sorted(self.root.rglob('*.h5')) + sorted(self.root.rglob('*.hdf5'))
        if not files:
            raise ValueError(f'no fastMRI HDF5 files found under {self.root}')
        return [{'path': str(path.relative_to(self.root)), 'group': path.stem,
                 'kind': 'fastmri'} for path in files]

    @staticmethod
    def _to_triplet(volume, rng):
        import cv2
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim == 2:
            volume = volume[None]
        if volume.ndim != 3:
            raise ValueError(f'expected a 2D/3D MRI array, got {volume.shape}')
        # mmap-backed Kaggle arrays are read-only; copy before replacing NaNs.
        volume = np.nan_to_num(volume, copy=True)
        lo, hi = np.percentile(volume, [1., 99.])
        if not np.isfinite(lo + hi) or hi <= lo:
            raise ValueError('degenerate external MRI intensity')
        volume = np.clip((volume - lo) / (hi - lo), 0., 1.)
        n = len(volume)
        center = int(rng.integers(1, max(2, n - 1))) if n > 2 else n // 2
        indices = np.clip([center - 1, center, center + 1], 0, n - 1)
        frames = [cv2.resize(volume[index], (384, 384), interpolation=cv2.INTER_AREA)
                  for index in indices]
        return np.asarray(np.clip(np.asarray(frames) * 255. + .5, 0, 255), dtype=np.uint8)

    def __len__(self): return len(self.groups)

    def __getitem__(self,key):
        step,group_index,draw=key
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,group_index,draw]))
        units=self.group_units[self.groups[group_index]]
        index=int(rng.choice(units)); record=self.records[index]
        if self.source == 'KMAR-50K':
            path=record['cache_path']
            if path not in self.cache:
                self.cache[path]=np.load(self.root/path,mmap_mode='r')
                if len(self.cache)>24:self.cache.popitem(last=False)
            vol=self.cache[path]; n=len(vol)
            # Stratification covers the anatomical middle 80%, rather than one fixed central slice.
            depth_bin=(step+draw)%4
            low=.1+.2*depth_bin; high=low+.2
            center=int(round(rng.uniform(low,high)*(n-1)))
            positions=np.clip([center-1,center,center+1],0,n-1)
            x=np.array(vol[positions],copy=True)
        elif self.source == 'RSNA-train-fold':
            if self.arrays is None:
                self.arrays={k:np.load(self.root/(k+'.npy'),mmap_mode='r') for k in ['windows','masks','group_id']}
            row=record['row']; candidates=np.flatnonzero(self.arrays['masks'][row])
            if not len(candidates):raise ValueError('empty study in pretraining cohort')
            slots=self.arrays['group_id'][row,candidates]
            group=rng.choice(np.unique(slots)); candidates=candidates[slots==group]
            slot=int(rng.choice(candidates));x=np.array(self.arrays['windows'][row,slot],copy=True)
        else:
            record = self.records[index]
            path = self.root / record['path']
            if record['kind'] == 'mrnet':
                volume = np.load(path, mmap_mode='r')
            elif record['kind'] == 'oai_array':
                array = np.load(path, mmap_mode='r')
                index = record.get('array_index')
                volume = array[index] if index is not None else array
            elif record['kind'] == 'kneemri':
                import pickle
                with path.open('rb') as handle:
                    volume = pickle.load(handle)
            else:
                try:
                    import h5py
                except ImportError as exc:
                    raise RuntimeError('fastMRI pretraining requires h5py') from exc
                with h5py.File(path, 'r') as handle:
                    key = next((name for name in ('reconstruction_rss', 'reconstruction_esc', 'kspace')
                                if name in handle), None)
                    if key is None:
                        raise ValueError(f'no supported fastMRI image key in {path}')
                    volume = np.asarray(handle[key])
                    if np.iscomplexobj(volume):
                        volume = np.abs(volume)
            x = self._to_triplet(volume, rng)
        return torch.from_numpy(x), group_index


class FixedStepBatches(Sampler):
    def __init__(self,dataset,batch_size,steps,start=0):
        self.dataset,self.batch_size,self.steps,self.start=dataset,batch_size,steps,start
        if batch_size<2 or batch_size>len(dataset):raise ValueError('insufficient independent sampling groups')
    def __len__(self):return self.steps-self.start
    def __iter__(self):
        for step in range(self.start,self.steps):
            rng=np.random.default_rng(np.random.SeedSequence([self.dataset.seed,step,991]))
            groups=rng.choice(len(self.dataset),self.batch_size,replace=False)
            yield [(step,int(group),i) for i,group in enumerate(groups)]


def geometric_view(images):
    x=images.float()/255.;n=len(x)
    angle=torch.empty(n,device=x.device).uniform_(-5,5)*math.pi/180
    scale=torch.empty(n,device=x.device).uniform_(.95,1.05)
    theta=x.new_zeros(n,2,3)
    theta[:,0,0]=angle.cos()*scale;theta[:,0,1]=-angle.sin()*scale
    theta[:,1,0]=angle.sin()*scale;theta[:,1,1]=angle.cos()*scale
    theta[:,:,2]=torch.empty(n,2,device=x.device).uniform_(-.025,.025)
    return F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False).clamp(0,1)


def intensity_view(x):
    gamma=torch.empty(len(x),1,1,1,device=x.device).uniform_(.9,1.1)
    gain=torch.empty_like(gamma).uniform_(.95,1.05)
    return (x.pow(gamma)*gain).clamp(0,1)*2-1


def foreground_mask(x,ratio=.4,stride=32):
    foreground=F.avg_pool2d((x.mean(1,keepdim=True)>.035).float(),stride,stride)[:,0]>.15
    mask=torch.zeros_like(foreground)
    for i in range(len(x)):
        indices=foreground[i].flatten().nonzero().flatten()
        if len(indices)<2:
            # Keep the cached valid-window definition unchanged. A nearly empty
            # image contributes neither masked nor visible feature targets.
            foreground[i]=False
            continue
        count=min(len(indices)-1,max(1,round(len(indices)*ratio)))
        selected=indices[torch.randperm(len(indices),device=x.device)[:count]]
        mask[i].flatten()[selected]=True
    if not mask.any():raise ValueError('entire batch lacks foreground for masked pretraining')
    pixel=F.interpolate(mask[:,None].float(),size=x.shape[-2:],mode='nearest')
    return mask,foreground,pixel


def feature_loss(student,target,mask=None):
    # Every spatial position is compared in the same geometry, across channels.
    a=F.normalize(student.float(),dim=1);b=F.normalize(target.detach().float(),dim=1)
    loss=1-(a*b).sum(1)
    return loss[mask].mean() if mask is not None else loss.mean()


def freeze_pretraining_layers(backbone):
    backbone.stem.requires_grad_(False);backbone.stages[0].requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.requires_grad_(False);module.eval()


class DomainModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from kneemri.coatnet_model import build_model
        cfg=dict(cfg,gradient_checkpointing=False)
        # Build twice with the same seed; all start from strictly loaded ImageNet tensors.
        self.backbone=build_model(cfg,pretrained=True).backbone
        self.teacher=build_model(cfg,pretrained=True).backbone
        self.teacher.requires_grad_(False);self.teacher.eval()
        freeze_pretraining_layers(self.backbone)
        self.method=cfg['method'];self.mask_ratio=cfg['mask_ratio'];self.anchor_weight=cfg['anchor_weight']
        dim=self.backbone.num_features
        if self.method=='masked_features':
            self.predictor=nn.Conv2d(dim,dim,1)
            nn.init.zeros_(self.predictor.weight);nn.init.zeros_(self.predictor.bias)
        elif self.method=='anchored_vicreg':
            self.projector=nn.Sequential(nn.Linear(dim,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,256))
        else:raise ValueError('unsupported adaptation objective')

    def train(self,mode=True):
        super().train(mode);self.teacher.eval()
        self.backbone.stem.eval();self.backbone.stages[0].eval()
        for module in self.backbone.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def forward(self,images):
        from kneemri.kmar_pretraining import vicreg_loss
        geometry=geometric_view(images);a=intensity_view(geometry)
        with torch.no_grad():target=self.teacher.forward_features(a)
        if self.method=='masked_features':
            mask,foreground,pixel=foreground_mask(geometry,self.mask_ratio)
            maps=self.backbone.forward_features(a*(1-pixel))
            prediction=maps+self.predictor(maps)
            masked=feature_loss(prediction,target,mask)
            visible=feature_loss(maps,target,foreground & ~mask)
            loss=masked+.25*visible
            return loss,{'masked_feature_loss':masked,'visible_anchor_loss':visible,'feature_cosine':1-feature_loss(maps,target),
                         'masked_fraction':mask.float().mean(),'masked_valid_images':mask.flatten(1).any(1).sum(),
                         'masked_skipped_images':(~mask.flatten(1).any(1)).sum()}
        b=intensity_view(geometry)
        maps=self.backbone.forward_features(a);other=self.backbone.forward_features(b)
        fa=self.backbone.forward_head(maps);fb=self.backbone.forward_head(other)
        za=self.projector(fa);zb=self.projector(fb)
        ssl,terms=vicreg_loss(za,zb)
        spatial_anchor=feature_loss(maps,target)
        with torch.no_grad():ft=self.teacher.forward_head(target)
        global_anchor=1-F.cosine_similarity(fa.float(),ft.float()).mean()
        loss=ssl/25+self.anchor_weight*(global_anchor+spatial_anchor)
        return loss,{'vicreg_scaled':ssl/25,'anchor_loss':global_anchor+spatial_anchor,
                    'feature_cosine':1-global_anchor,**terms}
