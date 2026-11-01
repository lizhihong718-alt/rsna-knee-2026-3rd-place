"""Offline prediction from a self-contained v2 checkpoint."""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from kneemri.config import config_from_dict
from kneemri.dataset import KneeMRIDataset, LABEL_COLS
from kneemri.image_train import forward_images
from kneemri.model import build_model
from kneemri.train import move_batch


def load_checkpoint(path,device='cpu'):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if ckpt.get('schema_version')!=2 or ckpt.get('label_cols')!=LABEL_COLS:
        raise ValueError('legacy/incompatible checkpoint: retrain using corrected v2 pipeline')
    cfg=config_from_dict(ckpt['pipeline_config'])
    if ckpt['slot_ids']!=list(cfg.slots.definitions):
        raise ValueError('checkpoint slot manifest is inconsistent')
    cfg.model.encoder.pretrained=False;cfg.model.encoder.weights=None
    model=build_model(cfg,len(ckpt['slot_ids']),cfg.slots.k_series_per_slot,cfg.dataset.centers_per_series or cfg.sampling.slices_per_slot)
    if ckpt.get('model_kind')=='image' or cfg.model.architecture=='patch_query':
        model.load_state_dict(ckpt['model'])
    else:
        model.encoder.load_state_dict(ckpt['encoder'])
        model.aggregator.load_state_dict(ckpt['model'])
        model.cached_precision=True  # FP16 encoder, FP32 aggregator, matching cache training
    return model.to(device).eval(),cfg


def preprocess_test(cfg,data_root,output_dir,study_uids):
    import cv2
    cv2.setNumThreads(1)
    from kneemri.headers import study_frame
    from kneemri.dicom import decode_pixel_array
    from kneemri.pipeline import process_study,save_study,preprocess_signature,StudyResult
    cfg=copy.deepcopy(cfg);cfg.blacklist=[]
    table=pd.read_csv(Path(data_root)/'test_series.csv')
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    for i,uid in enumerate(study_uids):
        dest=Path(output_dir)/(uid+'.npz')
        if dest.exists():
            with np.load(dest) as z:
                if str(z.get('preprocess_signature',''))==preprocess_signature(cfg): continue
        try:
            root=Path(data_root)/'test_series'/uid
            meta=study_frame(str(root));cfg.study_root=str(root)
            result=process_study(uid,cfg,meta,table[table.StudyInstanceUID.eq(uid)],decode_pixel_array)
        except Exception as exc:
            result=StudyResult(uid,failed=True,error=f'{type(exc).__name__}: {exc}')
        save_study(result,cfg,out_dir=str(output_dir))
        if result.failed: print(f'[preprocess] {uid}: {result.error}',flush=True)
        if (i+1)%50==0: print(f'[preprocess] {i+1}/{len(study_uids)}',flush=True)


@torch.no_grad()
def predict_studies(model,cfg,processed_dir,study_uids,device='cpu',batch_size=1,workers=0):
    cfg=copy.deepcopy(cfg);cfg.dataset.labels_csv='';cfg.dataset.train_csv='';cfg.dataset.augment=False
    cfg.dataset.require_targets=False
    files=[str(Path(processed_dir)/(uid+'.npz')) for uid in study_uids]
    if len(set(study_uids))!=len(study_uids): raise ValueError('duplicate submission study IDs')
    ds=KneeMRIDataset(str(processed_dir),cfg=cfg,files=files)
    dl=DataLoader(ds,batch_size=batch_size,num_workers=workers)
    values=[];seen=[]
    amp=str(device).startswith('cuda')
    from kneemri.image_train import amp_dtype
    dtype=amp_dtype(cfg,device)
    for batch in dl:
        batch=move_batch(batch,device)
        with torch.autocast(device_type='cuda' if amp else 'cpu',dtype=dtype,enabled=amp and not getattr(model,'cached_precision',False)):
            logits=forward_images(model,batch,cfg.model.architecture)
        # Explicit neutral fallback for a study with no decoded images.
        logits=torch.where(batch['series_pad'].flatten(1).all(1)[:,None],torch.zeros_like(logits),logits)
        probs=logits.float().sigmoid().cpu().numpy()
        if not np.isfinite(probs).all(): raise FloatingPointError('non-finite submission prediction')
        values.append(probs);seen+=batch['study_uid']
    result=pd.DataFrame(np.concatenate(values) if values else np.empty((0,12)),columns=LABEL_COLS)
    result.insert(0,'StudyInstanceUID',seen)
    if seen!=list(study_uids): raise ValueError('submission ordering mismatch')
    return result
