"""Image training, including frozen spatial queries and partial DINO fine-tuning."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import random
import time
import math
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader
from kneemri.config import abspath
from kneemri.dataset import KneeMRIDataset, LABEL_COLS
from kneemri.model import build_model
from kneemri.losses import CombinedLoss
from kneemri.metrics import evaluate_predictions, format_report
from kneemri.train import move_batch


def forward_images(model,batch,architecture):
    if architecture=='patch_query':
        return model(batch)
    return model(batch['images'],batch['z_norm'],batch['slice_pad'],batch['series_pad'],batch['plane_codes'],batch['fluid_codes'],batch['fat_sup_codes'])


def save_checkpoint(path,payload):
    path=Path(path)
    torch.save(payload,str(path)+'.tmp')
    Path(str(path)+'.tmp').replace(path)


def train_image_fold(cfg,fold,folds,out_dir,epochs=8,batch_size=1,lr=3e-4,
                     encoder_lr=1e-5,num_workers=2,device=None,limit_train=0,
                     limit_val=0,max_steps=0,resume=None,init_checkpoint=None,
                     accumulation_steps=1,warmup_epochs=0.5,log_every=20):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    from kneemri.cv import fold_split
    fold_split(folds,fold)
    if accumulation_steps<1 or epochs<1 or warmup_epochs<0:
        raise ValueError('invalid training schedule')
    if accumulation_steps>1 and cfg.loss.lambda_rank:
        raise ValueError('accumulation does not create cross-microbatch AUC pairs; disable rank loss')
    if resume and init_checkpoint:
        raise ValueError('choose resume or init_checkpoint')
    def dataset(training,limit):
        part=folds[folds.fold.ne(fold) if training else folds.fold.eq(fold)]
        files=[str(Path(abspath(cfg.dataset.processed_dir))/(u+'.npz')) for u in part.StudyInstanceUID]
        if not limit and any(not Path(p).exists() for p in files):
            raise FileNotFoundError('preprocessing is incomplete; complete it before full training')
        files=[p for p in files if Path(p).exists()]
        if limit: files=files[:limit]
        ds=KneeMRIDataset(cfg.dataset.processed_dir,cfg=cfg,training=training,files=files)
        ds.label_table.validate_coverage([ds.study_uid(i) for i in range(len(ds))])
        return ds
    train_ds=dataset(True,limit_train);val_ds=dataset(False,limit_val)
    if not len(train_ds) or not len(val_ds): raise ValueError('empty train/validation split')
    loader_args={'num_workers':num_workers,'pin_memory':device.startswith('cuda')}
    if num_workers: loader_args['prefetch_factor']=1
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,**loader_args)
    val_dl=DataLoader(val_ds,batch_size=batch_size,**loader_args)
    model=build_model(cfg,len(train_ds.slot_ids),train_ds.k_series,train_ds.slices_per_slot).to(device)
    val_uids=[val_ds.study_uid(i) for i in range(len(val_ds))]
    if init_checkpoint:
        initialize_model(model,cfg,init_checkpoint,val_uids)
    encoder_params=[p for p in model.encoder.parameters() if p.requires_grad]
    rest=[p for name,p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    groups=[{'params':rest,'lr':lr}]
    if encoder_params: groups.append({'params':encoder_params,'lr':encoder_lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    steps=min(len(train_dl),max_steps) if max_steps else len(train_dl)
    updates=math.ceil(steps/accumulation_steps)
    total_updates=epochs*updates; warmup_updates=round(warmup_epochs*updates)
    def lr_multiplier(step):
        if step<warmup_updates: return (step+1)/max(1,warmup_updates)
        progress=min(1,max(0,(step-warmup_updates)/max(1,total_updates-warmup_updates)))
        return 0.05+0.95*0.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
    criterion=CombinedLoss(cfg.loss.gold_weight,cfg.loss.pseudo_default_weight,cfg.loss.lambda_rank,cfg.loss.normalization)
    schedule={'epochs':epochs,'batch_size':batch_size,'accumulation_steps':accumulation_steps,
              'steps_per_epoch':steps,'warmup_epochs':warmup_epochs,'lr':lr,'encoder_lr':encoder_lr}
    print(f'[images] backbone={cfg.model.encoder.backbone_name} '
          f'trainable_encoder={sum(p.numel() for p in encoder_params):,} '
          f'train={len(train_ds)} val={len(val_ds)} schedule={schedule}',flush=True)
    amp=device.startswith('cuda')
    dtype=amp_dtype(cfg,device)
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    best=-float('inf');start=0;history=[]
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        from kneemri.config import config_from_dict
        if asdict(config_from_dict(state['pipeline_config']))!=asdict(cfg) or state['slot_ids']!=train_ds.slot_ids or state.get('schedule')!=schedule:
            raise ValueError('resume configuration/slot manifest differs')
        if state.get('label_fingerprints') != train_ds.label_table.fingerprints:
            raise ValueError('resume labels differ from the checkpoint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        start=state['epoch']+1;best=state['best_auc'];history=state['history']
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if amp and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,epochs):
        train_ds.set_epoch(epoch);model.train();losses=[];t0=time.time()
        batches=iter(islice(train_dl,steps)); step=0
        while True:
            window=list(islice(batches,accumulation_steps))
            if not window: break
            denominators=[float(criterion.soft_bce.denominator(b['targets'],b['target_mask'],b['gold_mask'])) for b in window]
            denominator=sum(denominators)
            if denominator<=0:
                step+=len(window)
                continue
            optimizer.zero_grad(set_to_none=True)
            for batch,weight in zip(window,denominators):
                batch=move_batch(batch,device)
                with torch.autocast(device_type='cuda' if amp else 'cpu',dtype=dtype,enabled=amp):
                    logits=forward_images(model,batch,cfg.model.architecture)
                    loss,_=criterion(logits.float(),batch['targets'],batch['target_mask'],batch['gold_mask'])
                if not torch.isfinite(loss): raise FloatingPointError('non-finite training loss')
                scaler.scale(loss*(weight/denominator)).backward()
                losses.append(float(loss.detach()));step+=1
            scaler.unscale_(optimizer)
            # GradScaler has already recorded overflow during unscale_; let it
            # skip that update and lower the scale instead of aborting FP16 training.
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=not scaler.is_enabled())
            old_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old_scale: scheduler.step()
            if log_every and (step%log_every<len(window) or step==steps):
                print(f'[images fold {fold}] epoch {epoch} step {step}/{steps} loss {np.mean(losses[-log_every:]):.5f}',flush=True)
        model.eval();scores=[];targets=[];masks=[];gold=[];uids=[]
        with torch.no_grad():
            for batch in val_dl:
                batch=move_batch(batch,device)
                with torch.autocast(device_type='cuda' if amp else 'cpu',dtype=dtype,enabled=amp):
                    score=forward_images(model,batch,cfg.model.architecture)
                score=score.masked_fill(batch['series_pad'].flatten(1).all(1)[:,None],0)
                scores.append(score.float().cpu().numpy());targets.append(batch['targets'].cpu().numpy())
                masks.append(batch['target_mask'].cpu().numpy());gold.append(batch['is_gold'].cpu().numpy());uids+=batch['study_uid']
        pred=np.concatenate(scores);target=np.concatenate(targets);mask=np.concatenate(masks);is_gold=np.concatenate(gold)
        report=evaluate_predictions(pred,target,mask,is_gold)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),**report,'elapsed_s':time.time()-t0})
        score=report['macro_auc'];improved=np.isfinite(score) and score>best
        if improved: best=score
        payload={'schema_version':2,'architecture':cfg.model.architecture,'pipeline_config':asdict(cfg),
                 'slot_ids':train_ds.slot_ids,'label_cols':LABEL_COLS,'model':model.state_dict(),
                 'model_kind':'image','optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'scaler':scaler.state_dict(),'epoch':epoch,'metrics':report,'best_auc':best,'history':history,
                 'schedule':schedule,'label_fingerprints':train_ds.label_table.fingerprints,
                 'train_uids':[train_ds.study_uid(i) for i in range(len(train_ds))],'val_uids':uids,
                 'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if amp else None}
        save_checkpoint(out/f'fold{fold}_last.pt',payload)
        if improved or not (out/f'fold{fold}_best.pt').exists():
            save_checkpoint(out/f'fold{fold}_best.pt',payload)
            np.savez(out/f'fold{fold}_oof.npz',logits=pred,targets=target,label_mask=mask,is_gold=is_gold,study_uids=uids)
        (out/f'fold{fold}_history.json').write_text(json.dumps(history,indent=2))
        print(format_report(report),flush=True)
    return {'best_macro_auc':best,'history':history}


def amp_dtype(cfg,device):
    requested=cfg.model.amp_dtype
    if requested not in ('auto','float16','bfloat16'):
        raise ValueError('unknown AMP dtype')
    bf16=str(device).startswith('cuda') and torch.cuda.is_bf16_supported()
    if requested=='bfloat16' and not bf16:
        raise ValueError('requested bfloat16 is unavailable on this device')
    return torch.bfloat16 if bf16 and requested!='float16' else torch.float16


def initialize_model(model,cfg,path,val_uids):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('schema_version')!=2 or state.get('label_cols')!=LABEL_COLS:
        raise ValueError('incompatible initialization checkpoint')
    if state['slot_ids']!=list(cfg.slots.definitions) or state['architecture']!=cfg.model.architecture:
        raise ValueError('initialization slot/architecture differs')
    if set(val_uids)-set(state.get('val_uids',[])):
        raise ValueError('initialization checkpoint did not hold out every validation study')
    if set(val_uids)&set(state.get('train_uids',[])):
        raise ValueError('initialization checkpoint saw validation studies')
    if state.get('model_kind')=='image':
        old_encoder={k[len('encoder.'):]:v for k,v in state['model'].items() if k.startswith('encoder.')}
    else:
        old_encoder=state['encoder']
    for key in ('mean','std'):
        if key in old_encoder and not torch.allclose(old_encoder[key],getattr(model.encoder,key).cpu()):
            raise ValueError('initialization encoder normalization differs from configuration')
    if state.get('model_kind')=='image':
        model.load_state_dict(state['model'])
    else:
        model.encoder.load_state_dict(state['encoder'])
        model.aggregator.load_state_dict(state['model'])
    print(f'[images] initialized from {path}; validation study exclusion verified',flush=True)
