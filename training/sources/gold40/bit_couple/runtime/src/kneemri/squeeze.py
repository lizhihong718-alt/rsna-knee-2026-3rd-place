"""Small continuation adapter over the pinned J_v3 trainer; deployment uses original head."""
from pathlib import Path
import json, math, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

KEYS={'squeeze_mode','squeeze_cosine_floor','squeeze_expert_csv','squeeze_expert_weights_csv','squeeze_schedule'}

class ClassReadout(nn.Module):
    def __init__(self, model, independent=False):
        super().__init__()
        self.weight=nn.Parameter(model.clsW.detach().clone() if independent else torch.zeros_like(model.clsW))
        self.bias=nn.Parameter(model.clsb.detach().clone() if independent else torch.zeros_like(model.clsb))
    def forward(self, h):
        return (h.float()*self.weight.float()).sum(-1)+self.bias.float()

def replacement(logits, auxiliary, targets, weights, pw, cfg, denominator, mask, expert, teacher, delta):
    """Keep K(m) once. Only replace the existing explicit expert/teacher cells."""
    u=logits.float()+delta
    m=targets.float()
    k=weights.float()/(1+(pw-1)*m)/torch.as_tensor(cfg['mass_normalizer'],device=u.device)
    bce=lambda x,y:F.binary_cross_entropy_with_logits(x,y,pos_weight=pw,reduction='none')
    old=bce(u,m)
    mode=cfg['squeeze_mode']
    if mode=='M1_SHAM': new=.825*old+.175*bce(u+auxiliary,m)
    elif mode=='M2_EXPERT': new=.525*bce(u,expert)+.3*bce(u,teacher)+.175*bce(u+auxiliary,expert)
    elif mode=='M3_FREE': new=.3*bce(u,teacher)+.7*bce(auxiliary+delta,expert)
    else: raise ValueError(mode)
    denom=(weights.sum() if denominator is None else denominator.sum()).clamp_min(torch.finfo(torch.float32).eps)
    return torch.where(mask,(new-old)*k,torch.zeros_like(k)).sum()/denom

def install(trainer):
    trainer.DATA_KEYS.update(KEYS)
    old_normalize=trainer.normalize_config
    def normalize(cfg):
        assert cfg.get('seed') in (2026, 2027, 2028, 2029, 2030)  # runtime_nq4: more seed replicates
        if cfg.get('squeeze_schedule')=='full25':
            assert cfg['epochs']==25 and cfg['scheduler']=='onecycle'
            return old_normalize(cfg)
        assert cfg.get('scheduler')=='cosine' and cfg.get('squeeze_cosine_floor')==.2
        normalized=old_normalize(dict(cfg,warmup_fraction=.15))
        normalized['warmup_fraction']=0.
        return normalized
    trainer.normalize_config=normalize
    old_build,old_loss,old_receipt=trainer.build_model,trainer.weighted_loss,trainer.source_receipt
    old_cohort=trainer.load_cohort
    old_scheduler=trainer.create_scheduler
    state=trainer._oai_state
    models={}
    def cohort(cfg):
        result=old_cohort(cfg)
        if cfg.get('squeeze_mode','').startswith('M'):
            e=pd.read_csv(cfg['squeeze_expert_csv'],index_col=0)[trainer.LABEL_COLS]
            w=pd.read_csv(cfg['squeeze_expert_weights_csv'],index_col=0)[trainer.LABEL_COLS]
            state['expert']={u:row.to_numpy(np.float32) for u,row in e.iterrows()}
            state['expert_mask']={u:row.to_numpy(np.float32)>0 for u,row in w.iterrows()}
        return result
    def build(cfg,pretrained=False):
        model=old_build(cfg,pretrained)
        if cfg.get('squeeze_schedule')!='full25':
            for name,p in model.named_parameters():
                p.requires_grad_(not name.startswith('backbone.') or name.startswith(('backbone.stages.3.','backbone.norm.')))
        mode=cfg.get('squeeze_mode','C0_CONT')
        if mode.startswith('M'):
            model.squeeze_aux=ClassReadout(model,independent=mode=='M3_FREE')
        models['model']=model
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        assert any(n.startswith('backbone.stages.3.') for n in names)
        if cfg.get('squeeze_schedule')!='full25':
            assert not any(n.startswith(('backbone.stem.','backbone.stages.0.','backbone.stages.1.','backbone.stages.2.')) for n in names)
        print(json.dumps({'event':'squeeze_trainable','names':names,'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}),flush=True)
        return model
    from kneemri.coatnet_model import CoAtNetAblationModel
    old_pool=CoAtNetAblationModel._pool
    def pool(self,*args,**kwargs):
        h=old_pool(self,*args,**kwargs)
        if hasattr(self,'squeeze_aux') and self.training:
            self._squeeze_private_logits=self.squeeze_aux(h)
        return h
    CoAtNetAblationModel._pool=pool
    def loss(logits,targets,weights,pos_weight,cfg,group_denominators=None):
        base=old_loss(logits,targets,weights,pos_weight,cfg,group_denominators)
        if not cfg.get('squeeze_mode','').startswith('M'): return base
        ids=state['current_ids']; device=logits.device
        mask=np.stack([state['expert_mask'].get(u,np.zeros(12,bool)) for u in ids])
        expert=np.stack([state['expert'].get(u,np.zeros(12,np.float32)) for u in ids])
        mask=torch.as_tensor(mask,device=device)&(weights>0)&torch.isfinite(targets)
        expert=torch.as_tensor(np.nan_to_num(expert),device=device)
        # Targets are the serialized six-decimal mixture; recover its exact teacher
        # component to retain the original loss/gradient at zero residual.
        teacher=torch.where(mask,(targets.float()-.7*expert)/.3,torch.zeros_like(targets)).clamp(0,1)
        auxiliary=models['model']._squeeze_private_logits
        if not logits.requires_grad: auxiliary=auxiliary.detach()
        rows=torch.tensor([u.startswith('OAI.') for u in ids],device=device)[:,None]
        delta=rows.float()*state['delta'].to(device)[None]
        return base+replacement(logits,auxiliary,targets,weights,pos_weight,cfg,group_denominators,mask,expert,teacher,delta)
    def scheduler(optimizer,cfg,updates_per_epoch):
        if cfg.get('squeeze_schedule')=='full25': return old_scheduler(optimizer,cfg,updates_per_epoch)
        steps=cfg['epochs']*updates_per_epoch
        floor=float(cfg['squeeze_cosine_floor'])
        return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:floor+(1-floor)*.5*(1+math.cos(math.pi*min(n,steps)/steps)))
    def clip(model,max_norm):
        main=[p for n,p in model.named_parameters() if not n.startswith('squeeze_aux.')]
        aux=[p for n,p in model.named_parameters() if n.startswith('squeeze_aux.')]
        norm=torch.nn.utils.clip_grad_norm_(main,max_norm)
        if aux:
            aux_norm=torch.nn.utils.clip_grad_norm_(aux,max_norm)
            if not torch.isfinite(aux_norm): raise FloatingPointError('nonfinite auxiliary gradient')
        return norm
    def receipt(output,cfg,cohort,synthetic=False):
        r=old_receipt(output,cfg,cohort,synthetic)
        for source in [Path(__file__),Path(__file__).parents[2]/'scripts/train_squeeze.py']:
            rel=source.relative_to(Path(__file__).parents[2]); dst=output/'source'/rel
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dst)
            r['sources'][str(rel)]=trainer.hash_file(source)
        r['squeeze']={'seed':2026,'fixed_epochs':cfg['epochs'],'optimizer_reset':True,
                      'trainable':'all original parameters' if cfg.get('squeeze_schedule')=='full25' else 'last BiT stage, final backbone norm, depth, MIL, head',
                      'base_gradient_clip_separate_from_aux':True,'schedule':cfg.get('squeeze_schedule','continuation5'),
                      'cosine_floor':cfg.get('squeeze_cosine_floor'),'deployment':'original 12-class readout only'}
        return r
    trainer.load_cohort=cohort; trainer.build_model=build; trainer.weighted_loss=loss
    trainer.create_scheduler=scheduler; trainer.squeeze_clip_gradients=clip; trainer.source_receipt=receipt
