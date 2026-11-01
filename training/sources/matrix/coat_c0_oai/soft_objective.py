"""Train-only interventions on a pinned canonical v152 student."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

MECHANISMS=('control','symmetric_power2','downward_power2','mass_equalized','mass_preserved')

def transform(target, mechanism):
    if mechanism=='symmetric_power2':
        a=target.square(); b=(1-target).square()
        return a/(a+b)
    if mechanism=='downward_power2':return target.square()
    return target

def install(trainer):
    trainer.DATA_KEYS |= {'soft_supervision','soft_manifest','mass_normalizer','mass_strength'}
    original_normalize=trainer.normalize_config
    original_loss=trainer.weighted_loss
    original_receipt=trainer.source_receipt
    original_smoke_subset=trainer.smoke_coverage_subset

    def normalize(c):
        cfg=original_normalize(c)
        if cfg.get('soft_supervision') not in MECHANISMS:raise ValueError('unknown soft mechanism')
        if cfg['loss']!='bce' or cfg['loss_normalization']!='global' or cfg.get('confidence_csv') or cfg.get('valid_mask_csv'):
            raise ValueError('first-round objective requires unchanged unit-weight global BCE')
        return cfg

    def weighted_loss(logits,targets,weights,pos_weight,cfg,group_denominators=None):
        method=cfg['soft_supervision']
        if method=='control':return original_loss(logits,targets,weights,pos_weight,cfg,group_denominators)
        if method not in ['mass_equalized','mass_preserved']:
            return original_loss(logits,transform(targets.float(),method),weights,pos_weight,cfg,group_denominators)
        x,y,w=logits.float(),targets.float(),weights.float()
        valid=torch.isfinite(y)&torch.isfinite(w)&(w>0)
        safe_y=torch.where(valid,y,torch.zeros_like(y)); safe_w=torch.where(valid,w,torch.zeros_like(w))
        # Weighted BCE = c(y) * BCE(x, w_pos*y/c(y)). Remove c(y),
        # preserving the per-cell optimum, then preserve each class's mean weight.
        norm=torch.as_tensor(cfg['mass_normalizer'],device=x.device,dtype=torch.float32)
        multiplier=(1+(pos_weight-1)*safe_y).reciprocal()/norm
        strength=float(cfg.get('mass_strength',1.0))
        assert 0.0 <= strength <= 1.0
        multiplier=(1.0-strength)+strength*multiplier
        values=F.binary_cross_entropy_with_logits(x,safe_y,pos_weight=pos_weight,reduction='none')
        values=torch.where(valid,values,torch.zeros_like(values))*safe_w*multiplier
        denominator=safe_w.sum(0) if group_denominators is None else group_denominators
        return values.sum()/denominator.sum().clamp_min(torch.finfo(torch.float32).eps)

    def receipt(output,cfg,cohort,synthetic=False):
        result=original_receipt(output,cfg,cohort,synthetic)
        for source in [Path(__file__),Path(__file__).with_name('train_soft.py')]:
            dest=output/'source/scripts'/source.name
            trainer.shutil.copy2(source,dest)
            result['sources']['scripts/'+source.name]=trainer.hash_file(source)
        manifest=Path(cfg['soft_manifest'])
        result['soft_supervision']={'mechanism':cfg['soft_supervision'],'manifest_sha256':trainer.hash_file(manifest),
            'positive_weights':'original untransformed training targets',
            'validation_targets':'original Astra, no transformation','fitted_only_on':('full4342 training cohort' if cfg['validation_strategy']=='full_refit' else 'training fold')}
        return result

    trainer.normalize_config=normalize
    trainer.weighted_loss=weighted_loss
    trainer.source_receipt=receipt
    # The parent smoke unconditionally covers Gold even when monitoring is off.
    # Preserve an empty disabled monitor without opening Gold58 for this goal.
    trainer.smoke_coverage_subset=lambda ids,labels,*a,**k: original_smoke_subset(ids,labels,*a,**k) if ids else []

def tests(trainer):
    install(trainer)
    cfg={'loss':'bce','loss_normalization':'global','soft_supervision':'control','mass_normalizer':[.6]*12}
    torch.manual_seed(42)
    x=torch.randn(8,12,requires_grad=True); y=torch.rand(8,12); weights=torch.ones_like(y); pos=torch.linspace(1,10,12)
    expected=F.binary_cross_entropy_with_logits(x,y,pos_weight=pos)
    actual=trainer.weighted_loss(x,y,weights,pos,cfg)
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    grid=torch.linspace(0,1,10001,dtype=torch.float64)
    for mechanism in MECHANISMS:
        q=transform(grid,mechanism)
        assert torch.isfinite(q).all() and bool((q[1:]>q[:-1]).all()) and q[0]==0 and q[-1]==1
        cfg['soft_supervision']=mechanism
        loss=trainer.weighted_loss(x,y,weights,pos,cfg)
        g=torch.autograd.grad(loss,x,retain_graph=True)[0]
        assert torch.isfinite(loss) and torch.isfinite(g).all()
        partition=sum(trainer.weighted_loss(x[i:i+4],y[i:i+4],weights[i:i+4],pos,cfg,weights.sum(0)) for i in (0,4))
        torch.testing.assert_close(loss,partition,rtol=1e-6,atol=1e-7)
    cfg['soft_supervision']='mass_equalized'
    optimum=(pos*y/(1+(pos-1)*y)).logit().detach().requires_grad_()
    loss=trainer.weighted_loss(optimum,y,weights,pos,cfg)
    assert torch.autograd.grad(loss,optimum)[0].abs().max()<1e-7
    print(json.dumps({'control_exact':True,'ordering_and_endpoints':True,'finite_losses_gradients':True,
                      'accumulation_denominators':True,'mass_optimum_preserved':True}))
