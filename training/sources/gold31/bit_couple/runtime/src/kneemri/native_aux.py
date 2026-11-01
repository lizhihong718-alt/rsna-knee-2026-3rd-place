"""Native MOAKS 14-region A/F cumulative ordinal auxiliary on shared image features."""
from pathlib import Path
import hashlib,json,math,shutil,time,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KEYS={'native_labels','native_mask','native_warmup_dir','native_lambda_path','native_warmup_epochs','native_warmup_lr','native_couple'}
# runtime_nq5 (10-19): native_couple -> each compartment OA logit (Medial / Lateral / PF OA) gets + a_c * z_c, where
# z_c = max over the compartment's MOAKS subregions of the auxiliary head's expected full-thickness grade
# (sum of the 3 cumulative sigmoids, detached: the auxiliary head stays MOAKS-supervised only) and a_c is a learned
# scalar initialised at 0 (so the run starts exactly as the uncoupled one). The main heads decide on competition
# labels how much the MOAKS-grounded cartilage readout counts. Compartments whose regions are absent are not coupled.
COUPLE={'Medial OA':['FMC','FMP','TMA','TMC','TMP'],'Lateral OA':['FLC','FLP','TLA','TLC','TLP'],'PF OA':['PM','PL','FMA','FLA']}

class NativeOrdinal(nn.Module):
    def __init__(self,dim,fields=28):  # runtime_nq3: field count follows the label file (28 = native14_af)
        super().__init__()
        self.score=nn.Linear(dim,fields)
        self.query_logits=nn.Parameter(torch.zeros(fields,12))
        self.first=nn.Parameter(torch.full((fields,),-.5))
        self.gaps=nn.Parameter(torch.full((fields,2),math.log(math.expm1(1.))))
    def forward(self,h):
        # All thresholds share a score per region/field and are ordered by construction.
        offsets=torch.cat([torch.zeros_like(self.first[:,None]),F.softplus(self.gaps).cumsum(-1)],-1)
        threshold=self.first[:,None]+offsets
        assert h.ndim==3 and h.shape[1]==12
        mixing=self.query_logits.float().softmax(-1)
        regional=torch.einsum('bcd,fc->bfd',h.float(),mixing)
        score=(regional*self.score.weight.float()[None]).sum(-1)+self.score.bias.float()[None]
        return score[:,:,None]-threshold[None]

def install(t):
    t.DATA_KEYS.update(KEYS)
    old_cohort,old_build,old_loss,old_group,old_receipt=t.load_cohort,t.build_model,t.weighted_loss,t.group_supervision,t.source_receipt
    state={}
    def cohort(cfg):
        result=old_cohort(cfg)
        if 'native_labels' not in cfg: return result
        a=np.load(cfg['native_labels'])
        state.update(cfg=cfg,targets=dict(zip(a['ids'].tolist(),a['targets'])),
            masks=dict(zip(a['ids'].tolist(),a[cfg['native_mask']])),all_ids=result[0],train_ids=result[1],labels=result[3])
        assert set(state['targets'])<=set(t._oai_state['cell_weights'])
        # The mature smoke-only interleave truncates the OAI pool to the RSNA
        # count. Restrict smoke warmup to that declared training cohort; the
        # production check remains strict against the complete training pool.
        if '--smoke' in sys.argv:
            allowed=set(result[1])
            state['targets']={u:v for u,v in state['targets'].items() if u in allowed}
            state['masks']={u:v for u,v in state['masks'].items() if u in allowed}
        assert set(state['targets'])<=set(result[1])
        assert not set(state['targets'])&set(result[2])
        return result
    def tensors(ids,device):
        shape=next(iter(state['targets'].values())).shape
        target=torch.as_tensor(np.stack([state['targets'].get(u,np.zeros(shape,np.float32)) for u in ids]),device=device)
        mask=torch.as_tensor(np.stack([state['masks'].get(u,np.zeros(shape,bool)) for u in ids]),device=device)
        return target,mask
    def build(cfg,pretrained=False):
        model=old_build(cfg,pretrained)
        if 'native_labels' not in cfg: return model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            model.squeeze_aux=NativeOrdinal(model.feature_dim,int(np.load(cfg['native_labels'])['targets'].shape[1]))
        if cfg.get('native_couple'):
            fields=[str(f) for f in np.load(cfg['native_labels'])['fields']]
            pairs=[(t.LABEL_COLS.index(c),[fields.index(r+'_F') for r in regs if r+'_F' in fields]) for c,regs in COUPLE.items()]
            state['couple']=[(c,f) for c,f in pairs if f]
            assert state['couple'],'native_couple needs F fields of at least one compartment'
            model.squeeze_couple=nn.Parameter(torch.zeros(len(state['couple'])))
        state['model']=model
        return model
    from kneemri.coatnet_model import CoAtNetAblationModel
    old_pool=CoAtNetAblationModel._pool
    def pool(self,*args,**kwargs):
        h=old_pool(self,*args,**kwargs)
        if isinstance(getattr(self,'squeeze_aux',None),NativeOrdinal) and (self.training or getattr(self,'_native_warming',False)
                                                                           or getattr(self,'squeeze_couple',None) is not None):
            self._native_h=h
            self._native_logits=self._squeeze_private_logits if self.training else self.squeeze_aux(h)
        return h
    # The parent squeeze hook also calls squeeze_aux in training. It has no RNG or
    # side effect; this wrapper reuses its result and exposes features for calibration.
    CoAtNetAblationModel._pool=pool
    old_finalize=CoAtNetAblationModel._finalize_logits
    def finalize(self,logits,valid):
        out=old_finalize(self,logits,valid)
        a=getattr(self,'squeeze_couple',None)
        if a is None: return out
        e=torch.sigmoid(self._native_logits.detach().float()).sum(-1)  # [B,F] expected grade, no grad to the aux head
        cols=torch.as_tensor([c for c,_ in state['couple']],device=out.device)
        z=torch.stack([e[:,f].max(1).values for _,f in state['couple']],1)  # [B,C]
        add=torch.zeros(out.shape,device=out.device,dtype=torch.float32).index_add(1,cols,z*a.float()[None])
        return out+add.to(out.dtype)
    CoAtNetAblationModel._finalize_logits=finalize
    def group(batches,weights):
        out=old_group(batches,weights)
        if state:
            n=0
            for batch in batches:
                _,mask=tensors(batch['study_ids'],'cpu')
                mask=mask&batch['window_mask'].any(1)[:,None,None]
                n+=int(mask.sum())
            state['group_denominator']=max(n,1)
        return out
    def aux_loss(ids,logits,denominator=None):
        target,mask=tensors(ids,logits.device)
        values=F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')
        return (values*mask).sum()/(max(int(mask.sum()),1) if denominator is None else denominator)
    def loss(logits,targets,weights,pw,cfg,group_denominators=None):
        base=old_loss(logits,targets,weights,pw,cfg,group_denominators)
        if 'native_labels' not in cfg: return base
        auxiliary=state['model']._native_logits
        if not logits.requires_grad: auxiliary=auxiliary.detach()
        ids=t._oai_state['current_ids']
        den=state['group_denominator'] if group_denominators is not None else None
        return base+state['lambda']*aux_loss(ids,auxiliary,den)
    def warmup(model,cfg,device):
        if 'native_labels' not in cfg: return
        from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags
        from kneemri.coatnet_training_data import coatnet_collate
        out=Path(cfg['native_warmup_dir']);out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists() and '--resume' in sys.argv:
            # runtime_nq2: crash-resume; the checkpoint restores the auxiliary head, lambda is the frozen file value
            state['lambda']=json.loads(Path(cfg['native_lambda_path']).read_text())['lambda_value']
            return
        assert not (out/'complete.json').exists(),'refuse repeat native warmup'
        original={n:p.requires_grad for n,p in model.named_parameters()}
        model.requires_grad_(False);model.squeeze_aux.requires_grad_(True);model.eval();model._native_warming=True
        ids=[u for u in state['train_ids'] if u in state['masks'] and state['masks'][u].any()]
        smoke='--smoke' in sys.argv
        if smoke: ids=ids[:8]
        dc=dict(cfg,rsna_per_epoch=0,oai_per_epoch='all',oai_until_epoch=None)  # runtime_nq2: warmup is OAI-only
        ds=CoAtNetOriginalNeighborBags(dc,state['all_ids'],ids,state['labels'],train=True,uint8_output=True)
        gen=torch.Generator(); loader=t.make_loader(ds,dc,device,train=True,generator=gen)
        opt=torch.optim.AdamW(model.squeeze_aux.parameters(),lr=cfg['native_warmup_lr'],weight_decay=.01)
        started=time.time(); history=[]
        passes=1 if smoke else cfg['native_warmup_epochs']
        for epoch in range(passes):
            t.seed_all(2026+epoch); gen.manual_seed(2026+epoch)
            total=0.;seen=0
            for step,batch in enumerate(loader):
                x=t.device_images(batch,device);metadata=t.model_inputs(batch,device)
                opt.zero_grad(set_to_none=True)
                with t.amp_context(device,torch.bfloat16):
                    model(x,**metadata)
                    L=aux_loss(batch['study_ids'],model._native_logits)
                L.backward(); torch.nn.utils.clip_grad_norm_(model.squeeze_aux.parameters(),1.)
                opt.step();total+=float(L.detach())*len(batch['study_ids']);seen+=len(batch['study_ids'])
                if step%100==0:print(json.dumps(dict(event='native_warmup',epoch=epoch+1,step=step,steps=len(loader),loss=total/seen,elapsed_s=time.time()-started)),flush=True)
            history.append(dict(epoch=epoch+1,studies=seen,loss=total/seen))
        assert all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith('squeeze_aux.'))
        torch.save(model.squeeze_aux.state_dict(),out/'aux_warmup.pt')
        for n,p in model.named_parameters():p.requires_grad_(original[n])
        # Common lambda fixed on a preselected training batch, no validation inputs.
        lp=Path(cfg['native_lambda_path'])
        if cfg['native_mask']=='strict_mask' and cfg.get('squeeze_mode')!='N3_AF_QUERY':
            model.train();model.zero_grad(set_to_none=True)
            cal_ids=ids[:2]
            cal=CoAtNetOriginalNeighborBags(cfg,state['all_ids'],cal_ids,state['labels'],train=True,uint8_output=True)
            t.seed_all(2026)
            batch=coatnet_collate([cal[i] for i in range(len(cal_ids))])
            main_weights=t.supervision_weights(cfg,state['train_ids'],{})
            den=old_group([batch],main_weights).to(device)
            x=t.device_images(batch,device);meta=t.model_inputs(batch,device)
            with t.amp_context(device,torch.bfloat16):
                z=model(x,**meta)
                Lmain=old_loss(z,batch['targets'].to(device),batch['label_weights'].to(device),torch.tensor(t._oai_state['pw'],device=device),cfg,den)
                Laux=aux_loss(cal_ids,model._native_logits)
            gm=torch.autograd.grad(Lmain,model._native_h,retain_graph=True)[0].float().norm()
            ga=torch.autograd.grad(Laux,model._native_h)[0].float().norm()
            assert torch.isfinite(gm) and torch.isfinite(ga) and gm>0 and ga>0
            lam=.1*float(gm/ga)
            calibration=dict(lambda_value=lam,target_gradient_ratio=.1,main_feature_gradient=float(gm),aux_feature_gradient=float(ga),
                             ids_sha256=hashlib.sha256('\n'.join(cal_ids).encode()).hexdigest(),calibration_studies=len(cal_ids),scope='training native knees only',seed=2026)
            assert not lp.exists(),'do not silently recalibrate lambda'
            lp.write_text(json.dumps(calibration,indent=2)+'\n')
        else:
            assert lp.exists(),'N2/N3 require frozen N1 training-only calibration before joint training'
            calibration=json.loads(lp.read_text())
        state['lambda']=calibration['lambda_value']
        model.zero_grad(set_to_none=True);model._native_warming=False
        for name in ['_native_h','_native_logits','_squeeze_private_logits']:
            if hasattr(model,name):delattr(model,name)
        summary=dict(status='PASS',smoke=smoke,warmup_epochs=passes,history=history,lambda_value=state['lambda'],
                     frozen_main_parameters=True,main_optimizer_updates=0,elapsed_s=time.time()-started,
                     joint_rng='parent trainer seed_all(2026+epoch) and loader_generator reset unchanged')
        (out/'complete.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    def receipt(out,cfg,cohort,synthetic=False):
        r=old_receipt(out,cfg,cohort,synthetic)
        if 'native_labels' in cfg:
            source=Path(__file__);dst=out/'source/src/kneemri/native_aux.py';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
            r['sources']['src/kneemri/native_aux.py']=t.hash_file(source)
            r['native_aux']={'labels_sha256':t.hash_file(cfg['native_labels']),'mask':cfg['native_mask'],'ordered_thresholds':True,
                'couple':bool(cfg.get('native_couple')),
                'feature':'region/field-specific learned softmax mixture of twelve class-specific pooled image features; uniform initialization, never final logits','denominator':'effective-batch valid ordinal cells',
                'warmup':'two passes, main frozen, original joint RNG restored','lambda_file':cfg['native_lambda_path']}
        return r
    t.load_cohort=cohort;t.build_model=build;t.weighted_loss=loss;t.group_supervision=group
    t.squeeze_native_warmup=warmup;t.source_receipt=receipt
