"""Explicit relocation adapter; original fold checkpoint is provenance, never initialization.

final_matrix_20261020 (Gu handoff): identical to the 10-17 tools/portable_refit.py except that the
Full4342/Gold58 cohort check and the mass_normalizer re-derivation use the competition rows only
(kneemri.oai_pseudo appends the OAI knees, ids starting with "OAI.", to trained_study_uids).
"""
import ast, copy, hashlib, inspect, json
from pathlib import Path

def install(trainer, expected, recipe):
 import numpy as np
 original_receipt=trainer.source_receipt
 def validate(cfg,cohort):
  # Do not pretend that the historical seed-2026 fold selection was rerun.
  for k,v in expected.items():
   if cfg.get(k)!=v:raise ValueError('Frozen requested recipe changed: '+k)
  if cfg['epochs']!=25 or cfg['scheduler_total_epochs']!=50:raise ValueError('Fixed endpoint/horizon changed')
  train=[u for u in cohort['trained_study_uids'] if not u.startswith('OAI.')];gold=cohort['gold_study_uids'];peers=cohort['excluded_report_group_peers']
  if len(train)!=4342 or len(gold)!=58 or len(peers)!=4:raise ValueError('Full4342/Gold58/peer cohort mismatch')
  if set(cohort['trained_study_uids'])&(set(gold)|set(peers)):raise ValueError('Protected cohort overlap')
  y=trainer.read_labels(cfg['labels_csv']).loc[train,trainer.LABEL_COLS].to_numpy(np.float32)
  pos=trainer.positive_weights(y,np.ones_like(y),cfg)
  norm=1/(1+(pos-1)*y).astype(np.float64).mean(0)
  np.testing.assert_allclose(cfg['mass_normalizer'],norm,rtol=0,atol=1e-12)
  return dict(parent_checkpoint=cfg['refit_parent_checkpoint'],parent_checkpoint_sha256='historical-provenance-only-see-parent_provenance.json',parent_selected_epoch=25,parent_selection_auc=None,parent_validation_role='historical frozen recipe; portable seed extension',parent_validation_strategy='grouped_cv',parent_val_fold=cfg['val_fold'],parent_scheduler_total_epochs=50,fresh_initialization_required=True,parent_weights_used_for_initialization=False,portable_refit=True)
 def receipt(*a,**kw):
  result=original_receipt(*a,**kw);p=Path(__file__)
  result['portable_refit_adapter']=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),recipe=recipe['id'],historical_fold_selection_repeated=False,training_body_unchanged=True)
  return result
 trainer.validate_refit_parent=validate;trainer.source_receipt=receipt
 if recipe.get('retain_prefix_epoch'):
  save=trainer.save_atomic
  def keep(value,path):
   save(value,path)
   if Path(path).name=='last.pt' and value['epoch']==16:save(value,Path(path).with_name('prefix_ep017.pt'))
  trainer.save_atomic=keep

def install_prefix(trainer,path,expected_parent_id):
 """Inject only full-state restore into the pinned original ranking training body."""
 code=inspect.getsource(trainer.main)
 marker='    from auc_objective import ReliableRanking'
 if code.count(marker)!=1:raise ValueError('Pinned training body changed')
 def restore(cfg,cohort,receipt,model,optimizer,scheduler,scaler,device,out):
  import torch
  state=torch.load(path,map_location=device,weights_only=True)
  if state['epoch']!=16 or len(state['history'])!=17:raise ValueError('Requires completed epoch17')
  pc=state['config']
  if pc['seed']!=cfg['seed'] or pc['epochs']!=25:raise ValueError('Wrong seed or prefix schedule')
  if state['receipt'].get('portable_refit_adapter',{}).get('recipe')!=expected_parent_id:raise ValueError('Prefix does not belong to the declared G07 dependency')
  if pc.get('auc_weight',0)!=0 or pc.get('rank_weight',0)!=0:raise ValueError('G07 prefix must be pure BCE')
  if state.get('best_selection_auc') is not None:raise ValueError('Prefix must use the fixed endpoint policy')
  if [x['epoch'] for x in state['history']]!=list(range(1,18)):raise ValueError('Prefix history is incomplete')
  # Explicit G07 -> G16 changes are all fixed by the published requested templates.
  ignored={'validation_role','soft_manifest','eval_budgets','eval_sweep_every','workers','checkpoint_every','monitor_gold','fixed_positive_weights'}
  changed={k for k in set(pc)|set(cfg) if pc.get(k)!=cfg.get(k)}
  unexpected={k for k in changed if k not in ignored and not k.startswith('auc_')}
  if unexpected:raise ValueError('Unexpected prefix config differences: '+str(sorted(unexpected)))
  for k in ['trained_study_uids','gold_study_uids','excluded_report_group_peers']:
   if state['cohort'][k]!=cohort[k]:raise ValueError('Prefix cohort differs: '+k)
  if state['scheduler']['last_epoch']!=state['optimizer_updates'] or state['optimizer_updates']!=9231:raise ValueError('Incomplete prefix optimizer state')
  for k in ['initial_weights_sha256','label_order','normalization','folds_sha256','corpus_manifest_sha256']:
   if state['receipt'][k]!=receipt[k]:raise ValueError('Prefix receipt differs: '+k)
  model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
  trainer.write_json(Path(out)/'branch_receipt.json',dict(prefix=str(path),sha256=trainer.hash_file(path),seed=cfg['seed'],completed_epochs=17,full_optimizer_scheduler_scaler_restored=True,allowed_changes=sorted(changed)))
  return 17,state['best_selection_auc'],state['history'],state['optimizer_updates']
 trainer._portable_restore=restore
 code=code.replace(marker,"    if args.resume or smoke or args.evaluate_checkpoint: raise ValueError('PF continuation requires a fresh branch from G07 prefix')\n    start, best, history, updates = _portable_restore(cfg, cohort, receipt, model, optimizer, scheduler, scaler, device, args.out)\n"+marker)
 exec(compile(code,str(Path(__file__)), 'exec'),trainer.__dict__)
