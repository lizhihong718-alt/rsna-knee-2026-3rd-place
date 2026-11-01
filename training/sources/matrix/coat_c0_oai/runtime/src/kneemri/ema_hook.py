"""Per-update weight EMA for the window-parallel CoAtNet C0 trainer, without editing its training body.

final_matrix_20261020 (Gu handoff). The C0 trainer (sources/coat_c0, 10-17 handoff) predates the WeightEma
port, so this module adds the same EMA from outside; ``WeightEma`` is copied verbatim from n014/runtime's
train_coatnet_ablation.py (the 014 port of oai_weak3's EMA):
  * the shadow is built from the trained module itself (captured in astra_ddp_support.shard_encoder, i.e.
    after the SyncBatchNorm conversion, the exact object DDP updates) when the scheduler is created;
  * it is updated after every completed optimizer update: the trainer calls scheduler.step() exactly then;
  * ``last.pt`` is followed by ``last_ema.pt`` (EMA state, for --resume) and ``final.pt`` by
    ``final_<tag>.pt`` (same dict, model = EMA weights); save_atomic is rank-0-only under DDP.
``tail_checkpoint_from: E`` (SWA inputs): this trainer refuses ``checkpoint_every`` with full_refit, so each
``last.pt`` of epoch >= E is followed by ``epXXX.pt`` = the same raw model dict without optimizer state.
With ``ema_decays: []`` and ``tail_checkpoint_from: null`` (defaults) training and outputs are unchanged.
Gold58 of the EMA weights: evaluate final_<tag>.pt afterwards with the trainer's --evaluate-checkpoint.
"""
import json
import sys
from pathlib import Path

import torch


class WeightEma:
    """Exponential moving averages of the model state, one FP32 shadow per decay.

    Updated after every completed optimizer update n (1-based) with the warm-up decay
    min(decay, (1 + n) / (10 + n)); non-floating buffers are copied. The shadows live
    on the training device and are only read for evaluation and checkpoints.
    """

    def __init__(self, model, decays):
        state = model.state_dict()
        self.decays = {self.tag(d): d for d in decays}
        self.floating = [k for k, v in state.items() if v.is_floating_point()]
        self.dtypes = {k: v.dtype for k, v in state.items()}
        self.shadows = {tag: {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
                              for k, v in state.items()} for tag in self.decays}
        self.updates = 0

    @staticmethod
    def tag(decay):
        return 'ema' + f'{decay:g}'.replace('.', 'p')

    @torch.no_grad()
    def update(self, model, updates):
        state = model.state_dict()
        source = [state[k].float() for k in self.floating]
        for tag, decay in self.decays.items():
            shadow = self.shadows[tag]
            torch._foreach_lerp_([shadow[k] for k in self.floating], source,
                                 1. - min(decay, (1. + updates) / (10. + updates)))
            for k, v in state.items():
                if not v.is_floating_point():
                    shadow[k].copy_(v)
        self.updates = updates

    def cpu_state(self, tag):
        """Same dtypes and contiguous layout as cpu_state(model)."""
        return {k: v.detach().to('cpu', dtype=self.dtypes[k], memory_format=torch.contiguous_format, copy=True)
                for k, v in self.shadows[tag].items()}

    def load_into(self, net, tag):
        net.load_state_dict(self.shadows[tag], strict=True)

    def describe(self, tag):
        return dict(decay=self.decays[tag], optimizer_updates=self.updates,
                    warmup='min(decay, (1 + n) / (10 + n))', update='after every completed optimizer update')

    def state_dict(self):
        return dict(updates=self.updates, shadows={tag: self.cpu_state(tag) for tag in self.shadows})

    def load_state_dict(self, state):
        if set(state['shadows']) != set(self.shadows):
            raise ValueError('resume EMA decays differ')
        for tag, shadow in self.shadows.items():
            for k, v in shadow.items():
                v.copy_(state['shadows'][tag][k])
        self.updates = state['updates']


class _EmaScheduler:
    """Delegates everything to the trainer's scheduler; step() also advances the EMA."""

    def __init__(self, scheduler, ema, module):
        self._scheduler, self._ema, self._module = scheduler, ema, module

    def step(self, *args, **kwargs):
        result = self._scheduler.step(*args, **kwargs)
        self._ema.update(self._module, self._ema.updates + 1)
        return result

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


def install(trainer):
    import astra_ddp_support as ddp
    trainer.DEFAULTS.setdefault('ema_decays', [])
    trainer.DEFAULTS.setdefault('tail_checkpoint_from', None)
    state = {}
    argv = sys.argv
    if '--resume' in argv:
        state['resume'] = Path(argv[argv.index('--resume') + 1]).with_name('last_ema.pt')

    original_shard = ddp.shard_encoder

    def shard_encoder(module):
        state['module'] = module
        return original_shard(module)

    ddp.shard_encoder = shard_encoder
    original_normalize = trainer.normalize_config

    def normalize_config(config):
        cfg = original_normalize(config)
        decays = cfg['ema_decays']
        if (not isinstance(decays, list) or len(set(decays)) != len(decays)
                or any(isinstance(d, bool) or not isinstance(d, float) or not 0 < d < 1 for d in decays)):
            raise ValueError('ema_decays must be a list of distinct floats in (0, 1)')
        tail = cfg['tail_checkpoint_from']
        if tail is not None and (isinstance(tail, bool) or not isinstance(tail, int) or not 1 <= tail <= cfg['epochs']):
            raise ValueError('tail_checkpoint_from must be null or an epoch in [1, epochs]')
        state['tail'] = tail
        return cfg

    trainer.normalize_config = normalize_config
    original_scheduler = trainer.create_scheduler

    def create_scheduler(optimizer, cfg, updates_per_epoch):
        scheduler = original_scheduler(optimizer, cfg, updates_per_epoch)
        if not cfg['ema_decays']:
            return scheduler
        if 'module' not in state:
            raise RuntimeError('EMA needs the window-parallel module (run under torch.distributed.run)')
        ema = WeightEma(state['module'], cfg['ema_decays'])
        if state.get('resume') is not None:
            if not state['resume'].exists():
                raise FileNotFoundError(f'{state["resume"]} missing: cannot resume the EMA shadow')
            ema.load_state_dict(torch.load(state['resume'], map_location='cpu', weights_only=True))
        state['ema'] = ema
        return _EmaScheduler(scheduler, ema, state['module'])

    trainer.create_scheduler = create_scheduler
    original_save = trainer.save_atomic

    def save_atomic(value, path):
        original_save(value, path)
        ema, path = state.get('ema'), Path(path)
        tail = state.get('tail')
        if path.name == 'last.pt' and tail is not None and value['epoch'] + 1 >= tail:
            raw = {k: v for k, v in value.items() if k not in
                   ('optimizer', 'scheduler', 'scaler', 'best_selection_auc', 'history', 'optimizer_updates')}
            original_save(raw, path.with_name(f'ep{value["epoch"] + 1:03d}.pt'))
        if ema is None:
            return
        if path.name == 'last.pt':
            original_save(ema.state_dict(), path.with_name('last_ema.pt'))
        if path.name == 'final.pt':
            for tag in ema.decays:
                original_save(dict(value, model=ema.cpu_state(tag), weight_ema=ema.describe(tag)),
                              path.with_name(f'final_{tag}.pt'))
            path.with_name('weight_ema.json').write_text(json.dumps(dict(
                decays=list(ema.decays.values()), optimizer_updates=ema.updates,
                warmup='min(decay, (1 + n) / (10 + n))', update='after every completed optimizer update',
                precision='fp32', checkpoints='final_<tag>.pt; final.pt stays the raw model',
                module='astra_ddp_support.shard_encoder input (SyncBatchNorm-converted)'), indent=2))

    trainer.save_atomic = save_atomic
    original_receipt = trainer.source_receipt

    def source_receipt(output, cfg, cohort, synthetic=False):
        result = original_receipt(output, cfg, cohort, synthetic)
        root = Path(__file__).resolve().parents[3]
        for rel in ('runtime/src/kneemri/ema_hook.py', 'portable_refit_oai.py', 'soft_objective.py', 'train_soft.py'):
            target = Path(output) / 'source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / rel).read_bytes())
            result['sources'][rel] = trainer.hash_file(root / rel)
        result['weight_ema_hook'] = dict(ema_decays=cfg['ema_decays'], tail_checkpoint_from=cfg['tail_checkpoint_from'])
        return result

    trainer.source_receipt = source_receipt
