#!/usr/bin/env python3
"""One scoring worker of the final 36-member ensemble: one training-runtime family, some of its members, one GPU.

R2X57 (2026-10-21): family A gains the bit_depth_up factory (final-matrix BiT@224, model_input_size 320: the
images entering DepthInferenceModel.prefix are bilinearly upsampled, as the bit224 trainer does at the start of
forward_features), and family J (Gu coat_oai runtime: CoAtNet native384/512, train_coatnet_n384.build_model).

Best29W3 (2026-10-20, copied from final_ensemble36r_20261019): a family-A worker whose members include a coupled
oai_weak3 model (config key deploy_native_couple) installs native_couple_deploy (shipped next to this file; the
eval-mode hooks of the RK4C single-model bundle, no-ops for models without the coupling) before building members.

The members of a family share Python modules that differ from other families (train_coatnet_ablation, kneemri.*),
so every family runs in its own process with its own runtime first on sys.path. Inside the process each member is
built exactly as its own single-model entry builds it (same factory, same config registration, strict load of the
fixed-final primary weight) and scored by that runtime's audited ExactUint8Inference engine under the same
inference_mode / amp_context / FP16-on-T4 rules.

Inputs come from the shared per-study cache written once by ensemble_main.py (one DICOM decode per study). For a
member, the payload is rebuilt exactly as its entry's prepare would build it: the window source (legacy original384
or native R), canonical laterality flags (resnet_canonical.apply_flags, copied below), lateral positions
(where(multiplier != 0, positions * multiplier, positions)), then coatnet_fast_runtime.compact_windows.

Protocol: JSON lines. stdin {"op": "run", "batch": b, "studies": [[uid, cache_dir or null], ...]} | {"op": "exit"};
the protocol stdout is a dup of fd 1 taken before any import, and fd 1 then points at stderr so library prints
cannot corrupt it. Replies: ready / done / result / error.

Fault tolerance (Final36R, after 57406556 died on the hidden test with an unhandled error): a batch never raises.
Every "done" reply carries the batch's float32 logits ({member: {uid: [12]}}), so the orchestrator keeps them even if
this process dies later. A member whose batched call fails (CUDA OOM included) gets its cached blocks returned and
the studies retried one at a time, then with a quarter-size encode chunk; a study that still fails or gives
nonfinite logits is reported in "failures" and left out of that member only. If every member failed every study
of a batch (a broken CUDA context), the process exits so the orchestrator restarts it.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

PROTOCOL = os.fdopen(os.dup(1), 'w', buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ensemble_efficiency as efficiency
INPUT_CACHE = efficiency.BatchCache()

OAI_KEYS = {'oai_index', 'oai_labels_csv', 'oai_per_epoch', 'rsna_per_epoch', 'oai_loss_weight', 'oai_target_transform',
            'init_checkpoint', 'oai_weights_csv', 'oai_fixed_denominator', 'oai_logit_adjust', 'oai_sampling',
            'oai_importance_weight', 'oai_until_epoch'}


def send(**message):
    PROTOCOL.write(json.dumps(message) + '\n')
    PROTOCOL.flush()


def apply_flags(windows, flip, swap):
    """resnet_canonical.apply_flags (identical in every runtime): mirror W, reverse the slice channels."""
    windows = np.array(windows, copy=True)
    windows[flip] = windows[flip][..., ::-1]
    windows[swap] = windows[swap][:, ::-1]
    return windows


def load_state(path, fmt):
    if fmt == 'pt':
        return torch.load(path, map_location='cpu', weights_only=True)['model']
    try:
        from safetensors.torch import load_file
        return load_file(str(path), device='cpu')
    except ImportError:  # plain safetensors reader: 8-byte header size, JSON header, raw little-endian tensors
        dtypes = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16, 'I64': torch.int64,
                  'I32': torch.int32, 'U8': torch.uint8, 'BOOL': torch.bool, 'F64': torch.float64}
        raw = Path(path).read_bytes()
        size = int.from_bytes(raw[:8], 'little')
        header = json.loads(raw[8:8 + size])
        base = 8 + size
        state = {}
        for name, item in header.items():
            if name == '__metadata__':
                continue
            start, stop = item['data_offsets']
            buffer = bytearray(raw[base + start:base + stop])
            tensor = torch.frombuffer(buffer, dtype=dtypes[item['dtype']]) if stop > start else \
                torch.empty(0, dtype=dtypes[item['dtype']])
            state[name] = tensor.reshape(item['shape']).clone()
        return state


class Family:
    """Import one runtime's modules and expose its member factories, mirroring that family's inference entry."""

    def __init__(self, family, runtime_dir):
        runtime_dir = Path(runtime_dir).resolve()
        sys.path[:0] = [str(runtime_dir / 'src'), str(runtime_dir / 'scripts')]
        self.family = family
        if family == 'K':
            # pfoa_full_c0 bundle run.py: soft-supervision keys, then the original-neighbour entry.
            import train_coatnet_ablation as trainer
            trainer.DATA_KEYS |= {'soft_supervision', 'soft_manifest', 'mass_normalizer', 'mass_strength'}
        import inference_coatnet_original_dicom as runtime
        import train_coatnet_ablation as trainer
        from coatnet_fast_runtime import compact_windows
        self.runtime, self.trainer, self.compact_windows = runtime, trainer, compact_windows
        self.factories = {}
        if family in ('A', 'D'):
            import resnet_depth_inference as rdi  # CONFIG_KEYS | DEPTH_KEYS
            self.rdi = rdi
            self.factories['bit_depth'] = self._bit_depth
        if family == 'A':
            from kneemri import native_aug
            trainer.DATA_KEYS.update(native_aug.DEGRADE_KEYS)  # bundle ENTRY of package_weak3 / package_ema
            trainer.DATA_KEYS.update(OAI_KEYS)
            import swin_inference  # noqa: F401  (registers the Swin / E2E / warp keys exactly as its entry)
            from kneemri import swin
            self.factories['swin'] = swin.build_model
        if family == 'C':
            import inference_native384_dicom  # noqa: F401  (registers ConvNeXt/DINO keys)
            from kneemri import depth384
            trainer.MODEL_KEYS.update(depth384.DEPTH_MODEL_KEYS)
            trainer.DATA_KEYS.update(depth384.DEPTH_DATA_KEYS)
            self.factories['cnx_depth384'] = depth384.build_model
        if family == 'N':
            import native224_inference  # noqa: F401
            from kneemri import native224
            self.factories['native224'] = native224.build_model
        if family == 'M':
            import inference_new_backbone_dicom  # noqa: F401  (new-backbone + fullstrength key registration)
            from kneemri import mambaout_depth_e2e  # train_mambaout_depth_e2e.install(): keys and factory
            trainer.MODEL_KEYS.update(mambaout_depth_e2e.E2E_MODEL_KEYS)
            trainer.DATA_KEYS.update(mambaout_depth_e2e.E2E_DATA_KEYS)
            self.factories['mamba_depth'] = mambaout_depth_e2e.build_model
        if family == 'K':
            self.factories['coatnet'] = runtime.build_model
        if family == 'A':
            # final-matrix BiT@224 (bit224_oai runtime_up): bilinear upsample to model_input_size inside forward_features
            trainer.MODEL_KEYS.add('model_input_size')
            self.factories['bit_depth_up'] = self._bit_depth_up
        if family == 'J':
            # final-matrix CoAtNet native384/512 (coat_oai): the key registration and factory of train_coatnet_n384.main
            from kneemri import native_aug
            from kneemri import resnet_depth_e2e as e2e
            from kneemri import coat_e2e
            trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS | native_aug.DEGRADE_KEYS | OAI_KEYS)
            trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS | coat_e2e.COAT_E2E_MODEL_KEYS | {'drop_path_rate'})
            import train_coatnet_n384
            self.factories['coat_e2e'] = train_coatnet_n384.build_model

    def _bit_depth_up(self, cfg, pretrained=False):
        # bit224_oai train_resnet_n384.build_model upsamples bilinearly to model_input_size at the start of the
        # backbone's forward_features. The streaming DepthInferenceModel bypasses forward_features (prefix runs
        # backbone.stem, then the stages), so the same interpolation is applied to the images entering prefix.
        import torch.nn.functional as F
        size = cfg.get('model_input_size')
        if isinstance(size, bool) or not isinstance(size, int) or size % 32 or size <= int(cfg['resolution']):
            raise ValueError('model_input_size must be a multiple of 32 above resolution')
        model = self._bit_depth(cfg, pretrained)
        prefix = model.prefix

        def upsampled_prefix(images):
            if images.shape[-1] != size or images.shape[-2] != size:
                images = F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False)
            return prefix(images)
        model.prefix = upsampled_prefix
        return model

    def _bit_depth(self, cfg, pretrained=False):
        # package_depth.py / package_weak3.py: the fixed-final E2E (or plain) base wrapped for streaming inference.
        stage = dict(use_depth=False, depth=None, trainable_stages=[3], stage1_e2e=True,
                     canonical_positions=bool(cfg.get('depth_canonical_positions')), stage1_run='deployed',
                     stage1_epoch=25)
        return self.rdi.build_depth_model(dict(cfg, depth_stage=stage))

    def config(self, raw):
        trainer = self.trainer
        known = set(trainer.DEFAULTS) | trainer.MODEL_KEYS | trainer.DATA_KEYS
        extra = sorted(set(raw) - known)
        trainer.DATA_KEYS.update(extra)  # training-only fields of the source runtime (reported in "ready")
        return trainer.normalize_config(dict(raw)), extra

    def build(self, member, state, device):
        cfg, extra = self.config(member['config'])
        model = self.factories[member['factory']](cfg, pretrained=False).eval()
        target = model.base if member['factory'] in ('bit_depth', 'bit_depth_up') else model
        target.load_state_dict(state, strict=True)
        if device.type == 'cuda' and cfg['eval_amp'] == 'bf16' and torch.cuda.get_device_capability(device)[0] < 8:
            cfg['eval_amp'] = 'fp16'  # same rule as the single-model runtime on T4
        if os.environ.get('ENSEMBLE_EVAL_AMP'):
            cfg['eval_amp'] = os.environ['ENSEMBLE_EVAL_AMP']
        model = model.to(device).eval()
        dtype = self.trainer.select_eval_amp(cfg, device)
        return cfg, model, dtype, extra


class Member:
    def __init__(self, spec, family, device, chunk):
        self.spec, self.id = spec, spec['id']
        state = load_state(spec['weight_path'], spec['weight_format'])
        for prefix in spec['drop_state_prefixes']:
            state = {k: v for k, v in state.items() if not k.startswith(prefix)}
        self.cfg, self.model, self.dtype, self.extra_keys = family.build(spec, state, device)
        del state
        if int(self.cfg['resolution']) != spec['resolution'] or self.cfg['arch'] != spec['arch']:
            raise ValueError(f'{self.id}: config contract changed')
        self.engine = family.runtime.ExactUint8Inference(self.model, self.cfg, device, chunk)
        self.batched = hasattr(self.engine, 'logits_compact_batch')
        self.frozen_ws_count = efficiency.freeze_standardized_convolutions(self.model) if os.environ.get('ENSEMBLE_FREEZE_WS', '1') == '1' else 0
        self.variant = spec['variant']
        self.key = (self.variant['source'], self.variant['canonical'], self.variant['lateral'])


def assemble(cache_dir, key, compact_windows):
    source, canonical, lateral = key
    cache_dir = Path(cache_dir)
    with open(cache_dir / 'meta.pkl', 'rb') as handle:
        meta = pickle.load(handle)
    windows = np.load(cache_dir / f'{source}.npy')
    if canonical:
        windows = apply_flags(windows, meta['flip'], meta['swap'])
    positions = meta['positions']
    if lateral:
        multiplier = meta['multiplier']
        positions = np.where(multiplier != 0, positions * multiplier, positions).astype(np.float32)
    arrays = (meta['head'], windows, meta['mask'], meta['series'], meta['groups'], meta['planes'],
              positions, meta['reliable'], meta['details'])
    return compact_windows(arrays)


def release(device):
    INPUT_CACHE.clear()  # free shared tensors before yielding the GPU lock
    if device.type == 'cuda' and os.environ.get('ENSEMBLE_EMPTY_CACHE', '1') == '1':
        # Several family processes share one T4: return this batch's activation blocks before the
        # next process takes the GPU lock, so only one activation peak is resident at a time.
        torch.cuda.empty_cache()


def engine_logits(member, family, device, batch, profile):
    with torch.inference_mode(), family.trainer.amp_context(device, member.dtype):
        if member.batched and os.environ.get('ENSEMBLE_SHARE_INPUTS', '1') == '1':
            values = INPUT_CACHE.logits(member, batch)
        elif member.batched:
            values = member.engine.logits_compact_batch(batch)
        else:
            values = torch.cat([member.engine.logits_compact(p) for p in batch], 0)
    if profile and device.type == 'cuda':
        torch.cuda.synchronize(device)
    return values.float().cpu().numpy()


def describe(error):
    return f'{type(error).__name__}: {str(error)[:300]}'


def score_member(member, family, device, uids, batch, profile):
    """{uid: float32 logits} and {uid: error} of one member; the first call is the unchanged batched call."""
    logits, failures = {}, {}
    try:
        values = engine_logits(member, family, device, batch, profile)
    except Exception as error:  # noqa: BLE001  (CUDA OOM included): retry study by study
        first = describe(error)
        release(device)
    else:
        for row, uid in enumerate(uids):
            if np.isfinite(values[row]).all():
                logits[uid] = values[row]
            else:
                failures[uid] = 'nonfinite logits'
        return logits, failures
    full = member.engine.chunk_size
    for uid, payload in zip(uids, batch):
        failures[uid] = first
        for chunk in dict.fromkeys((full, max(1, full // 4))):
            member.engine.chunk_size = chunk
            try:
                value = engine_logits(member, family, device, [payload], profile)[0]
            except Exception as error:  # noqa: BLE001
                failures[uid] = f'{first} | single chunk {chunk}: {describe(error)}'
                release(device)
                continue
            finally:
                member.engine.chunk_size = full
            if np.isfinite(value).all():
                logits[uid] = value
                del failures[uid]
            else:
                failures[uid] = f'{first} | single chunk {chunk}: nonfinite logits'
            break
    return logits, failures


def inject_fault(name, batch, out):
    """Local checks only: ENSEMBLE_FAULT=kind:worker:batch[:once],... with kind exit (process dies) or hang."""
    for spec in filter(None, os.environ.get('ENSEMBLE_FAULT', '').split(',')):
        kind, worker, at, *once = spec.split(':')
        if worker != name or int(at) != batch:
            continue
        marker = out.with_name(f'{name}.fault{batch}')
        if once and marker.exists():
            continue
        marker.touch()
        if kind == 'exit':
            os._exit(9)
        time.sleep(1e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--members', required=True, help='JSON file with this worker\'s member specs')
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--lock', default='')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(int(os.environ.get('ENSEMBLE_TORCH_THREADS', '1')))  # >1 only for CPU checks
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        family = Family(args.family, args.runtime)
        specs = json.loads(Path(args.members).read_text())
        if args.family == 'A' and any(spec['config'].get('deploy_native_couple') for spec in specs):
            import native_couple_deploy
            native_couple_deploy.install(family.rdi)
        members = [Member(spec, family, device, args.chunk) for spec in specs]
    except Exception:
        send(op='error', name=args.name, error=traceback.format_exc())
        raise
    send(op='ready', name=args.name, seconds=time.monotonic() - started,
         members=[dict(id=m.id, eval_amp=m.cfg['eval_amp'], extra_config_keys=m.extra_keys, batched=m.batched,
                       variant=m.variant, frozen_ws_count=m.frozen_ws_count) for m in members],
         device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
         memory_allocated=torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0)
    lock = open(args.lock, 'a+') if args.lock else None
    logits = {m.id: {} for m in members}
    member_seconds = {m.id: 0.0 for m in members}
    profile = os.environ.get('ENSEMBLE_PROFILE') == '1'
    for line in sys.stdin:
        message = json.loads(line)
        if message['op'] == 'exit':
            break
        begin = time.monotonic()
        live = [(uid, cache) for uid, cache in message['studies'] if cache]
        # R2X57 rounds: a batch names the members this round scores (default: every member of the worker)
        wanted = set(message.get('members') or [m.id for m in members])
        active = [m for m in members if m.id in wanted]
        batch_logits = {m.id: {} for m in active}
        failures = {m.id: {} for m in active}
        payloads = {}
        for member in active:
            if member.key in payloads:
                continue
            payloads[member.key] = ([], [])
            for uid, cache in live:
                try:
                    payloads[member.key][1].append(assemble(cache, member.key, family.compact_windows))
                    payloads[member.key][0].append(uid)
                except Exception as error:  # noqa: BLE001
                    for other in active:
                        if other.key == member.key:
                            failures[other.id][uid] = 'assemble ' + describe(error)
        prepared = time.monotonic()
        batch_seconds = {}
        if lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
        send(op='busy', name=args.name, batch=message['batch'])
        inject_fault(args.name, message['batch'], Path(args.out))
        try:
            for member in active:
                tick = time.monotonic()
                uids, batch = payloads[member.key]
                if uids:
                    if os.environ.get('ENSEMBLE_FAULT_MEMBER') == member.id:  # fault-injection check
                        batch = [dict(p, windows=p['windows'][:, :1]) for p in batch]
                    values, failed = score_member(member, family, device, uids, batch, profile)
                    batch_logits[member.id].update(values)
                    failures[member.id].update(failed)
                member_seconds[member.id] += time.monotonic() - tick
                batch_seconds[member.id] = time.monotonic() - tick
        finally:
            release(device)
            if lock:
                fcntl.flock(lock, fcntl.LOCK_UN)
        for member in active:
            logits[member.id].update(batch_logits[member.id])
        failures = {k: v for k, v in failures.items() if v}
        if live and failures and all(len(failures.get(m.id, ())) == len(live) for m in active):
            send(op='error', name=args.name, batch=message['batch'], error='every member failed every study',
                 failures=failures)
            sys.exit(3)  # a broken CUDA context: the orchestrator restarts this worker
        send(op='done', name=args.name, batch=message['batch'], seconds=time.monotonic() - begin,
             assemble_seconds=prepared - begin, failures=failures, member_seconds=batch_seconds,
             logits={k: {uid: v.tolist() for uid, v in values.items()} for k, values in batch_logits.items()})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for m in members:
        arrays[m.id + '__uids'] = np.array(list(logits[m.id]), dtype=str)
        arrays[m.id + '__logits'] = np.array(list(logits[m.id].values()), dtype=np.float32).reshape(-1, 12)
    np.savez(out, **arrays)
    send(op='result', name=args.name, path=str(out), member_seconds=member_seconds,
         seconds=time.monotonic() - started,
         peak_memory=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)


if __name__ == '__main__':
    main()
