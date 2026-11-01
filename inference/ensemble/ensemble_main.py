#!/usr/bin/env python3
"""Best29W3 (user request 2026-10-20): Best29 for nine columns, the coupled OAI pool for lateral OA / PF OA / synovitis.

Copied from final_ensemble36r_20261019 (the fault-tolerant 36-member orchestrator). Changed only: the member table is
members.json; a member may carry "columns" (the label columns it contributes to, default all 12) and each column is
the equal-weight sigmoid mean over the members routed to it (with no "columns" anywhere this is exactly the
original mean); a cell with no routed member scoring that study falls back to the mean over every member that
scored it, then to the column median.

History: Final 36-member ensemble (user request 2026-10-19): decode every study once, score every member, mean sigmoid.

Preprocessing (CPU, forked pool, runtime A = the swin512/oai_host runtime whose DICOM files are byte-identical to every
family's): one ReadOnceStudy decode and one legacy pipeline pass per study yield
  * legacy384.npy  - build_one(0, uid, serialized, 384, ...) windows (original384 contract: CoAtNet C0, BiT E2E_D/S,
                     MambaOut),
  * native{R}.npy  - native_windows(..., R) for every native resolution any member needs (224 / 384 / 512),
  * meta.pkl       - the legacy metadata arrays, the canonical flags (rc.window_flags) and the lateral multiplier
                     (rdi.lateral_multipliers), computed once.
Every family worker (ensemble_worker.py, one process per family x GPU, models loaded once) rebuilds each member's
payload from this cache exactly as that member's own single-model entry would (window source, canonical flags,
lateral positions, compact_windows) and scores it with its runtime's ExactUint8Inference engine.

The cache is bounded (ENSEMBLE_INFLIGHT batches in flight) and deleted batch by batch once every worker scored it.
Combination: equal-weight mean of the 36 sigmoid probability vectors, all 12 columns. A study whose preprocessing fails
twice (pool, then in the parent) gets the per-column median of the other studies.

Fault tolerance (Final36R; 57406556 = this ensemble died on the hidden test with an unhandled error after the
same code had passed Gold58 and the visible test on T4): workers stream their logits with every batch; a worker
that dies, errors or makes no progress for ENSEMBLE_WATCHDOG_S is killed and restarted with the batches it still
owes (a batch that took the worker down twice is skipped for it; after ENSEMBLE_MAX_DEATHS deaths, or a failed
reload, the worker is dropped). Each study's probability is the mean over the members that scored it; when every
member scored every study this is exactly the plain 36-member mean. Past ENSEMBLE_DEADLINE_S (default 8 h) no
new batch is started and unscored studies get the median. The submission is written when at least
ENSEMBLE_MIN_COVERAGE (0.5) of the studies were scored, or ENSEMBLE_MIN_COVERAGE_ON_ERROR (0.9) after an unexpected
orchestrator error; the receipt records every degradation.

Environment: ENSEMBLE_DATA_ROOT (test.csv / test_series.csv / test_series/), ENSEMBLE_OUTPUT (submission.csv),
ENSEMBLE_MEMBER_DIR (per-member CSVs), ENSEMBLE_WEIGHT_ROOTS (os.pathsep list searched recursively for the weight
files; default /kaggle/input), ENSEMBLE_CACHE_DIR, ENSEMBLE_WORKERS (preprocessing processes, default 3),
ENSEMBLE_READER_THREADS (4), ENSEMBLE_STUDY_BATCH (2), ENSEMBLE_CHUNK (16), ENSEMBLE_INFLIGHT (6), ENSEMBLE_LANES
(worker plan JSON, default lanes.json), ENSEMBLE_GPU_LOCK (1).
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import copy
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import pickle
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

BUNDLE = Path(__file__).resolve().parents[1]
RUNTIME = BUNDLE / 'runtimes' / 'A'
sys.path[:0] = [str(RUNTIME / 'src'), str(RUNTIME / 'scripts')]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import coatnet_fast_runtime as cfr  # noqa: E402
import resnet_canonical as rc  # noqa: E402
import resnet_depth_inference as rdi  # noqa: E402
import resnet_native512_inference as rni  # noqa: E402
from build_coatnet_original_corpus import SLOTS, SPAN, build_one  # noqa: E402
from inference_coatnet_ablation import preprocessing_contract  # noqa: E402
from kneemri.config import load_config  # noqa: E402
from kneemri.pipeline import preprocess_signature  # noqa: E402

LABEL_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
              'Synovitis', "Baker's", 'Contusion', 'Fracture']
_STATE = {}


def env(name, default):
    return type(default)(os.environ.get(name, default))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def init_pool(pipeline, contract, data_root, reader_threads, native, cache_root):
    import cv2
    cv2.setNumThreads(1)
    _STATE.update(pipeline=pipeline, contract=contract, data_root=Path(data_root), reader_threads=int(reader_threads),
                  native=tuple(native), cache_root=Path(cache_root))


def nothing():
    return os.getpid()


def prepare_all(index, uid, records):
    """One decode + one legacy pass; the legacy384 windows, every needed native resolution and the flags."""
    started = time.monotonic()
    state = _STATE
    pipeline = copy.deepcopy(state['pipeline'])
    pipeline.study_root = str(state['data_root'] / 'test_series' / uid)
    reader = cfr.ReadOnceStudy(pipeline.study_root, cfr.process_reader_pool(state['reader_threads']))
    if reader.metadata.empty:
        raise ValueError(f'no readable DICOM headers for {uid}')
    result = cfr.process_study(uid, pipeline, reader.metadata, records, reader.decode)
    if result.failed:
        raise ValueError(f'preprocessing failed for {uid}: {result.error}')
    serialized = cfr.result_arrays(result, pipeline)
    contract = state['contract']
    legacy = build_one(0, uid, serialized, 384, contract['source_preprocess_signature'], contract.get('crop_fov_mm'))
    out = state['cache_root'] / f'{index:06d}'
    partial = out.with_name(out.name + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    count = len(legacy[1])
    for resolution in state['native']:
        windows = rni.native_windows(result, serialized, legacy[8], reader.metadata, reader.decode,
                                     pipeline.study_root, pipeline, resolution)
        if len(windows) != count:
            raise ValueError('native and legacy window tables differ in length')
        np.save(partial / f'native{resolution}.npy', windows)
    np.save(partial / 'legacy384.npy', legacy[1])
    flip, swap, _, _ = rc.window_flags(result, serialized, reader.metadata, count)
    multiplier = rdi.lateral_multipliers(result, serialized, reader.metadata, count)
    meta = dict(head=legacy[0], mask=legacy[2], series=legacy[3], groups=legacy[4], planes=legacy[5],
                positions=legacy[6], reliable=legacy[7], details=legacy[8], flip=flip, swap=swap,
                multiplier=multiplier, uid=uid)
    with open(partial / 'meta.pkl', 'wb') as handle:
        pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    return dict(index=index, uid=uid, cache=str(out), seconds=time.monotonic() - started,
                slices=int(len(reader.metadata)), windows=int(np.count_nonzero(legacy[2])),
                canonical_flips=int(flip.sum()), canonical_swaps=int(swap.sum()),
                lateral_windows=int((multiplier != 0).sum()))


class Worker:
    def __init__(self, spec, members, run_dir, chunk, lock_dir, events, generation=0):
        self.name, self.spec, self.generation = spec['name'], spec, generation
        self.args = (spec, members, run_dir, chunk, lock_dir, events)
        self.outstanding = []        # batches sent to this process and not yet done, in order
        self.ready = False
        self.busy_since = None       # when this process last took its GPU lock for a batch
        self.progress = time.monotonic()
        member_file = run_dir / f'{self.name}.members.json'
        member_file.write_text(json.dumps(members))
        self.out = run_dir / 'results' / f'{self.name}.npz'
        suffix = f'.{generation}' if generation else ''
        self.log = open(run_dir / f'{self.name}.log{suffix}', 'w')
        device = '' if os.environ.get('ENSEMBLE_FORCE_CPU') == '1' else str(spec['gpu'])  # CPU checks only
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
        command = [sys.executable, str(BUNDLE / 'ensemble' / 'ensemble_worker.py'), '--name', self.name,
                   '--family', spec['family'], '--runtime', str(BUNDLE / 'runtimes' / spec['family']),
                   '--members', str(member_file), '--chunk', str(spec.get('chunk', chunk)), '--out', str(self.out)]
        if lock_dir:
            command += ['--lock', str(lock_dir / f'gpu{spec["gpu"]}.lock')]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, bufsize=1, env=environment)
        self.thread = threading.Thread(target=self._read, args=(events,), daemon=True)
        self.thread.start()

    def _read(self, events):
        for line in self.process.stdout:
            try:
                events.put((self.name, self.generation, json.loads(line)))
            except json.JSONDecodeError:
                events.put((self.name, self.generation, dict(op='noise', text=line)))
        events.put((self.name, self.generation, dict(op='eof', returncode=self.process.wait())))

    def send(self, message):
        try:
            self.process.stdin.write(json.dumps(message) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # the process is gone; its eof event restarts it

    def run(self, batch, studies):
        self.outstanding.append(batch)
        self.send(dict(op='run', batch=batch, studies=studies))

    def respawn(self):
        self.kill()
        return Worker(*self.args, generation=self.generation + 1)

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass


def locate_weights(members, roots):
    names = {m['weight_file'] for m in members}
    found = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.name in names and path.is_file() and path.name not in found:
                found[path.name] = path
    missing = sorted(names - set(found))
    if missing:
        raise FileNotFoundError(f'weight files not found under {roots}: {missing}')
    with ThreadPoolExecutor(4) as pool:
        digests = dict(zip(sorted(found), pool.map(lambda n: sha256(found[n]), sorted(found))))
    for m in members:
        if digests[m['weight_file']] != m['weight_sha256']:
            raise ValueError(f'{m["id"]}: weight file {found[m["weight_file"]]} sha256 differs')
    return {m['id']: str(found[m['weight_file']]) for m in members}


def check_contracts(pipeline):
    contracts = {}
    for name in ('legacy384', 'native224', 'native384', 'native512'):
        contract = preprocessing_contract(json.loads((BUNDLE / 'contracts' / f'{name}.json').read_text()))
        if preprocess_signature(pipeline) != contract['source_preprocess_signature']:
            raise ValueError(f'{name}: DICOM preprocessing configuration differs from training')
        if (contract['slots'] != [list(slot) for slot in SLOTS] or contract['span'] != list(SPAN)
                or contract['neighbor_gap_ratio'] != 1.8 or contract['source_schema'] != 2
                or float(contract.get('crop_fov_mm')) != 140.0):
            raise ValueError(f'{name}: window builder differs from training')
        if name != 'legacy384' and (contract.get('crop_stage') != rni.NATIVE_STAGE
                                    or int(contract['image']) != int(name[len('native'):])):
            raise ValueError(f'{name}: not the native single-resample corpus')
        contracts[name] = contract
    return contracts


def main():
    started = time.monotonic()
    data_root = Path(os.environ['ENSEMBLE_DATA_ROOT'])
    output = Path(os.environ['ENSEMBLE_OUTPUT'])
    member_dir = Path(os.environ.get('ENSEMBLE_MEMBER_DIR', output.parent / 'members'))
    roots = os.environ.get('ENSEMBLE_WEIGHT_ROOTS', '/kaggle/input').split(os.pathsep)
    workers_n, reader_threads = env('ENSEMBLE_WORKERS', 3), env('ENSEMBLE_READER_THREADS', 4)
    study_batch, chunk, inflight = env('ENSEMBLE_STUDY_BATCH', 2), env('ENSEMBLE_CHUNK', 16), env('ENSEMBLE_INFLIGHT', 6)
    table = json.loads((BUNDLE / 'members.json').read_text())
    members = table['members']
    weights = np.array([m.get('class_weights', [float(c in m.get('columns', LABEL_COLS)) for c in LABEL_COLS]) for m in members], dtype=np.float64)
    if weights.shape != (len(members), 12) or not np.isfinite(weights).all() or (weights < 0).any() or not (weights.sum(0) > 0).all():
        raise ValueError('invalid per-column probability weights')
    use = weights > 0
    if any(set(m.get('columns', LABEL_COLS)) - set(LABEL_COLS) for m in members) or not use.any(0).all():
        raise ValueError('every member column must be a label and every label needs a routed member')
    by_id = {m['id']: m for m in members}
    lanes = json.loads(Path(os.environ.get('ENSEMBLE_LANES', BUNDLE / 'lanes.json')).read_text())
    planned = [i for spec in lanes['workers'] for i in spec['members']]
    if sorted(planned) != sorted(by_id) or len(planned) != len(set(planned)):
        raise ValueError('the worker plan must cover every member exactly once')
    if any(by_id[i]['family'] != spec['family'] for spec in lanes['workers'] for i in spec['members']):
        raise ValueError('a member is planned on a foreign runtime family')

    pipeline = load_config(str(BUNDLE / 'preprocess.yaml'))
    pipeline.blacklist = []
    contracts = check_contracts(pipeline)
    ids = pd.read_csv(data_root / 'test.csv').StudyInstanceUID.astype(str).tolist()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('submission IDs must be nonempty and unique')
    series = pd.read_csv(data_root / 'test_series.csv')
    records = {str(uid): rows for uid, rows in series.groupby(series.StudyInstanceUID.astype(str))}
    if not set(ids) <= set(records):
        raise ValueError('test series table misses requested studies')
    sample = data_root / 'sample_submission.csv'
    if sample.exists() and pd.read_csv(sample, nrows=0).columns.tolist() != ['StudyInstanceUID', *LABEL_COLS]:
        raise ValueError('submission label order differs from sample')
    native = sorted({int(m['variant']['source'][len('native'):]) for m in members
                     if m['variant']['source'].startswith('native')})

    run_dir = Path(os.environ.get('ENSEMBLE_RUN_DIR') or tempfile.mkdtemp(prefix='ens36_run_'))
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    cache_root = Path(os.environ.get('ENSEMBLE_CACHE_DIR') or tempfile.mkdtemp(prefix='ens36_cache_'))
    cache_root.mkdir(parents=True, exist_ok=True)
    init_args = (pipeline, contracts['legacy384'], str(data_root), reader_threads, native, str(cache_root))
    init_pool(*init_args)  # the parent retries failed studies itself
    context = get_context('fork')

    def new_pool():
        pool = ProcessPoolExecutor(max_workers=workers_n, mp_context=context, initializer=init_pool,
                                   initargs=init_args)
        # Fork every process now, before any thread of this parent exists.
        for future in [pool.submit(nothing) for _ in range(workers_n)]:
            future.result()
        return pool

    pool = new_pool()
    weight_paths = locate_weights(members, roots)
    events = queue.Queue()
    lock_dir = run_dir / 'locks' if env('ENSEMBLE_GPU_LOCK', 1) else None
    if lock_dir:
        lock_dir.mkdir(exist_ok=True)
    max_deaths, watchdog = env('ENSEMBLE_MAX_DEATHS', 4), env('ENSEMBLE_WATCHDOG_S', 1200.0)
    deadline = env('ENSEMBLE_DEADLINE_S', 28800.0)
    workers = {}
    for spec in lanes['workers']:
        payload = [dict(by_id[i], weight_path=weight_paths[i]) for i in spec['members']]
        workers[spec['name']] = Worker(spec, payload, run_dir, chunk, lock_dir, events)
    deaths = {name: 0 for name in workers}
    batch_deaths = {}       # (worker, batch) -> deaths while that batch was the worker's oldest outstanding one
    dropped = {}            # worker -> reason; its members are left out of every later study
    incidents = []          # every degradation, for the receipt
    logits = {m['id']: {} for m in members}
    member_failures = {m['id']: {} for m in members}
    ready = {}

    def log_incident(**incident):
        incident['elapsed_s'] = round(time.monotonic() - started, 1)
        incidents.append(incident)
        print(json.dumps(dict(event='incident', **incident))[:3000], flush=True)

    while len(ready) + len(dropped) < len(workers):
        try:
            name, generation, message = events.get(timeout=10)
        except queue.Empty:
            stuck = [n for n, w in workers.items() if n not in ready and n not in dropped
                     and time.monotonic() - w.progress > watchdog]
            if not stuck:
                continue
            name, generation, message = stuck[0], workers[stuck[0]].generation, dict(op='eof', returncode='load watchdog')
            workers[name].kill()
        worker = workers[name]
        if generation != worker.generation:
            continue
        if message['op'] == 'ready':
            ready[name] = message
            worker.ready, worker.progress = True, time.monotonic()
            print(json.dumps(dict(event='worker_ready', name=name, seconds=round(message['seconds'], 1),
                                  device=message['device'], members=[m['id'] for m in message['members']],
                                  extra_keys={m['id']: m['extra_config_keys'] for m in message['members']},
                                  memory_mb=message['memory_allocated'] >> 20)), flush=True)
        elif message['op'] == 'error':
            log_incident(kind='load_error', worker=name, generation=generation, error=str(message['error'])[-2000:])
        elif message['op'] == 'eof':
            deaths[name] += 1
            if deaths[name] >= 2:
                worker.kill()
                dropped[name] = 'failed to load twice'
                log_incident(kind='worker_dropped', worker=name, reason=dropped[name])
            else:
                workers[name] = worker.respawn()
    if len(dropped) == len(workers):
        raise RuntimeError('no worker could load its members')
    loaded = time.monotonic()

    batches = [ids[i:i + study_batch] for i in range(0, len(ids), study_batch)]
    index_of = {uid: i for i, uid in enumerate(ids)}
    caches, failed, prep_stats = {}, {}, []
    pending = {}            # future -> study index
    submitted = 0           # studies handed to the pool, in order
    dispatched = 0          # batches sent to the workers
    done = {b: set() for b in range(len(batches))}
    batch_studies = {}
    complete_batches = set()
    finished = 0            # batches every live worker has scored (contiguous prefix not required)
    oldest = 0              # lowest batch not yet finished by all live workers
    worker_seconds = {name: 0.0 for name in workers}
    restarts = 0
    last_report = time.monotonic()
    stopping = False
    crash = None

    def submit_more():
        nonlocal submitted
        limit = min(len(ids), (oldest + inflight + 2) * study_batch)
        while not stopping and submitted < limit and len(pending) < workers_n + 2:
            uid = ids[submitted]
            pending[pool.submit(prepare_all, submitted, uid, records[uid])] = submitted
            submitted += 1

    def recover(index, error):
        uid = ids[index]
        try:
            caches[index] = prepare_all(index, uid, records[uid])['cache']
        except Exception as second:  # noqa: BLE001
            failed[uid] = f'{type(error).__name__}: {error} | parent retry {type(second).__name__}: {second}'
            caches[index] = None
            print(json.dumps(dict(event='study_failed', uid=uid, error=failed[uid][:500])), flush=True)

    def settle():
        nonlocal finished, oldest
        live = {name for name in workers if name not in dropped}
        for b in range(oldest, dispatched):
            if b not in complete_batches and live <= done[b]:
                complete_batches.add(b)
                finished += 1
                for uid in batches[b]:
                    cache = caches.get(index_of[uid])
                    if cache:
                        shutil.rmtree(cache, ignore_errors=True)
        while oldest < dispatched and oldest in complete_batches:
            oldest += 1

    def on_death(name, reason):
        worker = workers[name]
        deaths[name] += 1
        current = worker.outstanding[0] if worker.outstanding else None
        if current is not None:
            batch_deaths[(name, current)] = batch_deaths.get((name, current), 0) + 1
        log_incident(kind='worker_died', worker=name, generation=worker.generation, reason=reason, batch=current,
                     deaths=deaths[name])
        if deaths[name] > max_deaths:
            worker.kill()
            dropped[name] = f'died {deaths[name]} times; last: {reason}'
            log_incident(kind='worker_dropped', worker=name, reason=dropped[name])
            return
        fresh = worker.respawn()
        workers[name] = fresh
        for b in worker.outstanding:
            if batch_deaths.get((name, b), 0) >= 2:
                done[b].add(name)
                log_incident(kind='batch_skipped', worker=name, batch=b, studies=batches[b])
            else:
                fresh.run(b, batch_studies[b])

    try:
        while True:
            settle()
            if finished == dispatched and (dispatched == len(batches) or stopping):
                break
            if not stopping and time.monotonic() - started > deadline:
                stopping = True
                log_incident(kind='deadline', dispatched=dispatched, batches=len(batches))
            if stopping and time.monotonic() - started > deadline + 900:
                log_incident(kind='deadline_abandon', finished=finished, dispatched=dispatched)
                break
            submit_more()
            if pending:
                complete, _ = wait(list(pending), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in complete:
                    index = pending.pop(future)
                    try:
                        stats = future.result()
                        caches[index] = stats['cache']
                        prep_stats.append(stats)
                    except BrokenProcessPool as error:
                        restarts += 1
                        outstanding = [i for i in pending.values()]
                        pending.clear()
                        pool.shutdown(cancel_futures=True)
                        recover(index, error)
                        if restarts > 8:  # decode the rest in this process
                            for other in outstanding:
                                recover(other, error)
                            workers_n = 0
                            pool = ThreadPoolExecutor(1)
                            log_incident(kind='pool_disabled', restarts=restarts)
                        else:
                            pool = new_pool()
                            for other in outstanding:
                                pending[pool.submit(prepare_all, other, ids[other], records[ids[other]])] = other
                    except Exception as error:  # noqa: BLE001
                        recover(index, error)
            while (not stopping and dispatched < len(batches) and dispatched - oldest < inflight
                   and all(index_of[uid] in caches for uid in batches[dispatched])):
                studies = [[uid, caches[index_of[uid]]] for uid in batches[dispatched]]
                batch_studies[dispatched] = studies
                for name, worker in workers.items():
                    if name not in dropped:
                        worker.run(dispatched, studies)
                dispatched += 1
            try:
                while True:
                    name, generation, message = events.get(timeout=0 if pending else 0.05)
                    worker = workers[name]
                    if generation != worker.generation or name in dropped:
                        continue
                    op = message['op']
                    if op == 'ready':
                        ready[name] = message
                        worker.ready, worker.progress = True, time.monotonic()
                        log_incident(kind='worker_reloaded', worker=name, generation=generation)
                    elif op == 'busy':
                        worker.busy_since = time.monotonic()
                    elif op == 'done':
                        b = message['batch']
                        worker.busy_since, worker.progress = None, time.monotonic()
                        if b in worker.outstanding:
                            worker.outstanding.remove(b)
                        done[b].add(name)
                        worker_seconds[name] += message['seconds']
                        for member_id, values in message['logits'].items():
                            for uid, row in values.items():
                                logits[member_id][uid] = np.asarray(row, dtype=np.float32)
                        for member_id, errors in message['failures'].items():
                            member_failures[member_id].update(errors)
                            log_incident(kind='member_failed', worker=name, member=member_id, batch=b, errors=errors)
                    elif op == 'error':
                        log_incident(kind='worker_error', worker=name, generation=generation,
                                     batch=message.get('batch'), error=str(message.get('error'))[-2000:])
                    elif op == 'eof':
                        on_death(name, f'exited with {message.get("returncode")}')
            except queue.Empty:
                pass
            now = time.monotonic()
            for name, worker in list(workers.items()):
                if name in dropped or not worker.outstanding:
                    continue
                if worker.busy_since is not None and now - worker.busy_since > watchdog:
                    on_death(name, f'watchdog: held the GPU for {now - worker.busy_since:.0f} s')
                elif now - worker.progress > 3 * watchdog:
                    on_death(name, f'watchdog: no progress for {now - worker.progress:.0f} s')
            if now - last_report > 60:
                last_report = now
                print(json.dumps(dict(event='progress', finished_batches=finished, batches=len(batches),
                                      studies_prepared=len(caches), elapsed_s=round(now - started, 1),
                                      worker_busy_s={k: round(v, 1) for k, v in worker_seconds.items()},
                                      dropped=sorted(dropped), incidents=len(incidents))), flush=True)
    except Exception:  # noqa: BLE001
        crash = traceback.format_exc()
        log_incident(kind='orchestrator_error', error=crash[-3000:])
    if pending:
        for process in list(getattr(pool, '_processes', {}).values()):
            process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    else:
        pool.shutdown()
    results = {}
    waiting = {name for name in workers if name not in dropped}
    for name in waiting:
        workers[name].send(dict(op='exit'))
    exit_deadline = time.monotonic() + 600
    while waiting and time.monotonic() < exit_deadline:
        try:
            name, generation, message = events.get(timeout=5)
        except queue.Empty:
            continue
        if generation != workers[name].generation:
            continue
        if message['op'] == 'result':
            results[name] = message
            waiting.discard(name)
        elif message['op'] == 'eof':
            waiting.discard(name)
    for worker in workers.values():
        try:
            worker.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()

    member_probabilities = {}
    member_dir.mkdir(parents=True, exist_ok=True)
    for member in members:
        table_logits = logits[member['id']]
        values = np.full((len(ids), 12), np.nan)
        for row, uid in enumerate(ids):
            if uid in table_logits:
                values[row] = 1.0 / (1.0 + np.exp(-table_logits[uid].astype(np.float64)))
        member_probabilities[member['id']] = values
        frame = pd.DataFrame(values, columns=LABEL_COLS)
        frame.insert(0, 'StudyInstanceUID', ids)
        frame.to_csv(member_dir / f'{member["id"]}.csv', index=False)
    stacked = np.stack([member_probabilities[m['id']] for m in members])
    if not np.isfinite(stacked).all() or incidents or failed or dropped or crash:
        raise RuntimeError('G36 requires every model on every study; degraded output is forbidden')
    routed_cells_fallback = 0
    if np.isnan(stacked).any() or not use.all() or not np.all(weights == 1):
        masked = np.where(use[:, None, :], stacked, np.nan)
        counts = (np.isfinite(masked) * weights[:, None, :]).sum(0)
        everyone = np.isfinite(stacked).sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            ensemble = np.where(counts > 0, np.nansum(masked * weights[:, None, :], 0) / np.maximum(counts, 1e-300), np.nan)
            fallback = np.where(everyone > 0, np.nansum(stacked, 0) / np.maximum(everyone, 1), np.nan)
        routed_cells_fallback = int((~np.isfinite(ensemble) & np.isfinite(fallback)).sum())
        ensemble = np.where(np.isfinite(ensemble), ensemble, fallback)
    else:
        ensemble = stacked.mean(0)  # every member scored every study and every column: the plain mean
    scored = np.isfinite(ensemble).all(1)
    coverage = float(scored.mean())
    minimum = env('ENSEMBLE_MIN_COVERAGE_ON_ERROR', 0.9) if crash else env('ENSEMBLE_MIN_COVERAGE', 0.5)
    if coverage < minimum:
        raise RuntimeError(f'only {coverage:.3f} of the studies were scored' + (f'; orchestrator error:\n{crash}'
                                                                                 if crash else ''))
    if not scored.all():
        ensemble[~scored] = np.median(ensemble[scored], 0)
    if not np.isfinite(ensemble).all():
        raise ValueError('nonfinite ensemble probabilities')
    result = pd.DataFrame(ensemble, columns=LABEL_COLS)
    result.insert(0, 'StudyInstanceUID', ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.csv.tmp')
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    shutil.rmtree(cache_root, ignore_errors=True)
    prep = np.array([s['seconds'] for s in prep_stats]) if prep_stats else np.zeros(1)
    amp = {x['id']: x['eval_amp'] for r in ready.values() for x in r['members']}
    member_coverage = np.isfinite(stacked).all(2).sum(1)
    receipt = dict(
        schema='final-ensemble-w3-receipt-v1', combination=table['combination'], model_count=len(members),
        columns={m['id']: m.get('columns', LABEL_COLS) for m in members}, class_weights={m['id']: list(w) for m,w in zip(members,weights)}, routed_cells_fallback=routed_cells_fallback,
        member_ids=[m['id'] for m in members], studies=len(ids), failed_studies=failed,
        degraded=bool(incidents or failed), scored_fraction=coverage, unscored_studies=int((~scored).sum()),
        member_scored_studies={m['id']: int(n) for m, n in zip(members, member_coverage)},
        member_failures={k: v for k, v in member_failures.items() if v}, dropped_workers=dropped,
        worker_deaths=deaths, incidents=incidents, orchestrator_error=crash,
        members=[dict(id=m['id'], family=m['family'], arch=m['arch'], resolution=m['resolution'], readout=m['readout'],
                      weight_file=m['weight_file'], weight_sha256=m['weight_sha256'], variant=m['variant'],
                      eval_amp=amp.get(m['id'])) for m in members],
        workers={name: dict(gpu=w.spec['gpu'], family=w.spec['family'], members=w.spec['members'],
                            device=ready.get(name, {}).get('device', 'not loaded'),
                            load_s=ready.get(name, {}).get('seconds'), busy_s=worker_seconds[name],
                            member_seconds=results.get(name, {}).get('member_seconds'),
                            peak_memory_mb=results.get(name, {}).get('peak_memory', 0) >> 20)
                 for name, w in workers.items()},
        preprocessing=dict(workers=workers_n, reader_threads=reader_threads, native_resolutions=native,
                           passes_per_study=1, mean_s=float(prep.mean()), p90_s=float(np.percentile(prep, 90)),
                           pool_restarts=restarts),
        study_batch=study_batch, encode_chunk=chunk, inflight=inflight,
        timing=dict(load_s=loaded - started, total_s=time.monotonic() - started,
                    scoring_s=time.monotonic() - loaded),
        submission_sha256=sha256(output))
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps(dict(event='finished', studies=len(ids), failed=len(failed), total_s=receipt['timing']['total_s'],
                          load_s=receipt['timing']['load_s'], degraded=receipt['degraded'], scored_fraction=coverage,
                          dropped=sorted(dropped), incidents=len(incidents))), flush=True)


if __name__ == '__main__':
    main()
