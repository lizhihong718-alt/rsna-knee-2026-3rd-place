"""Four-GPU window-parallel v152 adapter, with the original global study batches.

All ranks receive identical bags. Only the image encoder/spatial pool is sharded
over windows; autograd all-gather restores the full bag before the original head.
DDP averages head gradients and the summed all-gather gradients exactly once.
SyncBatchNorm sees the original global window batch, including short last batches.
No study padding, dropped studies, or changed gradient-accumulation denominator.
"""
from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from timm.layers import convert_sync_batchnorm


class DistributedModel(nn.Module):
    def __init__(self, module, local_rank):
        super().__init__()
        self.ddp = nn.parallel.DistributedDataParallel(
            module, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, gradient_as_bucket_view=True)

    def forward(self, *args, **kwargs):
        return self.ddp(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('ddp').module, name)

    def named_parameters(self, *args, **kwargs):
        return self.ddp.module.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.ddp.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.ddp.module.load_state_dict(*args, **kwargs)


def shard_encoder(module):
    original = module.encode

    def encode(self, x, window_mask=None, group_ids=None, return_backbone_features=False):
        if not self.training:
            return original(x, window_mask, group_ids, return_backbone_features)
        if return_backbone_features:
            raise ValueError('window-parallel adapter is only for the v152 BCE objective')
        rank, world = dist.get_rank(), dist.get_world_size()
        b, k = x.shape[:2]
        if window_mask is None:
            window_mask = torch.ones((b, k), dtype=torch.bool, device=x.device)
        if group_ids is None:
            group_ids = torch.full((b, k), -1, dtype=torch.long, device=x.device)
        padding = (-k) % world
        if padding:
            x = torch.cat((x, x.new_zeros((b, padding, *x.shape[2:]))), dim=1)
            window_mask = torch.cat((window_mask, window_mask.new_zeros((b, padding))), dim=1)
            group_ids = torch.cat((group_ids, group_ids.new_full((b, padding), -1)), dim=1)
        local_mask = window_mask[:, rank::world]
        counts = window_mask.sum(0)
        if any(not bool(counts[r::world].sum()) for r in range(world)):
            raise ValueError('insufficient valid windows for synchronous encoder shards')
        encoded = original(x[:, rank::world], local_mask, group_ids[:, rank::world])
        pieces = all_gather(encoded.contiguous())
        return torch.stack(pieces, dim=2).flatten(1, 2)[:, :k]

    module.encode = types.MethodType(encode, module)
    return module


def setup(trainer, args):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    cpu_group = dist.new_group(backend='gloo', timeout=timedelta(hours=2))
    args.device = 'cuda:' + str(local_rank)
    # GPU5 is shared; its existing process is never touched. Cap our allocator
    # below its available memory, leaving room for CUDA/NCCL outside the allocator.
    if os.environ.get('ASTRA_LIMITED_RANK') == str(rank):
        limit_gib = float(os.environ.get('ASTRA_GPU_LIMIT_GIB', '12'))
        total = torch.cuda.get_device_properties(local_rank).total_memory
        torch.cuda.set_per_process_memory_fraction(limit_gib * 2**30 / total, local_rank)

    def broadcast(value):
        box = [value]
        dist.broadcast_object_list(box, src=0, group=cpu_group)
        return box[0]

    original_write, original_save = trainer.write_json, trainer.save_atomic
    original_receipt, original_build = trainer.source_receipt, trainer.build_model
    original_evaluate, original_inference = trainer.evaluate, trainer.inference_logits
    original_copy = trainer.shutil.copy2

    def write_json(path, value):
        if rank == 0:
            original_write(path, value)
        dist.barrier(group=cpu_group)

    def save_atomic(value, path):
        if rank == 0:
            original_save(value, path)
        dist.barrier(group=cpu_group)

    def copy2(*args, **kwargs):
        # Source receipt is rank0-only, so this deliberately has no collective.
        if rank == 0:
            return original_copy(*args, **kwargs)

    def source_receipt(output, cfg, cohort, synthetic=False):
        receipt = original_receipt(output, cfg, cohort, synthetic) if rank == 0 else None
        if rank == 0:
            support = Path(__file__)
            original_copy(support, output / 'source/scripts' / support.name)
            receipt['sources']['scripts/' + support.name] = trainer.hash_file(support)
            receipt['distributed'] = dict(world_size=world, scheme='window-parallel encoder, replicated study head',
                global_batch_size=cfg['batch_size'], gradient_accumulation=cfg['gradient_accumulation'],
                effective_study_batch_size=cfg['batch_size']*cfg['gradient_accumulation'],
                synchronization='timm SyncBatchNormAct and DDP', no_study_padding=True,
                evaluation='disjoint study shards; original fp16 inference, chunk capped at 24',
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                limited_rank=os.environ.get('ASTRA_LIMITED_RANK'),
                gpu_limit_gib=os.environ.get('ASTRA_GPU_LIMIT_GIB'))
        return broadcast(receipt)

    def build_model(cfg, pretrained=True):
        if cfg['rank_weight'] != 0 or cfg['batch_size'] != 4 or cfg['gradient_accumulation'] != 2:
            raise ValueError('this adapter preserves the v152 batch4/accumulation2 BCE pipeline')
        module = original_build(cfg, pretrained=pretrained)
        keys = set(module.state_dict())
        module = convert_sync_batchnorm(module).to(args.device)
        if set(module.state_dict()) != keys:
            raise ValueError('SyncBatchNorm conversion changed the inference state-dict contract')
        return DistributedModel(shard_encoder(module), local_rank)

    def evaluate(model, loader, cfg, device, dtype, output, epoch, budget, role=None, subdirectory=None):
        model.eval()
        ids = list(loader.dataset.ids)
        dataset = loader.dataset
        dataset.ids = ids[rank::world]
        local_loader = trainer.make_loader(dataset, cfg, device)
        records = []
        with torch.no_grad():
            for batch in local_loader:
                logits = original_inference(model, batch, device, dtype, min(24, cfg['eval_encode_chunk']))
                records.append(dict(study_ids=batch['study_ids'], logits=logits.numpy(),
                    targets=batch['targets'].numpy(), provenance=batch['provenance'],
                    label_source=batch['label_source']))
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(records, gathered, dst=0, group=cpu_group)
        report = None
        if rank == 0:
            rows = [row for shard in gathered for row in shard]
            by_uid = {row['study_ids'][0]: row for row in rows}
            if len(rows) != len(ids) or set(by_uid) != set(ids):
                raise ValueError('distributed evaluation omitted or duplicated studies')
            replay = [dict(_logits=torch.from_numpy(by_uid[u]['logits']),
                           targets=torch.from_numpy(by_uid[u]['targets']),
                           study_ids=by_uid[u]['study_ids'], provenance=by_uid[u]['provenance'],
                           label_source=by_uid[u]['label_source']) for u in ids]
            # Reuse the original reporter, metric implementation and NPZ contract.
            trainer.inference_logits = lambda model, batch, *unused: batch['_logits']
            trainer.write_json = original_write
            try:
                report = original_evaluate(model, replay, cfg, device, dtype, output, epoch, budget, role, subdirectory)
            finally:
                trainer.inference_logits = original_inference
                trainer.write_json = write_json
        return broadcast(report)

    trainer.write_json = write_json
    trainer.save_atomic = save_atomic
    trainer.shutil.copy2 = copy2
    trainer.source_receipt = source_receipt
    trainer.build_model = build_model
    trainer.evaluate = evaluate
    # Keep one readable log; any rank's exception still reaches torchrun stderr.
    trainer.print = (print if rank == 0 else lambda *a, **kw: None)
