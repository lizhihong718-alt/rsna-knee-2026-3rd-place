"""Optional eval-only optimizations for the pinned ExactUint8Inference worker.

No change to weights on disk, model membership, crop, window order, chunk size,
FP32 normalization, autocast or probability combination. Cache lifetime is one
worker batch and cached tensors are released before the inter-process GPU lock.
"""
import types
import numpy as np
import torch
import torch.nn.functional as F


def freeze_standardized_convolutions(model):
    """Cache timm StdConv2d's exact FP32 operation on its inference device.

    Keep original parameters, disable training and reject changed weights.
    No algebraic CPU rewrite: use the same F.batch_norm kernel as timm.
    """
    from timm.layers.std_conv import StdConv2d, StdConv2dSame
    from timm.layers.padding import pad_same
    count = 0
    for module in model.modules():
        if type(module) not in (StdConv2d, StdConv2dSame):
            continue
        if module.training:
            raise ValueError('standardized weight cache is inference-only')
        with torch.no_grad(), torch.autocast(device_type=module.weight.device.type, enabled=False):
            weight = F.batch_norm(module.weight.reshape(1, module.out_channels, -1),
                                  None, None, training=True, momentum=0., eps=module.eps).reshape_as(module.weight)
        module.register_buffer('_eval_standardized_weight', weight, persistent=False)
        module._eval_weight_version = module.weight._version
        module._eval_weight_pointer = module.weight.data_ptr()
        def forward(self, x):
            if (self.training or self.weight._version != self._eval_weight_version
                    or self.weight.data_ptr() != self._eval_weight_pointer):
                raise RuntimeError('frozen standardized weight was mutated; reload the inference model')
            if getattr(self, 'same_pad', False):
                x = pad_same(x, self.kernel_size, self.stride, self.dilation)
            return F.conv2d(x, self._eval_standardized_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        module.forward = types.MethodType(forward, module)
        count += 1
    return count


class PreparedBatch:
    """Original padded input, metadata and FP32 normalization, built once."""
    def __init__(self, engine, payloads):
        self.counts = [int(p['count']) for p in payloads]
        if any(c < 0 for c in self.counts):
            raise ValueError('compact payload count must be nonnegative')
        self.batch_size = len(payloads)
        self.images, self.valid, self.groups, self.metadata = [], [], [], []
        self.device = engine.device
        if not sum(self.counts):
            return
        chunk, resolution = engine.chunk_size, engine.resolution
        slots = ((max(self.counts) + chunk - 1) // chunk) * chunk
        windows = np.zeros((len(payloads), slots, 3, resolution, resolution), dtype=np.uint8)
        valid = np.zeros((len(payloads), slots), dtype=bool)
        groups = np.full((len(payloads), slots), -1, dtype=np.int64)
        for row, (p, count) in enumerate(zip(payloads, self.counts)):
            if not count:
                self.metadata.append(None)
                continue
            data = np.ascontiguousarray(p['windows'])
            if data.dtype != np.uint8 or data.shape != (count, 3, resolution, resolution):
                raise ValueError('optimized inference requires native-resolution uint8 windows')
            windows[row, :count] = data
            valid[row, :count] = True
            groups[row, :count] = np.asarray(p['group_ids'], dtype=np.int64)
            self.metadata.append(engine._metadata(count, p['group_ids'], p['series_ids'],
                                                  p['plane_ids'], p['positions'], p['position_mask']))
        pixels = engine._to_device(windows)
        for start in range(0, slots, chunk):
            stop = start + chunk
            self.images.append(engine._normalize_batch(pixels[:, start:stop]))
            self.valid.append(torch.as_tensor(valid[:, start:stop], device=engine.device))
            self.groups.append(torch.as_tensor(groups[:, start:stop], device=engine.device))

    def logits(self, model):
        if not sum(self.counts):
            return torch.zeros(self.batch_size, model.n, device=self.device)
        encoded = [model.encode(x, v, g) for x, v, g in zip(self.images, self.valid, self.groups)]
        features = torch.cat(encoded, dim=1)
        outputs = []
        for row, (count, metadata) in enumerate(zip(self.counts, self.metadata)):
            if not count:
                outputs.append(torch.zeros(1, model.n, device=self.device))
            else:
                outputs.append(model.head(features[row:row+1, :count], **metadata).float())
        return torch.cat(outputs, dim=0)


class BatchCache:
    def __init__(self):
        self.entries = {}
        self.references = []
        self.hits = self.misses = 0

    def clear(self):
        self.entries.clear()
        self.references.clear()

    def logits(self, member, payloads):
        engine = member.engine
        key = (tuple(id(p) for p in payloads), member.key, engine.resolution,
               engine.chunk_size, member.cfg['normalization'], str(engine.device))
        if key not in self.entries:
            self.entries[key] = PreparedBatch(engine, payloads)
            self.references.extend(payloads)  # prohibit recycled object IDs within the batch
            self.misses += 1
        else:
            self.hits += 1
        return self.entries[key].logits(member.model)
