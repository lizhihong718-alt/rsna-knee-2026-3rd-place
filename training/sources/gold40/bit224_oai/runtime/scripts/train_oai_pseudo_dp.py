#!/usr/bin/env python3
"""train_oai_pseudo.py with the DataParallel replicas called from the main thread (10-17, for --devices a,b).

torch's DataParallel.parallel_apply runs each replica's forward in a new Python thread per call. With this model the
threaded forward took ~1.1 s per step on two H20 (backward 0.24 s), so two cards were 3x slower than one. Calling the
replicas one after another from the main thread took ~0.13 s: kernel launches are asynchronous, so the two cards still
overlap. Same replicas, inputs, current streams, autocast and grad mode as parallel_apply -> the same computation.
Without --devices (single card) the trainer never builds a DataParallel and this is train_oai_pseudo.py.
"""
import torch

import train_oai_pseudo


def sequential_apply(self, replicas, inputs, kwargs):
    outputs = []
    for module, args, kw, device in zip(replicas, inputs, kwargs, self.device_ids):
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.current_stream(device)):
            outputs.append(module(*args, **kw))
    return outputs


if __name__ == '__main__':
    torch.nn.DataParallel.parallel_apply = sequential_apply
    train_oai_pseudo.main()
