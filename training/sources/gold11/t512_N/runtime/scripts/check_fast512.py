#!/usr/bin/env python3
"""Check kneemri.fast512 against the eager backbone: train-forward/grad agreement under row padding,
bit-identical eval path and state_dict keys, and step time for E2E_D-like variable window counts.

Usage: check_fast512.py CONFIG.yaml [rows ...]
"""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import yaml

from kneemri import coatnet_model, fast512


def main():
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    rows = [int(v) for v in sys.argv[2:]] or [37, 64, 21, 48]
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    eager = coatnet_model.build_model(cfg, pretrained=True).cuda().to(memory_format=torch.channels_last)
    fast = coatnet_model.build_model(cfg, pretrained=True).cuda().to(memory_format=torch.channels_last)
    fast.load_state_dict(eager.state_dict())
    keys = list(fast.state_dict())
    fast512.install(fast, int(cfg.get('compile_bucket', 16)))
    assert list(fast.state_dict()) == keys, 'state_dict keys changed'
    size = int(cfg['resolution'])
    for m in (eager, fast):
        m.train()
        # drop_path is stochastic; compare deterministic forwards
        for mod in m.modules():
            if type(mod).__name__ == 'DropPath':
                mod.drop_prob = 0.0
    def run(m, x, dtype):
        m.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype, enabled=dtype is not None):
            y = m.backbone.forward_features(x)
            loss = y.float().pow(2).mean()
        loss.backward()
        grad = torch.cat([p.grad.float().flatten() for p in m.backbone.parameters() if p.grad is not None])
        first = next(p for p in m.backbone.parameters() if p.grad is not None).grad.float().flatten()
        return y.float(), grad, first.clone()

    def rel(a, b):
        return ((a - b).norm() / a.norm()).item()
    for n in rows:
        x = torch.randn(n, 3, size, size, device='cuda').contiguous(memory_format=torch.channels_last)
        ref = run(eager, x, None)  # fp32 eager reference
        e16 = run(eager, x, torch.bfloat16)
        c16 = run(fast, x, torch.bfloat16)
        print(f'rows {n}: vs fp32 eager -> eager-bf16 out {rel(ref[0], e16[0]):.2e} grad {rel(ref[1], e16[1]):.2e} '
              f'stem {rel(ref[2], e16[2]):.2e} | compiled-bf16 out {rel(ref[0], c16[0]):.2e} '
              f'grad {rel(ref[1], c16[1]):.2e} stem {rel(ref[2], c16[2]):.2e}', flush=True)
        assert c16[0].shape == ref[0].shape
        assert rel(ref[1], c16[1]) < 2 * rel(ref[1], e16[1]) + 1e-3, 'compiled grads much noisier than eager bf16'
    for m in (eager, fast):
        m.eval()
    x = torch.randn(13, 3, size, size, device='cuda').contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        a, b = eager.backbone.forward_features(x), fast.backbone.forward_features(x)
    assert torch.equal(a, b), 'eval path must be the eager path'
    print('eval path bit-identical', flush=True)
    for name, m in (('eager', eager), ('compiled', fast)):
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-6)
        times = []
        for i in range(3 * len(rows)):
            n = rows[i % len(rows)]
            x = torch.randn(n, 3, size, size, device='cuda').contiguous(memory_format=torch.channels_last)
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = m.backbone.forward_features(x).float().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= len(rows):
                times.append((time.time() - t) / n)
        print(f'{name}: {1000 * sum(times) / len(times):.2f} ms/image (rows {rows})', flush=True)


if __name__ == '__main__':
    main()
