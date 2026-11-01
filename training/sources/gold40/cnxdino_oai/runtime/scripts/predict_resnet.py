#!/usr/bin/env python3
"""Predict from exactly one ResNet checkpoint on an original-neighbor cache."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import train_coatnet_ablation as trainer
from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags
from kneemri.labels import LABEL_COLS
from kneemri import resnet_model
from kneemri.resnet_model import build_model

if hasattr(resnet_model, 'register_config'):
    resnet_model.register_config(trainer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-submission', type=Path)
    parser.add_argument('--training-manifest', type=Path,
                        help='Original training cache manifest, required when its local path is unavailable')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['lab'] != LABEL_COLS or checkpoint['receipt']['schema'] != 'resnet-study-mil-v1':
        raise ValueError('checkpoint is not an audited ResNet study model')
    cfg = dict(checkpoint['config'], corpus=str(args.corpus.resolve()), eval_label_source='report_weak')
    training_manifest = args.training_manifest or Path(checkpoint['config']['corpus']) / 'manifest.json'
    if trainer.hash_file(training_manifest) != checkpoint['receipt']['corpus_manifest_sha256']:
        raise ValueError('training manifest does not match the checkpoint receipt')
    expected_pre = json.loads(training_manifest.read_text())['preprocessing']
    input_manifest = json.loads((args.corpus / 'manifest.json').read_text())
    geometry_keys = ['schema', 'window_semantics', 'slots', 'span', 'image', 'neighbor_gap_ratio',
                     'crop_fov_mm', 'crop_stage', 'source_orientation']
    if any(expected_pre[key] != input_manifest['preprocessing'].get(key) for key in geometry_keys):
        raise ValueError('inference geometry differs from the training cache')
    torch.set_num_threads(cfg['num_threads'])
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    if cfg['channels_last'] and device.type == 'cuda':
        model.to(memory_format=torch.channels_last)
    ids = input_manifest['study_uids']
    labels = {uid: np.zeros(12, dtype=np.float32) for uid in ids}
    dataset = CoAtNetOriginalNeighborBags(cfg, ids, ids, labels, train=False,
                                         uint8_output=cfg['uint8_transfer'])
    loader = trainer.make_loader(dataset, cfg, device)
    dtype = trainer.select_eval_amp(cfg, device)
    predictions, observed = [], []
    with torch.inference_mode():
        for batch in loader:
            logits = trainer.inference_logits(model, batch, device, dtype, cfg['eval_encode_chunk'])
            predictions.append(logits.float().sigmoid().cpu().numpy())
            observed.extend(batch['study_ids'])
    values = np.concatenate(predictions)
    if observed != ids or not np.isfinite(values).all():
        raise ValueError('prediction order/finite-value check failed')
    result = pd.DataFrame(values, columns=LABEL_COLS, index=pd.Index(ids, name='StudyInstanceUID'))
    if args.sample_submission:
        sample = pd.read_csv(args.sample_submission)
        if set(sample.StudyInstanceUID) != set(ids) or sample.StudyInstanceUID.duplicated().any():
            raise ValueError('sample submission IDs differ from the cache')
        result = result.loc[sample.StudyInstanceUID]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out)
    trainer.write_json(args.out.with_suffix('.receipt.json'), dict(
        checkpoint_sha256=trainer.hash_file(args.checkpoint), model_count=1,
        seed=cfg['seed'], checkpoint_epoch=checkpoint['epoch'] + 1, tta=False,
        arch=cfg['arch'], resolution=cfg['resolution'], windows=cfg['eval_windows'],
        corpus_manifest_sha256=trainer.hash_file(args.corpus / 'manifest.json'),
        rows=len(ids), submission_sha256=trainer.hash_file(args.out)))
    print(json.dumps({'rows': len(ids), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
