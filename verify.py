"""CPU-only archive, weight and fixed fusion verification."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    bundle = root / 'inference'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or sha256(path) != expected:
            raise ValueError('inference file identity differs: ' + relative)
    doc = json.loads((bundle / 'members.json').read_text())
    members = doc['members']
    if [m['id'] for m in members] != manifest['member_ids'] or len(members) != 36:
        raise ValueError('model order/count mismatch')
    groups = doc['groups']
    mature = set(groups['mature12'])
    matrix = {m for group in groups['matrix20_by_backbone'].values() for m in group}
    expert = set(groups['expert8'])
    weak = {'Lateral OA', 'PF OA', 'Synovitis'}
    for m in members:
        for c, label in enumerate(doc['labels']):
            base = (0.5 / 12 if m['id'] in mature else 0) + (0.5 / 20 if m['id'] in matrix else 0)
            expected = 0.5 * base + (0.5 / 8 if m['id'] in expert else 0) if label in weak else base
            if not math.isclose(m['class_weights'][c], expected, abs_tol=1e-15):
                raise ValueError('group fusion differs: ' + m['id'])
        if args.weights and sha256(args.weights / m['weight_file']) != m['weight_sha256']:
            raise ValueError('weight identity differs: ' + m['id'])
    if any(not math.isclose(sum(m['class_weights'][c] for m in members), 1, abs_tol=1e-14) for c in range(12)):
        raise ValueError('class coefficients do not sum to one')
    print(json.dumps({'model_count': 36, 'inference_files': len(manifest['files']),
                      'fixed_fusion': 'PASS', 'weights_checked': args.weights is not None,
                      'GPU_used': False}))


if __name__ == '__main__':
    main()
