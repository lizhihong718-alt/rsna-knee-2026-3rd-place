"""Invoke the preserved winning inference runtime with explicit input/output paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='competition directory containing test.csv')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ENSEMBLE_DATA_ROOT=str(args.data.resolve()),
               ENSEMBLE_WEIGHT_ROOTS=str(args.weights.resolve()), ENSEMBLE_OUTPUT=str(output))
    subprocess.run([sys.executable, str(root / 'inference/run.py')], env=env, check=True)


if __name__ == '__main__':
    main()
