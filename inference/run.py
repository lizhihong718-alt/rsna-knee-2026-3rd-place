"""Offline entry of the Best29W3 column-routed probability-mean ensemble bundle."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('bundle integrity failed: ' + relative)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    for wheel in sorted((bundle / 'wheels').glob('*.whl')):
        name, version = wheel.name.split('-')[:2]
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)], check=True)
    if not os.environ.get('ENSEMBLE_DATA_ROOT'):
        data = next((p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                                 Path('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'test.csv').exists()), None)
        if data is None:
            raise FileNotFoundError('competition input not found')
        os.environ['ENSEMBLE_DATA_ROOT'] = str(data)
    os.environ.setdefault('ENSEMBLE_OUTPUT', '/kaggle/working/submission.csv')
    runpy.run_path(str(bundle / 'ensemble' / 'ensemble_main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
