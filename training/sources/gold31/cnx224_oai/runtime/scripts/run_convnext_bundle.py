"""Offline entrypoint copied into the root of a single-checkpoint bundle."""
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
    if os.environ.get('RESNET_DATA_ROOT'):
        data = Path(os.environ['RESNET_DATA_ROOT'])
    else:
        data = next((p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                                Path('/kaggle/input/rsna-knee-abnormality-detection'))
                     if (p / 'test.csv').exists()), None)
    if data is None:
        raise FileNotFoundError('competition input not found')
    sys.path[:0] = [str(bundle / 'scripts'), str(bundle / 'src')]
    sys.argv = ['inference_convnext_fullstrength_dicom.py', '--checkpoint', str(bundle / 'model.pt'),
                '--training-manifest', str(bundle / 'training_manifest.json'),
                '--preprocess-config', str(bundle / 'preprocess.yaml'), '--data-root', str(data),
                '--out', os.environ.get('RESNET_OUTPUT', '/kaggle/working/submission.csv'),
                '--encode-chunk', os.environ.get('RESNET_ENCODE_CHUNK', '16')]
    for env, flag in (('RESNET_STUDY_BATCH_SIZE', '--study-batch-size'),
                      ('RESNET_CUDNN_BENCHMARK', '--cudnn-benchmark'),
                      ('RESNET_EVAL_AMP', '--eval-amp')):
        if os.environ.get(env):
            sys.argv.extend([flag, os.environ[env]])
    runpy.run_path(str(bundle / 'scripts/inference_convnext_fullstrength_dicom.py'), run_name='__main__')


if __name__ == '__main__':
    main()
