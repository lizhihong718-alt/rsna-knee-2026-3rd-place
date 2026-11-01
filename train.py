"""Run one original recipe after resolving separately acquired private inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--paths', type=Path, required=True,
                        help='JSON object mapping asset keys in training/input_assets.json to local paths')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    recipes = json.loads((ROOT / 'training/recipes.json').read_text())
    recipe = next((r for r in recipes if r['id'] == args.id), None)
    if recipe is None:
        parser.error('unknown model ID')
    paths = json.loads(args.paths.read_text())
    assets = json.loads((ROOT / 'training/input_assets.json').read_text())
    out = args.out.resolve()
    checked = set()

    def sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 << 20), b''):
                digest.update(block)
        return digest.hexdigest()

    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str) and value.startswith('@OUT@'):
            return str(out) + value[len('@OUT@'):]
        if isinstance(value, str) and value.startswith('assets/'):
            key = value.split('/', 1)[1]
            if assets.get(key, {}).get('kind') == 'historical_checkpoint':
                # The original portable_refit_oai adapter uses this field as lineage only.
                return 'historical-sha256:' + assets[key]['sha256']
            if key not in paths:
                raise ValueError('input path mapping missing: ' + key)
            path = Path(paths[key]).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError('input asset unavailable: ' + key)
            if key not in checked:
                meta = assets.get(key, {})
                if path.is_file() and meta.get('sha256') and sha256(path) != meta['sha256']:
                    raise ValueError('input file SHA256 differs: ' + key)
                if path.is_dir() and meta.get('manifest_sha256'):
                    manifest = path / 'manifest.json'
                    if not manifest.is_file() or sha256(manifest) != meta['manifest_sha256']:
                        raise ValueError('input cache manifest SHA256 differs: ' + key)
                checked.add(key)
            return str(path)
        return value

    config = resolve(json.loads((ROOT / 'training' / recipe['config']).read_text()))
    if '${' in json.dumps(config):
        raise ValueError('unresolved original asset placeholder')
    out.mkdir(parents=True, exist_ok=False)
    import yaml
    config_path = out / 'requested_config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (out / 'recipe.json').write_text(json.dumps(recipe, indent=2))
    if args.prepare_only:
        print(json.dumps({'id': args.id, 'config_prepared': True, 'training_started': False}))
        return
    entry = ROOT / 'training' / recipe['entry']
    runtime = entry.parents[1]
    sys.path[:0] = [str(runtime / 'src'), str(runtime / 'scripts'), str(runtime.parent)]
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    sys.argv = [str(entry), '--config', str(config_path), '--out', str(out)]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
