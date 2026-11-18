"""Offline runtime checks; no downloads, model loading, or inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(root, hashes=False, environment=False):
    rows = []
    for filename in ('config/weight_collection.json', 'config/runtime_assets.json'):
        manifest = json.loads((root / filename).read_text(encoding='utf8'))
        for item in manifest['rows']:
            path = root / item['path']
            status = 'OK' if path.is_file() else 'MISSING'
            if status == 'OK' and path.stat().st_size != item['bytes']:
                status = 'SIZE_MISMATCH'
            if status == 'OK' and hashes:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 2**20), b''):
                        digest.update(block)
                if digest.hexdigest() != item['sha256']:
                    status = 'HASH_MISMATCH'
            rows.append({'path': item['path'], 'status': status})
    env = []
    if environment:
        env.append({'name': 'python>=3.12,<3.13', 'status': 'OK' if sys.version_info[:2] == (3, 12) else 'UNVERIFIED', 'actual': sys.version.split()[0]})
        expected = json.loads((root / 'verification/runtime_versions.json').read_text('utf8'))
        for name, version in expected.items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = 'MISSING'
            env.append({'name': name, 'expected': version, 'actual': actual,
                        'status': 'OK' if actual == version else 'MISSING' if actual == 'MISSING' else 'UNVERIFIED'})
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        env.append({'name': 'CUDA', 'status': 'OK' if cuda else 'MISSING'})
    return {'success': all(r['status'] == 'OK' for r in rows + env),
            'hashes_checked': hashes, 'files': rows, 'environment': env,
            'scope': 'asset inventory and optional reference environment; not inference or legal clearance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--hashes', action='store_true', help='Read all ~9.2GB model bytes')
    parser.add_argument('--environment', action='store_true')
    parser.add_argument('--models', help='Model/component path configuration JSON')
    parser.add_argument('--inference', action='store_true', help='Only inference assets; excludes official validator and provenance metadata')
    parser.add_argument('--report', type=Path)
    sys.path.insert(0, str(ROOT/'code'))
    from temporal_segments import add_backend_arguments, backend_options, TemporalSegmenter
    add_backend_arguments(parser)
    args = parser.parse_args()
    backend_options(args, parser)
    if args.inference:
        from model_assets import check as inference_check, configuration
        result = inference_check(args.root.resolve(), configuration(args.models))
        result['environment'] = []
        try:
            backend = TemporalSegmenter(args.temporal_backend, args.legacy_kts_dir)
            backend_row = dict(name=args.temporal_backend, status='OK', implementation=backend.implementation)
        except (ImportError, OSError, ValueError) as exc:
            backend_row = dict(name=args.temporal_backend, status='UNAVAILABLE', error=str(exc))
        if args.environment:
            result['environment'] = check(args.root.resolve(), False, True)['environment']
        result['environment'].append(backend_row)
        result['success'] = result['success'] and all(r['status']=='OK' for r in result['environment'])
        if args.hashes:
            parser.error('--inference uses required-file/size checks; omit --inference for the historical full hash audit')
    else:
        result = check(args.root.resolve(), args.hashes, args.environment)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding='utf8')
    problems = [r for r in result['files'] + result['environment'] if r['status'] != 'OK']
    for row in problems[:20]:
        print(row['status'], row.get('path', row.get('name')))
    print(('PASS' if result['success'] else 'BLOCKED'), len(result['files']), 'assets;', len(problems), 'problems')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
