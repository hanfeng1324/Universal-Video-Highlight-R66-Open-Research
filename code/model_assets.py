"""Resolve actual inference assets independently of historical provenance files."""
import hashlib
import json
from pathlib import Path

OPTIONAL = {'README.md', 'LICENSE', 'model_manifest.json', 'source_metadata.json', 'PRETRAINED_SOURCE.json'}


def component(path):
    if '/models/qwen/' in path: return 'qwen'
    if '/clip-vit-base-patch32/' in path: return 'clip'
    if '/clip_tokenizer/' in path: return 'clip_tokenizer'
    if '/weights/dinov2-base/' in path: return 'dinov2'
    if '/weights/unisal/' in path: return 'unisal'
    return {'dsnet_tvsum.pt': 'dsnet', 'googlenet-1378be20.pth': 'googlenet',
            'uvcom.ckpt': 'uvcom', 'sam2.1_hiera_small.pt': 'sam2',
            'clip_cg_detr_qvhighlight.ckpt': 'cg_detr', 'weights.pth': 'gaic',
            'decision.joblib': 'r66', 'fcdb_ranker.joblib': 'fcdb',
            'dino_projection.pt': 'dino_projection'}.get(Path(path).name)


def configuration(filename=None):
    if not filename: return {}
    filename = Path(filename).resolve()
    config = json.loads(filename.read_text(encoding='utf-8-sig'))
    unknown = set(config) - {'models', 'paths', 'description'}
    if unknown: raise ValueError('Unknown model configuration keys: '+str(sorted(unknown)))
    for group in ('models', 'paths'):
        config.setdefault(group, {})
        for key, value in config[group].items():
            value = Path(value).expanduser()
            config[group][key] = str(value.resolve() if value.is_absolute() else (filename.parent/value).resolve())
    allowed={'qwen','clip','clip_tokenizer','dinov2','unisal','dsnet','googlenet','uvcom','sam2','cg_detr','gaic','r66','fcdb','dino_projection'}
    unknown=set(config['models'])-allowed
    if unknown: raise ValueError('Unknown model components: '+str(sorted(unknown)))
    return config


def resolve(root, relative, config):
    """Specific path/prefix overrides precede model-component overrides."""
    for prefix, value in sorted(config.get('paths', {}).items(), key=lambda x: -len(x[0])):
        if relative == prefix or relative.startswith(prefix.rstrip('/')+'/'):
            return Path(value)/relative[len(prefix):].lstrip('/')
    key = component(relative)
    if key in config.get('models', {}):
        base = Path(config['models'][key])
        return base/Path(relative).name if key in {'qwen','clip','clip_tokenizer','dinov2','unisal'} else base
    return root/relative


def model_rows(root):
    rows = json.loads((root/'config/weight_collection.json').read_text(encoding='utf-8'))['rows']
    return [r for r in rows if Path(r['path']).name not in OPTIONAL]


def check(root, config=None):
    config = config or {}
    checks = []
    for row in model_rows(root):
        path = resolve(root, row['path'], config)
        if component(row['path']) == 'r66' and (path.is_dir() or path.suffix == '.json'):
            from r66_checkpoint import inspect
            try:
                inspect(path)
                state = 'OK'
            except (OSError, ValueError, KeyError) as exc:
                state = 'INVALID_SPLIT_CHECKPOINT: '+str(exc)
        else:
            state = 'MISSING' if not path.is_file() else 'OK' if path.stat().st_size == row['bytes'] else 'SIZE_MISMATCH'
        checks.append(dict(component=component(row['path']), path=row['path'], resolved=str(path), status=state))
    support_index = root/'config/inference_support.json'
    if not support_index.is_file(): support_index = root/'config/runtime_assets.json'
    support = json.loads(support_index.read_text(encoding='utf-8'))['rows']
    for row in support:
        # The final scorer embeds its transition. Neither the separate training copy,
        # historical metadata nor the official validator belongs to this runtime.
        if '/validator/' in row['path'] or row['path'].endswith('sequence_transition.joblib'): continue
        path = resolve(root, row['path'], config)
        state = 'MISSING' if not path.is_file() else 'OK' if path.stat().st_size == row['bytes'] else 'SIZE_MISMATCH'
        checks.append(dict(component='implementation', path=row['path'], resolved=str(path), status=state))
    return dict(success=all(r['status']=='OK' for r in checks), files=checks,
                scope='Inference files only; no official validator or training provenance requirements. Size checks do not grant distribution rights.')
