"""Lossless component loading for trusted, owner-authorized R66 checkpoints.

Splitting is an engineering operation, not a license grant. Phase 6 cleared
these independent project parameters for research preparation under the recorded
training-source notices; it did not authorize publication of this private repo.
Joblib is executable serialization: never load untrusted downloaded files.
"""
import hashlib
import json
from pathlib import Path

PARTS = {'listwise': ['model'], 'listwise_gate': ['gate'], 'parent_a2': ['parent'],
         'sequence_iou': ['sequence_transition'],
         'metadata': ['input_dim', 'config', 'iterations', 'sequence_parameters', 'parent_rank_sha256', 'sequence_transition_sha256']}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()


def inspect(path):
    path = Path(path)
    index = path/'checkpoint.json' if path.is_dir() else path
    record = json.loads(index.read_text(encoding='utf-8'))
    if record.get('format') != 'r66-components-v1': raise ValueError('Unsupported R66 component format')
    if set(record['parts']) != set(PARTS): raise ValueError('Missing R66 components')
    for name, part in record['parts'].items():
        file = (index.parent/part['file']).resolve()
        if not file.is_relative_to(index.parent.resolve()): raise ValueError('Component escapes checkpoint directory')
        if not file.is_file() or sha(file) != part['sha256']: raise ValueError('Component hash mismatch: '+name)
    return index, record


def load(path):
    import joblib
    path = Path(path)
    if path.is_dir() or path.suffix == '.json':
        index, record = inspect(path)
        bundle = {}
        for name, part in record['parts'].items():
            values = joblib.load(index.parent/part['file'])
            if set(values) != set(PARTS[name]): raise ValueError('Unexpected component schema: '+name)
            bundle.update(values)
    else:
        bundle = joblib.load(path)
    if bundle['input_dim'] != 405 or bundle['sequence_parameters'] != 52:
        raise ValueError('Expected frozen R66 405-dimensional ranker with 52-parameter transition')
    return bundle


def split(source, destination):
    import joblib
    source, destination = Path(source), Path(destination)
    bundle = load(source)
    expected = set().union(*map(set, PARTS.values()))
    if set(bundle) != expected: raise ValueError('Unexpected checkpoint keys; refusing lossy split')
    destination.mkdir(parents=True, exist_ok=False)
    index = dict(format='r66-components-v1', original_sha256=sha(source),
                 redistribution='RESEARCH_PARAMETERS_PHASE6_NOTICES_REQUIRED', parts={})
    for name, keys in PARTS.items():
        file = destination/(name+'.joblib')
        joblib.dump({key:bundle[key] for key in keys}, file, compress=3)
        index['parts'][name] = dict(file=file.name, sha256=sha(file), keys=keys,
                                  redistribution='RESEARCH_PARAMETERS_PHASE6_NOTICES_REQUIRED', reason='Retain research-source notices from docs/RESEARCH_RELEASE_PHASE6_20261113.md; no data/features or external model weights included')
    (destination/'checkpoint.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    inspect(destination)
    return index
