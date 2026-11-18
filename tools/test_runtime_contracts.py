"""Focused release integration checks; no training, GPU run or official data."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from model_assets import check, configuration
from r66_checkpoint import inspect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split-checkpoint',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sparse=out/'no_assets';(sparse/'config').mkdir(parents=True)
    for name in ['weight_collection.json','runtime_assets.json']:
        shutil.copyfile(ROOT/'config'/name,sparse/'config'/name)
    missing=check(sparse)
    assert not missing['success'] and all(r['status']=='MISSING' for r in missing['files'])
    assert not any('/validator/' in r['path'] or r['path'].endswith('sequence_transition.joblib') or r['path'].endswith(('model_manifest.json','source_metadata.json','PRETRAINED_SOURCE.json')) for r in missing['files'])
    bad=out/'bad-models.json';bad.write_text(json.dumps({'models':{'typo_qwen':'not-real'}}))
    try: configuration(bad)
    except ValueError: pass
    else: raise AssertionError('Unknown model key was silently accepted')
    index,manifest=inspect(args.split_checkpoint)
    broken=out/'corrupt_split';broken.mkdir()
    import os
    for part in manifest['parts'].values():os.link(index.parent/part['file'],broken/part['file'])
    manifest['parts']['listwise_gate']['sha256']='0'*64
    (broken/'checkpoint.json').write_text(json.dumps(manifest))
    try:inspect(broken)
    except ValueError as exc: assert 'hash mismatch' in str(exc)
    else:raise AssertionError('Corrupted component index was accepted')
    (out/'RESULT.json').write_text(json.dumps(dict(success=True,missing_required_assets_rejected=True,official_validator_not_required=True,training_metadata_not_required=True,unknown_component_rejected=True,tampered_split_manifest_rejected=True,functional_inference_rerun=False),indent=2))
    print('PASS: missing assets, scope exclusions, bad config and tampered checkpoint checks')


if __name__=='__main__':main()
