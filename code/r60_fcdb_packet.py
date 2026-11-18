"""Build a public true-human-crop packet from the author's FCDB archive.

No networking, formal inputs, label warping, or test-to-train conversion.
Plan is frozen before new image pixels or crop metrics are accessed.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jpeg_size(data):
    assert data[:2] == b'\xff\xd8', 'not JPEG'
    p = 2
    while p < len(data):
        assert data[p] == 255
        while data[p] == 255:
            p += 1
        marker = data[p]
        p += 1
        if marker in [0x01, *range(0xD0, 0xDA)]:
            continue
        size = struct.unpack_from('>H', data, p)[0]
        if marker in [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF]:
            height, width = struct.unpack_from('>HH', data, p + 3)
            return width, height
        p += size
    raise ValueError('JPEG dimensions not found')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata-dir', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    meta = pathlib.Path(args.metadata_dir)
    out = pathlib.Path(args.output_dir)
    assert not out.exists(), 'use a new output directory; never overwrite'
    plan_path = meta / 'SOURCE_PLAN.json'
    plan = json.loads(plan_path.read_text('utf8'))
    archive = pathlib.Path(args.archive)
    assert archive.stat().st_size == plan['archive_bytes']
    archive_sha = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            archive_sha.update(block)
    rows, excluded = [], []
    with zipfile.ZipFile(archive) as z:
        byname = {}
        for entry in z.infolist():
            name = pathlib.PurePosixPath(entry.filename).name
            if name.lower().endswith('.jpg'):
                assert name not in byname, 'ambiguous duplicate archive basename'
                byname[name] = entry
        # Test and old exposed sources only produce SHA exclusions, never labels.
        reserved = set()
        for r in plan['reserved_sources']:
            if r['archive_basename'] in byname:
                reserved.add(sha(z.read(byname[r['archive_basename']])))
        hashes = {}
        for r in plan['eligible_official_train_sources']:
            data = z.read(byname[r['archive_basename']])
            digest = sha(data)
            width, height = jpeg_size(data)
            x, y, cw, ch = r['human_crop_xywh']
            reason = None
            if not (min(x, y) >= 0 and cw > 0 and ch > 0 and
                    x + cw <= width and y + ch <= height):
                reason = 'HUMAN_CROP_OUT_OF_BOUNDS_NO_LABEL_CORRECTION'
            elif digest in reserved:
                reason = 'EXACT_MEDIA_DUPLICATE_OF_OFFICIAL_TEST_OR_OLD_EXPOSED'
            elif digest in hashes:
                reason = 'EXACT_MEDIA_DUPLICATE_OF_OTHER_ELIGIBLE_SOURCE'
            if reason:
                excluded.append(dict(source_id=r['source_id'], reason=reason))
                continue
            hashes[digest] = r['source_id']
            path = pathlib.Path('images') / (r['source_id'] + '.jpg')
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).write_bytes(data)
            rows.append(dict(r, media=path.as_posix(), media_sha256=digest,
                             width=width, height=height,
                             label_type='AUTHOR_TRUE_HUMAN_NATIVE_CROP',
                             aic_target_ratio_truth=False))
    out.mkdir(parents=True, exist_ok=True)
    report = dict(rows=rows, quarantine=excluded,
                  plan_sha256=sha(plan_path.read_bytes()),
                  archive_sha256=archive_sha.hexdigest(),
                  script_sha256=sha(pathlib.Path(__file__).read_bytes()),
                  formal_inputs=0, labels_resized=False, training_executed=False,
                  old_splits_modified=False, original_official_test_used_for_training=False,
                  remaining='Perceptual cross-upload aliases and uploader families unverified; exact SHA exclusion alone does not certify full source independence.')
    (out / 'PUBLIC_MANIFEST.json').write_text(
        json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(dict(rows=len(rows), quarantined=len(excluded),
                          roles={k: sum(r['role'] == k for r in rows)
                                 for k in ['train', 'dev']})))


if __name__ == '__main__':
    main()
