"""R60 authentic FCDB human-IoU features, public inputs only.

Native human aspect ratio conditions a fixed generic crop lattice. Human
position/extent never seed proposals; xywh is used only for exclusive IoU.
No target-ratio rewriting, MOS supervision, or formal-media access.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import zipfile


def read(p):
    return json.loads(Path(p).read_text('utf8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), 'utf8')
    tmp.replace(p)


def download(url, dest, expected):
    if dest.exists():
        assert dest.stat().st_size == expected
        return
    part = dest.with_suffix('.zip.part')
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        assert offset <= expected
        if offset == expected:
            part.replace(dest)
            return
        req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                assert response.headers.get('content-type', '').split(';')[0] != 'text/html'
                if offset:
                    assert response.status == 206, 'server refused resumable range'
                    assert response.headers['content-range'].startswith(f'bytes {offset}-')
                with part.open('ab' if offset else 'wb') as f:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        assert f.tell() + len(block) <= expected
                        f.write(block)
                    f.flush()
                    os.fsync(f.fileno())
            assert part.stat().st_size == expected
            part.replace(dest)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(20, 2 ** attempt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--old-root', type=Path, default=Path('/root/autodl-fs/r59_20261021'))
    ap.add_argument('--metadata-dir', type=Path)
    ap.add_argument('--limit', type=int, default=0, help='smoke only; no DONE on subset')
    args = ap.parse_args()
    root = args.root
    meta = args.metadata_dir or root / 'FCDB_METADATA'
    plan = read(meta / 'SOURCE_PLAN.json')
    assert plan['formal_inputs'] is False
    cfg = read(meta / 'AUTHOR_ARCHIVE_METADATA.json')
    archive = root / 'public_data/FCDB.zip'
    download(cfg['url'], archive, cfg['content_length'])
    sys.path.insert(0, str(args.old_root))
    import cv2
    import numpy as np
    import torch
    import r59_gaic as gaic
    from r59_model import proposals, legal, descriptors
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gaic.Ranker().eval().to(device).requires_grad_(False)
    weights = args.old_root / 'data/gaic/weights.pth'
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    seals = dict(plan_sha256=sha(meta / 'SOURCE_PLAN.json'),
                 script_sha256=sha(__file__), gaic_weight_sha256=sha(weights),
                 archive_sha256=sha(archive), device=str(device), feature_dim=1327,
                 saliency='UNIFORM_UNAVAILABLE_NOT_UNISAL', exclusive_iou=True,
                 native_human_ratio_only=True, formal_inputs=False)
    sealpath = root / 'FCDB_FEATURES_SEAL.json'
    if sealpath.exists():
        # Device does not alter source identity; numerical parity is not assumed.
        old = read(sealpath)
        assert {k:v for k,v in old.items() if k != 'device'} == {k:v for k,v in seals.items() if k != 'device'}
    else:
        save(sealpath, seals)
    out = root / 'features_fcdb'
    out.mkdir(parents=True, exist_ok=True)
    rows, quarantine = [], []
    started = time.time()
    with zipfile.ZipFile(archive) as z:
        byname = {}
        for entry in z.infolist():
            name = Path(entry.filename).name
            if name.lower().endswith('.jpg'):
                assert name not in byname
                byname[name] = entry

        def image(row):
            data = z.read(byname[row['archive_basename']])
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert im is not None
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), hashlib.sha256(data).hexdigest()

        def phash(rgb):
            a = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            v = cv2.dct(cv2.resize(a, (32, 32)).astype(np.float32))[:8, :8].flatten()[1:]
            return v > np.median(v)

        oldph, oldsha = [], set()
        for row in plan['old_exposure_sources']:
            if row['archive_basename'] not in byname:
                continue
            rgb, digest = image(row)
            oldph.append(phash(rgb))
            oldsha.add(digest)
        oldph = np.array(oldph)
        # All official test images protect train/dev from exact/near duplication,
        # independently of whether test was selected for one-time evaluation.
        testph, testsha = [], set()
        for row in plan['all_official_test_sources']:
            rgb, digest = image(row)
            testph.append(phash(rgb))
            testsha.add(digest)
        testph = np.array(testph)
        seen, seenph = set(), []
        selected = plan['eligible_official_train_sources'] + plan['selected_test_sources']
        if args.limit:
            selected = selected[:args.limit]
        for number, row in enumerate(selected):
            ident = row['source_id']
            rgb, digest = image(row)
            h, w = rgb.shape[:2]
            x, y, cw, ch = row['human_crop_xywh']
            reason = None
            ph = phash(rgb)
            if not (min(x, y) >= 0 and min(cw, ch) > 0 and x + cw <= w and y + ch <= h):
                reason = 'INVALID_NATIVE_HUMAN_COORDINATES_NO_CORRECTION'
            elif digest in oldsha or (len(oldph) and np.min(np.count_nonzero(oldph != ph, axis=1)) <= 4):
                reason = 'OLD_EXPOSED_FINAL_EXACT_OR_PHASH_LE4'
            elif row['role'] != 'test' and (digest in testsha or np.min(np.count_nonzero(testph != ph, axis=1)) <= 4):
                reason = 'OFFICIAL_TEST_EXACT_OR_PHASH_LE4'
            elif digest in seen or (seenph and min(np.count_nonzero(p != ph) for p in seenph) <= 4):
                reason = 'INTRA_PACKET_EXACT_OR_PHASH_LE4_NO_REPLACEMENT'
            if reason:
                quarantine.append(dict(source=ident, role=row['role'], reason=reason))
                continue
            seen.add(digest)
            seenph.append(ph)
            dest = out / (ident + '.npz')
            if not dest.exists():
                ratio = [cw, ch]  # ONLY human native aspect, not its origin or scale.
                boxes = proposals(w, h, ratio)
                base = legal(w / 2, h / 2, min(w, h * cw / ch), ratio, w, h)
                baseline = int(np.square(boxes - base).sum(-1).argmin())
                assert np.array_equal(boxes[baseline], base)
                gt = np.array([[x, y, x + cw, y + ch]], np.float32)
                inter = np.maximum(0, np.minimum(boxes[:, 2:], gt[:, 2:]) - np.maximum(boxes[:, :2], gt[:, :2])).prod(-1)
                labels = inter / np.maximum((boxes[:, 2:] - boxes[:, :2]).prod(-1) + cw * ch - inter, 1e-9)
                s = 256 / min(h, w)
                nh, nw = [round(d * s / 32) * 32 for d in [h, w]]
                a = cv2.resize(rgb, (nw, nh)).astype(np.float32) / 256
                a = (a - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
                inp = torch.from_numpy(a.transpose(2, 0, 1).copy())[None].to(device)
                scaled = boxes.astype(float) * [nw / w, nh / h, nw / w, nh / h]
                scaled[:, :2] = np.floor(scaled[:, :2])
                scaled[:, 2:] = np.ceil(scaled[:, 2:])
                bt = torch.tensor(np.c_[np.zeros(len(boxes)), scaled], device=device, dtype=torch.float32)
                values, scores = [], []
                with torch.inference_mode():
                    f = model.encode(inp)
                    for i in range(0, len(bt), 48):
                        v = torch.cat((gaic.align(f, bt[i:i+48]), gaic.align(f, bt[i:i+48], True)), 1)
                        values.append(v.flatten(1).cpu().numpy())
                        scores.append(model.FC_layers(v).flatten().cpu().numpy())
                scores = np.concatenate(scores)
                geometry = descriptors(boxes, np.ones((32, 32), np.float32), w, h, scores)
                geometry[:, 20] = 0  # no additional semantics
                geometry[:, 21] = 0  # explicit saliency unavailable
                features = np.c_[np.concatenate(values), geometry].astype(np.float16)
                assert features.shape == (len(boxes), 1327)
                tmp = dest.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, x=features, y=labels.astype(np.float32), boxes=boxes,
                                    gt=gt, human_crop_xywh=np.array(row['human_crop_xywh']),
                                    gaic=scores, baseline=baseline, width=w, height=h,
                                    lengths=[len(boxes)], source=ident, role=row['role'],
                                    saliency_available=False, exclusive_iou=True,
                                    native_ratio=np.array(ratio), image_sha256=digest)
                tmp.replace(dest)
            else:
                cached = np.load(dest)
                assert str(cached['image_sha256']) == digest
                assert str(cached['role']) == row['role']
            rows.append(dict(file=str(dest.relative_to(root)), source='fcdb_' + ident,
                             role=row['role'], kind='human_iou', sha256=sha(dest),
                             label_type='AUTHOR_TRUE_HUMAN_NATIVE_CROP', saliency_available=False))
            if number % 10 == 0:
                save(root / 'FCDB_FEATURE_PROGRESS.json', dict(completed=len(rows), quarantined=len(quarantine), index=number, total=len(selected), elapsed=time.time()-started, device=str(device)))
                print('FCDB_FEATURE', number, len(rows), len(quarantine), flush=True)
    report = dict(rows=rows, quarantine=quarantine, seals=seals, formal_inputs=False,
                  old_exposed_sources_excluded=True, test_once_after_freeze=True,
                  smoke_limit=args.limit, labels='EXCLUSIVE_IOU_ONLY_NO_MOS',
                  elapsed=time.time()-started)
    save(root / 'FEATURES_FCDB.json', report)
    if not args.limit:
        save(root / 'FEATURES_FCDB_DONE.json', dict(success=True, rows=len(rows), quarantined=len(quarantine), roles={k:sum(r['role']==k for r in rows) for k in ['train','dev','test']}, manifest_sha256=sha(root/'FEATURES_FCDB.json')))


if __name__ == '__main__':
    main()
