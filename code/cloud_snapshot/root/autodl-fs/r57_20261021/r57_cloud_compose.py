"""Independent public seven-capability prototype; no official submission.

Semantic query conditions CG/UVCOM proposals. Event-linked Qwen boxes select
SAM candidates; native-pixel LK tracks the selected region. Scene/composition
interpretation affects ratio crops. All evidence remains uncalibrated.
"""
import gc
import math
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SEMANTIC = BASE / 'semantic_v3'
from r57_cloud_visual import manifest, read, save, sha, receipt, CLONE
sys.path[:0] = [str(BASE / 'deps/dsnet'), str(CLONE / 'assets/r22_20261016/lighthouse')]
import av
import cv2
import numpy as np
import torch
from helpers.bbox_helper import nms
from r57_cloud_public_semantic import uv_predict


def query_predict(rows):
    import r22_lighthouse_runtime as original
    original.ENCODER = CLONE / 'aic_highlight/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    original.BIG = CLONE / 'assets/r22_20261016'
    original.WEIGHT = original.BIG / 'weights/clip_cg_detr_qvhighlight.ckpt'
    rt = original.Runtime()
    for r in rows:
        dest = BASE / 'visual' / r['id']
        plan = read(SEMANTIC / (r['id'] + '.json'))['plan']
        query = 'The complete visible event: ' + plan['event'] + ' Include necessary preparation, interaction and outcome.'
        if not (dest / 'CG_EVENT.json').exists():
            save(dest / 'CG_EVENT.json', rt.predict(np.load(dest / 'CLIP.npz')['features'], query))
    del rt.net
    gc.collect()
    torch.cuda.empty_cache()
    from lighthouse.common.uvcom import build_model
    ck = torch.load(BASE / 'weights/uvcom.ckpt', map_location='cpu', weights_only=False)
    opts = ck['opt']
    opts.device = 'cuda'
    model, _ = build_model(opts)
    model.load_state_dict(ck['model'], strict=True)
    model.cuda().eval().requires_grad_(False)
    del ck
    for r in rows:
        dest = BASE / 'visual' / r['id']
        cg = read(dest / 'CG_EVENT.json')
        txt, mask = rt.text(cg['query'])
        if not (dest / 'UV_EVENT.json').exists():
            save(dest / 'UV_EVENT.json', dict(uv_predict(model, txt, mask, np.load(dest / 'CLIP.npz')['features']), query=cg['query']))
    del rt, model
    gc.collect()
    torch.cuda.empty_cache()


def event_pool(folder, semantic, fraction):
    d = np.load(folder / 'DSNET.npz')
    clock = d['clock']
    n, cells = len(clock), len(d['scores'])
    scores, bb = nms(d['scores'], np.clip(d['boxes'], 0, cells).round().astype(int), .5)
    rows, coverage = {}, np.zeros(n, np.float32)

    def add(lo, hi, expert):
        lo, hi = max(0, int(lo)), min(n, int(hi))
        if lo >= hi:
            return
        rows.setdefault((lo, hi), dict(start=lo, end=hi, experts=[]))['experts'].append(expert)

    for s, (a, b) in zip(scores, bb):
        lo, hi = int(d['picks'][a]), int(d['picks'][b]) if b < cells else n
        coverage[lo:hi] = np.maximum(coverage[lo:hi], s)
        add(lo, hi, 'DSNet')
    native = dict(DSNet=coverage)
    for expert, name in [('CG_DETR', 'CG_EVENT.json'), ('UVCOM', 'UV_EVENT.json')]:
        pred = read(folder / name)
        for lo, hi, _ in pred['pred_relevant_windows']:
            add(np.searchsorted(clock, lo), np.searchsorted(clock, hi), expert)
        s = pred['pred_saliency_scores']
        native[expert] = np.asarray(s)[np.minimum((clock / 2).astype(int), len(s) - 1)]
    # Coarse semantic event is a proposal, never an all-other-frames DROP rule.
    a, b = semantic['plan']['event_samples']
    picks = semantic.get('event_sample_frames', semantic['sample_frames'])
    add(picks[a], picks[b + 1] if b + 1 < len(picks) else n, 'QWEN_STVG')
    candidates = list(rows.values())
    for expert, values in native.items():
        s = np.asarray([values[c['start']:c['end']].mean() for c in candidates])
        for c, v in zip(candidates, s):
            rank = (float((s < v).sum()) + .5 * float((s == v).sum())) / len(s)
            c.setdefault('expert_percentiles', {})[expert] = rank
    for c in candidates:
        p = list(c['expert_percentiles'].values())
        c['votes'] = sum(x > .5 for x in p)
        c['median_percentile'] = float(np.median(p))
        c['disagreement'] = float(max(p) - min(p))
    candidates.sort(key=lambda c: (-c['votes'], -c['median_percentile'], c['start'], c['end']))
    chosen = np.zeros(n, bool)
    selected = []
    for k, c in enumerate(candidates):
        if chosen.sum() >= math.ceil(n * fraction):
            break
        if chosen[c['start']:c['end']].all():
            continue
        chosen[c['start']:c['end']] = True
        selected.append(k)
    assert chosen.any()
    return np.flatnonzero(chosen), dict(candidates=candidates, selected=selected, fraction=float(chosen.mean()),
                                     diagnostic_fraction=fraction, formal_budget=False)


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(1e-9, np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - inter))


def flow(gray0, gray1, mask):
    points = cv2.goodFeaturesToTrack(gray0, maxCorners=100, qualityLevel=.01, minDistance=5, mask=mask)
    if points is None or len(points) < 4:
        return None
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None)
    back, bok, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, moved, None)
    valid = ok[:, 0].astype(bool) & bok[:, 0].astype(bool) & (np.linalg.norm((back - points)[:, 0], axis=1) < 2)
    if valid.sum() < 4:
        return None
    affine, _ = cv2.estimateAffinePartial2D(points[valid, 0], moved[valid, 0], method=cv2.RANSAC, ransacReprojThreshold=2)
    if affine is None:
        return None
    return cv2.warpAffine(mask, affine, (gray1.shape[1], gray1.shape[0]), flags=cv2.INTER_NEAREST)


def crop(center, mask, ratio, width, height, composition):
    tw, th = ratio
    cw, ch = min(width, height * tw // th), None
    ch = math.ceil(cw * th / tw)
    desired = np.asarray(center, float).copy()
    if composition == 'subject_left':
        desired[0] += .2 * cw
    elif composition == 'subject_right':
        desired[0] -= .2 * cw
    xy = desired - [cw / 2, ch / 2]
    # Project to contain the measured mask when it fits the legal ratio crop.
    if mask is not None and mask.any():
        yy, xx = np.nonzero(mask)
        lo = np.maximum([xx.max() + 1 - cw, yy.max() + 1 - ch], [0, 0])
        hi = np.minimum([xx.min(), yy.min()], [width - cw, height - ch])
        if (lo <= hi).all():
            xy = np.clip(xy, lo, hi)
    x, y = np.rint(np.clip(xy, [0, 0], [width - cw, height - ch])).astype(int)
    assert x >= 0 and y >= 0 and x + cw <= width and y + ch <= height
    return [int(x), int(y), int(cw)], ch


def compose(row, frames, semantic):
    folder = BASE / 'visual' / row['id']
    spatial = np.load(folder / 'SPATIAL.npz')
    sam = np.load(folder / 'SAM_PROPOSALS.npz')
    sam_masks, sam_boxes, sam_quality = sam['masks'], sam['boxes'], sam['qualities']
    clock = spatial['clock']
    plan = semantic['plan']
    anchor = {int(f): k for k, f in enumerate(sam['frames'])}
    cuts = set(spatial['cuts'].tolist())
    wanted = set(frames.tolist())
    out = BASE / 'composed' / row['id']
    out.mkdir(parents=True, exist_ok=True)
    ratios = [[16, 9], [9, 16]]
    paths = [out / 'PUBLIC_16x9.mp4', out / 'PUBLIC_9x16.mp4']
    fps = 1 / float(np.median(np.diff(clock)))
    sizes = [(640, 360), (360, 640)]
    writers = [cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), fps, size) for p, size in zip(paths, sizes)]
    assert all(w.isOpened() for w in writers)
    predictions = [[], []]
    counters = Counter()
    previous_gray, mask = None, None
    last_qwen = None
    try:
        with av.open(row['media']) as con:
            decoded = 0
            for f, frame in enumerate(con.decode(video=0)):
                rgb = frame.to_ndarray(format='rgb24')
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                decoded += 1
                if f in cuts:
                    mask = None
                    previous_gray = None
                nearest = int(np.abs(np.asarray(semantic['sample_frames']) - f).argmin())
                box = plan['focus_boxes'][nearest]
                qbox = np.asarray(box, float) / 1000 * [row['width'], row['height'], row['width'], row['height']] if box is not None else None
                if f in anchor:
                    k = anchor[f]
                    exact = f == semantic['sample_frames'][nearest]
                    selected_id = plan.get('selected_mask_indices', [None] * 8)[nearest]
                    if exact and selected_id is not None:
                        best = int(selected_id)
                        counters['SG_exact_frame_mask_ID_anchors'] += 1
                        counters['SG_STVG_conditioned_mask_anchors'] += 1
                    elif qbox is not None:
                        overlap = [iou(bb, qbox) for bb in sam_boxes[k]]
                        best = max(range(3), key=lambda i: (overlap[i], float(sam_quality[k, i])))
                        counters['SG_STVG_conditioned_mask_anchors'] += 1
                    else:
                        best = int(sam_quality[k].argmax())
                        counters['unknown_semantic_mask_anchors'] += 1
                    mask = sam_masks[k, best].astype(np.uint8)
                    counters['Seg_anchors'] += 1
                elif mask is not None and mask.any() and previous_gray is not None:
                    mask = flow(previous_gray, gray, mask)
                    counters['Tracking_success_frames' if mask is not None and mask.any() else 'Tracking_failed_frames'] += 1
                comp = plan.get('anchor_compositions', [plan['composition']] * 8)[nearest]
                if plan['intent'] == 'environment' or comp == 'whole_scene':
                    center = spatial['centers'][f]
                    use_mask = None
                    counters['SI_environment_frames'] += 1
                elif mask is not None and mask.any():
                    yy, xx = np.nonzero(mask)
                    center = np.array([xx.mean(), yy.mean()])
                    use_mask = mask
                    counters['mask_center_frames'] += 1
                elif qbox is not None:
                    center = (qbox[:2] + qbox[2:]) / 2
                    use_mask = None
                    counters['SG_uncertain_fallback_frames'] += 1
                else:
                    center = spatial['centers'][f]
                    use_mask = None
                    counters['UNISAL_fallback_frames'] += 1
                previous_gray = gray
                if f not in wanted:
                    continue
                for k, ratio in enumerate(ratios):
                    box3, h = crop(center, use_mask, ratio, row['width'], row['height'], comp)
                    x, y, w = box3
                    baseline, _ = crop(spatial['centers'][f], None, ratio, row['width'], row['height'], 'unknown')
                    counters['changed_from_cluster_frames_' + str(k)] += box3 != baseline
                    predictions[k].append(dict(frame=f, bboxes=box3))
                    bgr = cv2.cvtColor(rgb[y:y + h, x:x + w], cv2.COLOR_RGB2BGR)
                    writers[k].write(cv2.resize(bgr, sizes[k]))
        assert decoded == row['frames'] == len(clock)
    finally:
        for w in writers:
            w.release()
    videos = []
    for k, path in enumerate(paths):
        with av.open(str(path)) as con:
            n = sum(1 for _ in con.decode(video=0))
        assert n == len(frames) == len(predictions[k])
        videos.append(dict(path=str(path), frames=n, full_decode=True, sha256=sha(path)))
    records = [dict(video_id=row['video_id'], targetRatioWH=ratio, predictions=p) for ratio, p in zip(ratios, predictions)]
    save(out / 'PREDICTIONS.json', records)
    save(out / 'SPATIAL_EVIDENCE.json', dict(counters=dict(counters), semantic_or_tracking_GT=False,
        event_link='One VLM event and eight timestamp-bound focus boxes; same-event identity unverified',
        source_sha256=row['media_sha256'], native_full_decode=True))
    return dict(id=row['id'], counters=dict(counters), videos=videos, native_frames=decoded, selected_frames=len(frames))


def main():
    assert read(BASE / 'SEMANTIC_V3_DONE.json')['valid'] == 4
    assert (BASE / 'CANDIDATES_DONE.json').exists()
    started = time.time()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.cuda.reset_peak_memory_stats()
    rows = manifest()
    query_predict(rows)
    result = []
    # Explicit diagnostic only. Freeze a justified formal retention policy later.
    for r in rows:
        semantic = read(SEMANTIC / (r['id'] + '.json'))
        assert semantic['status'] == 'VALID' and semantic['source_sha256'] == r['media_sha256']
        folder = BASE / 'visual' / r['id']
        frames, pool = event_pool(folder, semantic, .5)
        save(folder / 'EVENT_POOL.json', pool)
        q = compose(r, frames, semantic)
        result.append(q)
        print('PUBLIC_SEVEN_COMPOSED', r['id'], q['selected_frames'], flush=True)
    counts = {}
    for name in ['TEMPORAL_DONE.json', 'SPATIAL_DONE.json', 'SEMANTIC_V3_DONE.json', 'CANDIDATES_DONE.json']:
        for k, v in read(BASE / name)['parameter_counts'].items():
            assert k not in counts
            counts[k] = v
    total = sum(counts.values())
    assert total <= 9_000_000_000
    tracking = sum(r['counters'].get('Tracking_success_frames', 0) for r in result)
    grounding = sum(r['counters'].get('SG_STVG_conditioned_mask_anchors', 0) for r in result)
    assert tracking > 0 and grounding > 0
    q = receipt('EVENT_CONDITIONED_TG_AND_COMPOSITION', {}, started)
    q.update(rows=result, total_parameters=total, components=counts, status='PUBLIC_FUNCTIONAL_SEVEN_PROTOTYPE',
        capability_evidence=dict(TG='Actual DSNet and VQA-query-conditioned CG/UVCOM event decisions',
            SG='Qwen choices of numbered exact-frame SAM candidates; no free coordinate generation', Seg='Original SAM2 image multimasks',
            Tracking='Native-current-pixel LK forward/backward affine mask propagation',
            STVG='VLM event sample interval and eight event-linked spatial regions; identity quality unverified',
            VQA='Actual frozen Qwen video event/participant answers condition temporal query and spatial selection',
            SI='VLM scene/composition intent selects environment or subject and left/center/right target placement'),
        training=False, distillation=False, formal426=False, formal_budget=None, official_score=None,
        limitations=['All four sources previously exposed and possible UNISAL DHF pretraining overlap',
                     'Functional execution and schema only, no independent event/crop quality advantage',
                     'No complete-member or identity GT; sparse VLM localization and LK can drift',
                     'Maximal-width composition remains a structural baseline, not optimal crop scale',
                     'Formal licenses/dependency hashes and local resource execution still need final audit'],
        R50_or_Student_decisions_used=False, media_uploaded='PUBLIC_ONLY', automatic_submit=False)
    save(BASE / 'COMPOSE_DONE.json', q)
    print('PUBLIC_SEVEN_DONE', total, 'parameters', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(BASE / 'COMPOSE_FAILED.json', dict(error=repr(e)))
        raise
