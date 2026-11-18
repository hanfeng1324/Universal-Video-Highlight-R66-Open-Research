"""R63 label-free crop expansion, real GAIC features and frozen R60 scoring."""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_name, '4')
from pathlib import Path
import sys, json, hashlib
sys.path[:0] = ['/root/autodl-fs/r60_20261022', '/root/autodl-fs/r59_20261021']
import numpy as np
from r59_model import legal, frozen_features
from r60_fcdb_train import features

TREE = Path('/root/autodl-fs/r60_20261022/fcdb_ranker.joblib')
TREE_SHA = '65fa2a9632b973eb5f8c23941f430050ec2be78829df3b1a0561ffdad4bf09f7'
CONFIG = dict(version='R63-A-CROP-SEARCH-V1', seed_count=4,
              local_center_offsets=[-.08, -.04, 0., .04, .08],
              local_width_factors=[.90, 1., 1.10],
              global_width_fractions=[.40, .475, .625, .775, .93],
              region_padding=[1.05, 1.25, 1.60],
              search_rounds=1, unchanged_tree=True, feature_dim=156,
              path='original .18 center / .035 log-size; exact original optimizer',
              labels_used_in_generation=False, original_pool_preserved=True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_tree():
    import joblib
    assert sha(TREE) == TREE_SHA
    tree = joblib.load(TREE); tree.n_jobs = 2
    assert len(tree.estimators_) == 96
    return tree

def score(tree, raw):
    x = features(np.asarray(raw, np.float32)); assert x.shape[1] == 156
    out = tree.predict(x).astype(np.float32)
    assert np.isfinite(out).all()
    return out

def image_regions(rgb, sm):
    """Image-driven region hypotheses; these are not interaction ground truth."""
    import cv2
    h, w = rgb.shape[:2]; regions = []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh >= .005 * w * h:
            regions.append([x, y, x+bw, y+bh])
    m = np.maximum(np.asarray(sm, float), 0)
    if m.max() > m.min() and m.sum() > 0:
        flat = m.flatten(); ids = np.argsort(flat)[::-1]
        for fraction in [.5, .8]:
            chosen = ids[:max(1, int(np.searchsorted(np.cumsum(flat[ids]), fraction*flat.sum()))+1)]
            yy, xx = np.unravel_index(chosen, m.shape)
            regions.append([xx.min()/m.shape[1]*w, yy.min()/m.shape[0]*h,
                            (xx.max()+1)/m.shape[1]*w, (yy.max()+1)/m.shape[0]*h])
    return regions

def expand(boxes, utilities, w, h, ratio, regions=(), inherited=(), previous=()):
    """Generate proposals without RGB labels, annotation boxes or video IDs."""
    bb = np.asarray(boxes, np.int32); new = []; counts = {}
    seeds = []
    # Spatially distinct promising seeds prevent all local search from collapsing
    # onto nearly identical lattice cells. This limits compute, not output changes.
    for i in np.argsort(-np.asarray(utilities), kind='stable'):
        b = bb[i]; center = (b[:2]+b[2:])/2
        if all(np.linalg.norm((center-(q[:2]+q[2:])/2)/[w, h]) > .04
               or abs(np.log((b[2]-b[0])/(q[2]-q[0]))) > .10 for q in seeds):
            seeds.append(b)
        if len(seeds) == CONFIG['seed_count']: break
    for b in list(seeds)+list(inherited)+list(previous):
        b = np.asarray(b); cx, cy = (b[:2]+b[2:])/2; width = b[2]-b[0]
        for scale in CONFIG['local_width_factors']:
            for dx in CONFIG['local_center_offsets']:
                for dy in CONFIG['local_center_offsets']:
                    new.append(legal(cx+dx*w, cy+dy*h, width*scale, ratio, w, h))
    counts['local_and_temporal'] = len(new)
    mx = min(w, int(h*ratio[0]//ratio[1]))
    for scale in CONFIG['global_width_fractions']:
        cw = max(2, round(mx*scale)); ch = int(np.ceil(cw*ratio[1]/ratio[0]))
        for cx in np.linspace(cw/2, w-cw/2, 7 if cw < w else 1):
            for cy in np.linspace(ch/2, h-ch/2, 5 if ch < h else 1):
                new.append(legal(cx, cy, cw, ratio, w, h))
    counts['multiscale'] = len(new)-sum(counts.values())
    rr = list(regions)
    # Pairwise unions are image/SAM object hypotheses, never asserted relations.
    for i in range(min(len(regions), 4)):
        for j in range(i+1, min(len(regions), 4)):
            a, b = np.asarray(regions[i]), np.asarray(regions[j])
            rr.append(np.r_[np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])])
    start = len(new)
    for b in rr:
        b = np.asarray(b); cx, cy = (b[:2]+b[2:])/2
        width = max(b[2]-b[0], (b[3]-b[1])*ratio[0]/ratio[1])
        for padding in CONFIG['region_padding']:
            new.append(legal(cx, cy, width*padding, ratio, w, h))
    new.extend(legal(*((np.asarray(b)[:2]+np.asarray(b)[2:])/2),
                     np.asarray(b)[2]-np.asarray(b)[0], ratio, w, h) for b in inherited)
    counts['regions_and_interaction'] = len(new)-start
    original = {tuple(b) for b in bb}
    result = np.asarray(sorted({tuple(b) for b in new}-original), np.int32).reshape(-1, 4)
    if len(result):
        assert (result[:, :2] >= 0).all() and (result[:, 2:] > result[:, :2]).all()
        assert (result[:, 2] <= w).all() and (result[:, 3] <= h).all()
    counts.update(unique_new=len(result), original=len(bb))
    return result, counts

def evaluate_pool(expert, tree, rgb, original, sm, raw=None, regions=(), inherited=(), previous=()):
    h, w = rgb.shape[:2]
    raise RuntimeError('Use search with explicit target ratio')

def search(expert, tree, rgb, ratio, original, sm, raw=None, regions=(), inherited=(), previous=()):
    h, w = rgb.shape[:2]
    if raw is None: raw, _ = frozen_features(expert, rgb, original, sm)
    old_scores = score(tree, raw)
    extra, counts = expand(original, old_scores, w, h, ratio,
                           list(regions)+image_regions(rgb, sm), inherited, previous)
    if len(extra):
        new_raw, new_gaic = frozen_features(expert, rgb, extra, sm)
        new_scores = score(tree, new_raw)
    else:
        new_raw = np.empty((0, raw.shape[1]), np.float32); new_scores = np.empty(0, np.float32)
    boxes = np.concatenate([original, extra]); utilities = np.r_[old_scores, new_scores]
    return boxes, utilities, np.concatenate([raw, new_raw]), counts

def exact_iou(boxes, gt):
    b = np.asarray(boxes, float); g = np.asarray(gt, float).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(b[:, None, 2:], g[None, :, 2:])-
                       np.maximum(b[:, None, :2], g[None, :, :2])).prod(-1)
    area = np.maximum(0, b[:, 2:]-b[:, :2]).prod(-1)[:, None]
    ga = np.maximum(0, g[:, 2:]-g[:, :2]).prod(-1)
    return (inter/np.maximum(area+ga-inter, 1e-9)).mean(-1).astype(np.float32)
