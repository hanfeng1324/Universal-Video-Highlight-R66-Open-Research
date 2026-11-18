"""Fresh raw-public-input DSNet/CLIP/CG/UVCOM and UNISAL/cluster stages.

Runs only on the explicitly supplied persistent public manifest. Each stage
has independent atomic receipts. No training, formal media or paid APIs.
"""
import ast
import gc
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
BASE = Path(__file__).resolve().parent
CLONE = Path('/root/r53_autonomous_20261020')
sys.path[:0] = [str(BASE / 'deps_extra'), str(BASE / 'deps/dsnet'),
                str(CLONE / 'assets/r22_20261016/lighthouse')]
import av
import cv2
import numpy as np
import torch


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, q):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(q, indent=2, ensure_ascii=False))
    tmp.replace(p)


def functions(path, names, env):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def manifest():
    cfg = read(BASE / 'PUBLIC_INPUTS.json')
    assert cfg['data_role'] == 'PUBLIC_EXPOSED_ENGINEERING_ONLY' and not cfg['formal_inputs']
    rows = cfg['rows']
    assert len(rows) == 4
    for r in rows:
        assert r['role'].startswith('PUBLIC') and r['media'].startswith('public/')
        p = (BASE / r['media']).resolve()
        assert p.is_relative_to((BASE / 'public').resolve()) and sha(p) == r['media_sha256']
        r['media'] = str(p)
    return rows


def receipt(stage, counts, started):
    torch.cuda.synchronize()
    return dict(stage=stage, parameter_counts=counts, seconds=time.time() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                source_sha256=sha(Path(sys.argv[0]).resolve()),
                receipt_helper_sha256=sha(__file__), formal_inputs=False)


def temporal(rows):
    from anchor_free.dsnet_af import DSNetAF
    from helpers.video_helper import FeatureExtractor
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    cachedir = BASE / 'torchhub/checkpoints'
    cachedir.mkdir(parents=True, exist_ok=True)
    link = cachedir / 'googlenet-1378be20.pth'
    if not link.exists():
        link.symlink_to(BASE / 'weights/googlenet-1378be20.pth')
    torch.hub.set_dir(str(BASE / 'torchhub'))
    m = DSNetAF('attention', 1024, 128, 8).cuda().eval()
    m.load_state_dict(torch.load(BASE / 'weights/dsnet_tvsum.pt', map_location='cpu', weights_only=True), strict=True)
    extractor = FeatureExtractor()
    counts = dict(DSNet=sum(p.numel() for p in m.parameters()), GoogLeNet=sum(p.numel() for p in extractor.model.parameters()))
    for r in rows:
        out = BASE / 'visual' / r['id']
        out.mkdir(parents=True, exist_ok=True)
        dest = out / 'DSNET.npz'
        if dest.exists():
            assert read(out / 'DSNET.json')['source_sha256'] == r['media_sha256']
            continue
        clock, features, picks = [], [], []
        with av.open(r['media']) as con:
            for i, frame in enumerate(con.decode(video=0)):
                assert [frame.width, frame.height] == [r['width'], r['height']]
                clock.append(float(frame.time))
                if i % 15 == 0:
                    features.append(extractor.run(frame.to_ndarray(format='rgb24')))
                    picks.append(i)
        clock = np.asarray(clock) - clock[0]
        assert len(clock) == r['frames'] and (np.diff(clock) > 0).all()
        with torch.inference_mode():
            scores, boxes = m.predict(torch.tensor(np.asarray(features, np.float32), device='cuda')[None])
        assert np.isfinite(scores).all() and np.isfinite(boxes).all()
        np.savez_compressed(dest, clock=clock, features=features, picks=picks, scores=scores, boxes=boxes)
        save(out / 'DSNET.json', dict(source_sha256=r['media_sha256'], cache_sha256=sha(dest), full_native_decode=True))
        print('DSNET_FRESH', r['id'], len(clock), flush=True)
    del m, extractor
    gc.collect()
    torch.cuda.empty_cache()
    import r22_lighthouse_runtime as original
    original.ENCODER = CLONE / 'aic_highlight/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    original.BIG = CLONE / 'assets/r22_20261016'
    original.WEIGHT = original.BIG / 'weights/clip_cg_detr_qvhighlight.ckpt'
    rt = original.Runtime()
    counts.update(CLIP=sum(p.numel() for p in rt.clip.parameters()), CG_DETR=sum(p.numel() for p in rt.net.parameters()))
    for r in rows:
        dest = BASE / 'visual' / r['id']
        features, _, meta = rt.features(Path(r['media']), dest / 'CLIP.npz')
        assert meta['source_sha256'] == r['media_sha256']
        if not (dest / 'CG_GENERIC.json').exists():
            save(dest / 'CG_GENERIC.json', rt.predict(features, original.GENERIC_QUERIES[0]))
    del rt.net
    gc.collect()
    torch.cuda.empty_cache()
    from lighthouse.common.uvcom import build_model
    ck = torch.load(BASE / 'weights/uvcom.ckpt', map_location='cpu', weights_only=False)
    opts = ck['opt']
    opts.device = 'cuda'
    uv, _ = build_model(opts)
    uv.load_state_dict(ck['model'], strict=True)
    uv.cuda().eval().requires_grad_(False)
    counts['UVCOM'] = sum(p.numel() for p in uv.parameters())
    del ck
    from r57_cloud_public_semantic import uv_predict
    txt, mask = rt.text(original.GENERIC_QUERIES[0])
    for r in rows:
        dest = BASE / 'visual' / r['id']
        if not (dest / 'UV_GENERIC.json').exists():
            save(dest / 'UV_GENERIC.json', uv_predict(uv, txt, mask, np.load(dest / 'CLIP.npz')['features']))
        print('TEMPORAL_FRESH', r['id'], flush=True)
    save(BASE / 'TEMPORAL_DONE.json', receipt('TEMPORAL', counts, started))
    del uv, rt
    gc.collect()
    torch.cuda.empty_cache()


def spatial(rows):
    from PIL import Image
    from torchvision import transforms
    from scipy.sparse import coo_matrix
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    from sklearn.cluster import KMeans
    import hdbscan
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    pkg = types.ModuleType('unisal')
    pkg.__path__ = [str(BASE / 'deps/unisal/unisal')]
    sys.modules['unisal'] = pkg
    from unisal.model import UNISAL
    cfg = read(BASE / 'weights/unisal/UNISAL.json')
    cfg['cnn_cfg']['pretrained'] = False
    cfg['verbose'] = 0
    model = UNISAL(**cfg)
    model.load_state_dict(torch.load(BASE / 'weights/unisal/weights_best.pth', map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval().requires_grad_(False)
    sizefn = functions(BASE / 'deps/unisal/unisal/data.py', ['get_optimal_out_size'], {'np': np})['get_optimal_out_size']
    core = functions(BASE / 'deps/smartVidCrop.py', ['sc_init_crop_params', 'sc_clustering_filt', 'sc_find_center_of_mass'],
                     dict(np=np, cv2=cv2, coo_matrix=coo_matrix, KMeans=KMeans, sc_register_time=lambda *a: None))
    cp = core['sc_init_crop_params'](use_best_settings=True, print_dict=False)
    cluster = hdbscan.HDBSCAN(min_cluster_size=cp['hdbscan_min'], min_samples=cp['hdbscan_min_samples'])
    for r in rows:
        out = BASE / 'visual' / r['id']
        if (out / 'SPATIAL.npz').exists():
            assert read(out / 'SPATIAL.json')['source_sha256'] == r['media_sha256']
            continue
        samples, picks, clock, cuts, last = [], [], [], [0], None
        with av.open(r['media']) as con:
            for i, f in enumerate(con.decode(video=0)):
                clock.append(float(f.time))
                if i % 6:
                    continue
                small = cv2.resize(f.to_ndarray(format='rgb24'), (250, int(f.height * 250 / f.width)))
                hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_RGB2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if last is not None and cv2.compareHist(last, hist, cv2.HISTCMP_BHATTACHARYYA) > .65:
                    cuts.append(i)
                last = hist
                samples.append(small)
                picks.append(i)
        clock = np.asarray(clock) - clock[0]
        assert np.allclose(clock, np.load(out / 'DSNET.npz')['clock'], atol=.002)
        prep = transforms.Compose([transforms.ToPILImage(), transforms.Resize(sizefn(samples[0].shape[:2]), interpolation=Image.Resampling.LANCZOS),
            transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        centers, maps = [], []
        with torch.inference_mode():
            for k in range(0, len(samples), 8):
                x = torch.stack([prep(im) for im in samples[k:k + 8]]).cuda()[:, None]
                pred = model(x, target_size=samples[0].shape[:2], source='SALICON', static=True).exp()[:, 0, 0].cpu().numpy()
                for sm in pred:
                    sm = (sm / sm.max() * 255).astype(np.uint8)
                    maps.append(sm)
                    sm = sm.copy()
                    sm[sm < cp['t_threshold']] = 0
                    sm = core['sc_clustering_filt'](cluster, sm, cp)
                    cx, cy = core['sc_find_center_of_mass'](sm)
                    centers.append([cx / 250 * r['width'] if cx is not None else r['width'] / 2,
                                    cy / sm.shape[0] * r['height'] if cy is not None else r['height'] / 2])
        full = np.stack([np.interp(np.arange(len(clock)), picks, np.asarray(centers)[:, d]) for d in range(2)], 1)
        for a, b in zip(cuts, cuts[1:] + [len(clock)]):
            n = b - a
            window = min(31, n if n % 2 else n - 1)
            if window >= 5:
                full[a:b] = savgol_filter(median_filter(full[a:b], size=(5, 1)), window, 2, axis=0)
        np.savez_compressed(out / 'SPATIAL.npz', centers=full, maps=maps, sample_frames=picks, clock=clock,
                            width=r['width'], height=r['height'], cuts=cuts)
        save(out / 'SPATIAL.json', dict(source_sha256=r['media_sha256'], cache_sha256=sha(out / 'SPATIAL.npz'), raw_input_fresh=True))
        print('UNISAL_CLUSTER_FRESH', r['id'], flush=True)
    save(BASE / 'SPATIAL_DONE.json', receipt('UNISAL_CLUSTER', dict(UNISAL=sum(p.numel() for p in model.parameters())), started))


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(5701)
    free, total = torch.cuda.mem_get_info()
    # Conservative bound for small models, public clips <=30sec, CLIP batch24.
    assert free > 8 * 2**30 and total > 24 * 2**30, 'Cloud visual-stage resource admission failed'
    rows = manifest()
    assert max(r['frames'] for r in rows) < 2000
    if not (BASE / 'TEMPORAL_DONE.json').exists():
        temporal(rows)
    if not (BASE / 'SPATIAL_DONE.json').exists():
        spatial(rows)
    save(BASE / 'VISUAL_DONE.json', dict(status='FRESH_PUBLIC_TEMPORAL_SPATIAL_COMPLETE', sources=len(rows),
         stages=[read(BASE / 'TEMPORAL_DONE.json'), read(BASE / 'SPATIAL_DONE.json')], formal_inputs=False))
    print('VISUAL_DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(BASE / 'VISUAL_FAILED.json', dict(error=repr(e), source_sha256=sha(__file__)))
        raise
