"""Pinned open VLM on public videos only; semantics and spatiotemporal boxes.

Predictions are uncalibrated model evidence. Missing boxes are unknown, never
negative event labels. A parse failure blocks the complete-prototype claim.
"""
import gc
import json
import math
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
PROMPT = '''You are helping edit a video. Analyze only visible content, including animals, objects, interactions, scenery and natural events. Do not prefer people automatically. Return ONE JSON object, without Markdown, with exactly these keys:
"event": one short English description of the main visible event;
"participants": a list of essential visible participants, possibly empty for scenery;
"intent": one of "subject", "interaction", "environment", "detail", "unknown";
"event_samples": two integers [first,last] in 0..7 marking the main event in the eight input samples;
"focus_boxes": a list of exactly eight entries, one for each sampled frame in order. Each entry is [left,top,right,bottom] with normalized integer coordinates 0..1000, enclosing the event's main subject or interacting subset in THAT frame, or null when no reliable object region exists;
"composition": one of "subject_left", "subject_center", "subject_right", "whole_scene", "unknown", indicating the intended position of the main subject in a crop.
Do not invent participants or audio. A box encloses the selected event participants, not unrelated background. Use null when uncertain. Keep the event description and participant names concise.'''


@torch.inference_mode()
def uv_predict(model, text, mask, features):
    scores, windows = [], []
    for off in range(0, len(features), 75):
        v = F.normalize(torch.tensor(features[off:off + 75], device='cuda').float(), dim=-1)
        n = len(v)
        tef = torch.stack([torch.arange(n, device='cuda') / n, (torch.arange(n, device='cuda') + 1) / n], -1)
        out = model(src_vid=torch.cat([v, tef], -1)[None], src_vid_mask=torch.ones(1, n, device='cuda'),
                    src_txt=text, src_txt_mask=mask)
        scores.extend(out['saliency_scores'][0].float().cpu().tolist())
        spans = out['pred_spans'][0]
        ab = torch.stack([spans[:, 0] - spans[:, 1] / 2, spans[:, 0] + spans[:, 1] / 2], -1).clamp(0, 1) * n * 2 + off * 2
        conf = out['pred_logits'][0].softmax(-1)[:, 0]
        windows.extend(torch.cat([ab, conf[:, None]], -1).float().cpu().tolist())
    assert len(scores) == len(features) and np.isfinite(scores).all()
    return dict(pred_saliency_scores=scores, pred_relevant_windows=windows)


def parse(raw):
    a, b = raw.find('{'), raw.rfind('}')
    assert a >= 0 and b > a, 'JSON absent'
    q = json.loads(raw[a:b + 1])
    assert set(q) == {'event', 'participants', 'intent', 'event_samples', 'focus_boxes', 'composition'}
    assert isinstance(q['event'], str) and len(q['event']) > 3
    assert isinstance(q['participants'], list) and all(isinstance(x, str) for x in q['participants'])
    assert q['intent'] in ['subject', 'interaction', 'environment', 'detail', 'unknown']
    assert q['composition'] in ['subject_left', 'subject_center', 'subject_right', 'whole_scene', 'unknown']
    span = q['event_samples']
    assert len(span) == 2 and all(type(x) is int for x in span) and 0 <= span[0] <= span[1] < 8
    assert len(q['focus_boxes']) == 8
    for box in q['focus_boxes']:
        if box is None:
            continue
        assert len(box) == 4 and all(type(x) in [float, int] and math.isfinite(x) and 0 <= x <= 1000 for x in box)
        assert box[0] < box[2] and box[1] < box[3]
    return q


def main():
    from r57_cloud_visual import manifest, read, save, sha, receipt
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    assert (BASE / 'QWEN_READY.json').exists() and (BASE / 'VISUAL_DONE.json').exists()
    wait_start = time.time()
    while (BASE / 'VIDEO_TRACK.lock').exists():
        pid = int((BASE / 'VIDEO_TRACK.lock').read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        assert time.time() - wait_start < 360, 'tracking wait deadline'
        time.sleep(2)
    rows = manifest()
    pending = [r for r in rows if not (BASE / 'semantic' / (r['id'] + '.json')).exists()]
    if not pending:
        print('SEMANTIC_ALL_CACHED', flush=True)
        return
    free, total = torch.cuda.mem_get_info()
    # BF16 weights ~7GiB; bounded 8x64k pixels + <=640 new tokens; 22GiB admission.
    assert total > 24 * 2**30 and free > 22 * 2**30, 'VLM cloud memory admission failed'
    torch.set_num_threads(4)
    torch.manual_seed(5702)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE / 'models/qwen', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval().requires_grad_(False)
    proc = AutoProcessor.from_pretrained(BASE / 'models/qwen', local_files_only=True, use_fast=False)
    counts = dict(Qwen2_5_VL=sum(p.numel() for p in model.parameters()))
    for r in pending:
        dest = BASE / 'semantic' / (r['id'] + '.json')
        clock = np.load(BASE / 'visual' / r['id'] / 'DSNET.npz')['clock']
        targets = np.linspace(clock[0], clock[-1], 8)
        picks = np.abs(clock[:, None] - targets[None]).argmin(0)
        images = {}
        with av.open(r['media']) as con:
            for i, frame in enumerate(con.decode(video=0)):
                if i in set(picks.tolist()):
                    rgb = frame.to_ndarray(format='rgb24')
                    scale = min(1, math.sqrt(65536 / (frame.width * frame.height)))
                    rgb = cv2.resize(rgb, (max(28, round(frame.width * scale)), max(28, round(frame.height * scale))))
                    images[i] = rgb
        arr = np.stack([images[int(i)] for i in picks])
        msg = [dict(role='user', content=[dict(type='video'), dict(type='text', text=PROMPT)])]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        batch = proc(text=[text], videos=[arr], padding=True, return_tensors='pt', do_sample_frames=False,
                     fps=7 / max(.001, float(targets[-1] - targets[0]))).to('cuda')
        assert int(batch.input_ids.shape[1]) < 4096, 'bounded public token contract exceeded'
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**batch, max_new_tokens=640, do_sample=False)
        gen = out[:, batch.input_ids.shape[1]:]
        raw = proc.batch_decode(gen, skip_special_tokens=True)[0]
        q = dict(source_sha256=r['media_sha256'], sample_frames=picks.tolist(), sample_seconds=clock[picks].tolist(),
                 raw=raw, tokens=int(gen.shape[1]), input_tokens=int(batch.input_ids.shape[1]), seconds=time.time() - t,
                 source_sha=sha(__file__), prompt=PROMPT, uncalibrated=True, participant_or_crop_GT=False)
        try:
            assert gen.shape[1] < 640, 'output truncated'
            q.update(plan=parse(raw), status='VALID')
        except Exception as e:
            q.update(status='INVALID', error=repr(e))
        save(dest, q)
        print('SEMANTIC_PUBLIC', r['id'], q['status'], round(q['seconds'], 2), flush=True)
        del batch, out, gen, arr
        gc.collect()
    valid = [read(BASE / 'semantic' / (r['id'] + '.json'))['status'] == 'VALID' for r in rows]
    rec = receipt('VQA_SI_SG_STVG_VIDEO', counts, started)
    rec.update(valid=sum(valid), total=len(rows), schema_valid_not_quality=True)
    save(BASE / 'SEMANTIC_DONE.json', rec)
    assert all(valid), 'Invalid semantic outputs: stop before complete seven-capability claim'


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        from r57_cloud_visual import save
        save(BASE / 'SEMANTIC_FAILED.json', dict(error=repr(e)))
        raise
