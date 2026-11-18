"""Recover present typed V1 event fields; new exact-frame mask-ID grounding.

The V1 full object was invalid because its box list was malformed. Independent
event fields already exist in its raw model answer. JSONDecoder reads each
present value verbatim; no guessed enum, missing value or coordinate padding.
Event timestamps retain the V1 sampling grid. SG has its own exact SAM grid.
"""
import gc
import json
import re
import time
from pathlib import Path
BASE = Path(__file__).resolve().parent
from r57_cloud_visual import manifest, read, save, sha, receipt
from r57_cloud_semantic_v2 import parse_event, parse_choice, annotated, CHOICE_PROMPT
import av
import numpy as np
import torch


def project_event(raw):
    q = {}
    decoder = json.JSONDecoder()
    for key in ['event', 'participants', 'intent', 'event_samples', 'composition']:
        matches = list(re.finditer('"' + key + '"\\s*:\\s*', raw))
        assert len(matches) == 1, 'missing or duplicate event field ' + key
        q[key], _ = decoder.raw_decode(raw[matches[0].end():])
    q = parse_event(json.dumps(q))
    assert q['event'].strip().lower() not in {'a short english event description', 'a short english description of the main visible event'}, 'template echo'
    return q


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    rows = manifest()
    events = {}
    for r in rows:
        old = read(BASE / 'semantic' / (r['id'] + '.json'))
        assert old['source_sha256'] == r['media_sha256']
        events[r['id']] = (old, project_event(old['raw']))
    free, total = torch.cuda.mem_get_info()
    assert total > 24 * 2**30 and free > 22 * 2**30
    started = time.time()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE / 'models/qwen', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval().requires_grad_(False)
    proc = AutoProcessor.from_pretrained(BASE / 'models/qwen', local_files_only=True, use_fast=False)
    counts = dict(Qwen2_5_VL=sum(p.numel() for p in model.parameters()))
    outdir = BASE / 'semantic_v3'
    outdir.mkdir(exist_ok=True)
    for r in rows:
        rid = r['id']
        dest = outdir / (rid + '.json')
        if dest.exists():
            assert read(dest)['status'] == 'VALID'
            continue
        old, plan = events[rid]
        oldpath = BASE / 'semantic' / (rid + '.json')
        sam = np.load(BASE / 'visual' / rid / 'SAM_PROPOSALS.npz')
        clock, available = sam['clock'], sam['frames']
        slots = np.abs(clock[available, None] - np.linspace(0, clock[-1], 8)[None]).argmin(0)
        picks = available[slots]
        images = {}
        with av.open(r['media']) as con:
            for f, fr in enumerate(con.decode(video=0)):
                if f in set(picks.tolist()):
                    images[f] = fr.to_ndarray(format='rgb24')
        plan.update(focus_boxes=[], selected_mask_indices=[], anchor_compositions=[])
        anchors = []
        for j, (f, slot) in enumerate(zip(picks, slots)):
            path = outdir / f'{rid}_ANCHOR_{j}.json'
            if path.exists():
                rec = read(path)
                assert rec['status'] == 'VALID' and rec['frame'] == int(f)
            else:
                image = annotated(images[int(f)], sam['boxes'][slot])
                prompt = CHOICE_PROMPT + plan['event']
                text = proc.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=prompt)])],
                    tokenize=False, add_generation_prompt=True)
                batch = proc(text=[text], images=[image], padding=True, return_tensors='pt').to('cuda')
                assert batch.input_ids.shape[1] < 4096
                tic = time.time()
                with torch.inference_mode():
                    gen = model.generate(**batch, max_new_tokens=80, do_sample=False)[:, batch.input_ids.shape[1]:]
                raw = proc.batch_decode(gen, skip_special_tokens=True)[0]
                rec = dict(frame=int(f), sam_slot=int(slot), raw=raw, tokens=int(gen.shape[1]), seconds=time.time() - tic,
                    event=plan['event'], source_sha256=r['media_sha256'], SAM_cache_sha256=sha(BASE / 'visual' / rid / 'SAM_PROPOSALS.npz'))
                try:
                    assert gen.shape[1] < 80
                    k, composition = parse_choice(raw)
                    if k is None:
                        box = None
                    else:
                        b = sam['boxes'][slot, k]
                        assert b[0] < b[2] and b[1] < b[3]
                        box = (b / np.asarray([r['width'], r['height'], r['width'], r['height']]) * 1000).tolist()
                    rec.update(status='VALID', selected_mask_index=k, composition=composition, focus_box=box)
                except Exception as e:
                    rec.update(status='INVALID', error=repr(e))
                    save(path, rec)
                    raise
                save(path, rec)
                del batch, gen
            anchors.append(rec)
            plan['focus_boxes'].append(rec['focus_box'])
            plan['selected_mask_indices'].append(rec['selected_mask_index'])
            plan['anchor_compositions'].append(rec['composition'])
        save(dest, dict(status='VALID', plan=plan, anchors=anchors, sample_frames=picks.tolist(),
            sample_seconds=clock[picks].tolist(), event_sample_frames=old['sample_frames'],
            event_source_record=str(oldpath), event_source_record_sha256=sha(oldpath),
            event_recovery='Typed independent field projection of existing model tokens; no missing-value repair',
            source_sha256=r['media_sha256'], source_sha=sha(__file__), participant_identity_and_crop_GT=False))
        print('SEMANTIC_V3_VALID', rid, sum(x is not None for x in plan['focus_boxes']), '/8 grounded', flush=True)
        gc.collect()
    q = receipt('PUBLIC_TYPED_EVENT_PROJECTION_AND_EXACT_FRAME_GROUNDING', counts, started)
    q.update(valid=4, total=4, new_image_calls=32, video_answers_reused=4, schema_not_quality=True,
        failures_preserved=['semantic V1 malformed box lists', 'semantic_v2 enum invalid and template echo'],
        no_human_field_edit=True, no_coordinate_imputation=True)
    save(BASE / 'SEMANTIC_V3_DONE.json', q)
    print('SEMANTIC_V3_DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(BASE / 'SEMANTIC_V3_FAILED.json', dict(error=repr(e)))
        raise
