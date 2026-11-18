"""Public V2: bounded video event answer + exact-frame numbered SAM choices.

Preserves failed V1 outputs. Never repairs missing predicted boxes by hand or
pads them. Grounding returns a candidate ID, then uses that actual mask's box.
"""
import gc
import json
import math
import os
import re
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
from r57_cloud_visual import manifest, read, save, sha, receipt
from r57_cloud_public_semantic import uv_predict
import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

EVENT_PROMPT = '''Describe the main visible event in this video. People, animals, objects, natural events and scenery are all valid subjects. Do not invent audio or identities. Return one JSON object using this exact example structure, changing the values to match the video:
{"event":"a short English event description","participants":["essential visible subject"],"intent":"subject","event_samples":[0,7],"composition":"subject_center"}
intent must be subject, interaction, environment, detail or unknown. composition must be subject_left, subject_center, subject_right, whole_scene or unknown. event_samples is [first,last] among the eight supplied samples indexed 0..7. It is only a coarse event proposal. For scenery participants may be empty. No extra keys or prose.'''
CHOICE_PROMPT = '''The picture has three numbered candidate REGION BOXES. They are alternative, often nested regions, not three distinct objects. For the visible event described below, choose ONE candidate ID that best encloses the main event subject or interacting subset. Do not favor people automatically. Choose none for scenery or when all candidates miss the essential subject. A candidate's confidence is not its importance.
Answer exactly two numbered short lines:
1. One ID from 1, 2, 3, or none.
2. One composition instruction from subject_left, subject_center, subject_right, whole_scene, unknown.
Event: '''


def parse_event(raw):
    a, b = raw.find('{'), raw.rfind('}')
    assert a >= 0 and b > a
    q = json.loads(raw[a:b + 1])
    assert set(q) == {'event', 'participants', 'intent', 'event_samples', 'composition'}
    assert isinstance(q['event'], str) and len(q['event']) > 3
    assert isinstance(q['participants'], list) and all(isinstance(x, str) for x in q['participants'])
    assert q['intent'] in ['subject', 'interaction', 'environment', 'detail', 'unknown']
    assert q['composition'] in ['subject_left', 'subject_center', 'subject_right', 'whole_scene', 'unknown']
    a, b = q['event_samples']
    assert type(a) is int and type(b) is int and 0 <= a <= b < 8
    return q


def parse_choice(raw):
    lines = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r'(?:^|\n)\s*([12])[.)]\s*(.*?)(?=\n\s*[12][.)]|$)', raw.strip(), re.S)}
    assert set(lines) == {1, 2}, 'two numbered answers required'
    ids = re.findall(r'(?<!\d)[123](?!\d)', lines[1])
    none = re.search(r'\bnone\b', lines[1].lower()) is not None
    assert (len(ids) == 1 and not none) or (none and not ids), 'one ID or none required'
    enum = ['subject_left', 'subject_center', 'subject_right', 'whole_scene', 'unknown']
    selected = [v for v in enum if re.search(r'\b' + v + r'\b', lines[2].lower())]
    assert len(selected) == 1, 'one composition required'
    return (int(ids[0]) - 1 if ids else None), selected[0]


def annotated(rgb, boxes):
    image = Image.fromarray(rgb)
    d = ImageDraw.Draw(image)
    colors = ['#ff3b30', '#00bf63', '#187bff']
    font = ImageFont.load_default(size=20)
    for k, (box, color) in enumerate(zip(boxes, colors)):
        x, y, xx, yy = map(int, box)
        if x >= xx or y >= yy:
            continue
        d.rectangle((x, y, xx - 1, yy - 1), outline=color, width=3)
        # Distinct offsets keep labels readable when alternatives share bounds.
        tx, ty = min(image.width - 35, max(0, x + 4 + k * 32)), min(image.height - 30, max(0, y + 4))
        d.rectangle((tx, ty, tx + 28, ty + 26), fill=color)
        d.text((tx + 7, ty), str(k + 1), fill='white', font=font, stroke_width=1)
    return image


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    assert (BASE / 'QWEN_READY.json').exists() and (BASE / 'CANDIDATES_DONE.json').exists()
    free, total = torch.cuda.mem_get_info()
    assert total > 24 * 2**30 and free > 22 * 2**30
    started = time.time()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE / 'models/qwen', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval().requires_grad_(False)
    proc = AutoProcessor.from_pretrained(BASE / 'models/qwen', local_files_only=True, use_fast=False)
    counts = dict(Qwen2_5_VL=sum(p.numel() for p in model.parameters()))
    rows = manifest()
    outdir = BASE / 'semantic_v2'
    outdir.mkdir(exist_ok=True)
    for row in rows:
        rid = row['id']
        dest = outdir / (rid + '.json')
        if dest.exists():
            assert read(dest)['status'] == 'VALID', 'V2 failure preserved; no automatic retries'
            continue
        folder = BASE / 'visual' / rid
        sam = np.load(folder / 'SAM_PROPOSALS.npz')
        clock = sam['clock']
        available = sam['frames']
        wanted_seconds = np.linspace(0, clock[-1], 8)
        slots = np.abs(clock[available, None] - wanted_seconds[None]).argmin(0)
        picks = available[slots]
        assert len(set(picks.tolist())) == 8
        wanted = set(picks.tolist())
        images = {}
        with av.open(row['media']) as con:
            for f, fr in enumerate(con.decode(video=0)):
                if f in wanted:
                    images[f] = fr.to_ndarray(format='rgb24')
        small = []
        for f in picks:
            rgb = images[int(f)]
            scale = min(1, math.sqrt(65536 / (rgb.shape[0] * rgb.shape[1])))
            small.append(cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))))
        text = proc.apply_chat_template([dict(role='user', content=[dict(type='video'), dict(type='text', text=EVENT_PROMPT)])],
                                       tokenize=False, add_generation_prompt=True)
        batch = proc(text=[text], videos=[np.stack(small)], padding=True, return_tensors='pt', do_sample_frames=False,
                     fps=7 / max(.001, float(clock[picks[-1]] - clock[picks[0]]))).to('cuda')
        assert int(batch.input_ids.shape[1]) < 4096
        grid = batch['video_grid_thw'].cpu().tolist()
        patch = int(proc.video_processor.temporal_patch_size)
        assert len(grid) == 1 and grid[0][0] * patch == 8, 'eight-frame video interface mismatch'
        tic = time.time()
        eventpath = outdir / (rid + '_EVENT.json')
        if eventpath.exists():
            eventrec = read(eventpath)
            assert eventrec['source_sha256'] == row['media_sha256'] and eventrec['sample_frames'] == picks.tolist()
            raw = eventrec['raw']
        else:
            with torch.inference_mode():
                gen = model.generate(**batch, max_new_tokens=220, do_sample=False)[:, batch.input_ids.shape[1]:]
            raw = proc.batch_decode(gen, skip_special_tokens=True)[0]
            eventrec = dict(raw=raw, video_grid_thw=grid, temporal_patch_size=patch, tokens=int(gen.shape[1]),
                            seconds=time.time() - tic, sample_frames=picks.tolist(), source_sha256=row['media_sha256'])
            save(eventpath, eventrec)
            del gen
        assert eventrec['tokens'] < 220, 'event answer truncated'
        plan = parse_event(raw)
        del batch
        plan.update(focus_boxes=[], anchor_compositions=[], selected_mask_indices=[])
        anchors = []
        for j, (f, slot) in enumerate(zip(picks, slots)):
            path = outdir / f'{rid}_ANCHOR_{j}.json'
            boxes = sam['boxes'][slot]
            if path.exists():
                rec = read(path)
                assert rec['status'] == 'VALID' and rec['frame'] == int(f), 'saved failure requires explicit new attempt'
                anchors.append(rec)
                plan['focus_boxes'].append(rec['focus_box'])
                plan['anchor_compositions'].append(rec['composition'])
                plan['selected_mask_indices'].append(rec['selected_mask_index'])
                continue
            image = annotated(images[int(f)], boxes)
            prompt = CHOICE_PROMPT + plan['event']
            text = proc.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=prompt)])],
                                           tokenize=False, add_generation_prompt=True)
            # One 640x360 image; no unbounded native-resolution tensor.
            batch = proc(text=[text], images=[image], padding=True, return_tensors='pt').to('cuda')
            assert batch.input_ids.shape[1] < 4096
            t = time.time()
            with torch.inference_mode():
                gen = model.generate(**batch, max_new_tokens=80, do_sample=False)[:, batch.input_ids.shape[1]:]
            raw = proc.batch_decode(gen, skip_special_tokens=True)[0]
            rec = dict(frame=int(f), sam_slot=int(slot), raw=raw, seconds=time.time() - t, tokens=int(gen.shape[1]))
            try:
                assert gen.shape[1] < 80, 'anchor answer truncated'
                k, comp = parse_choice(raw)
                if k is not None:
                    box = boxes[k]
                    assert box[0] < box[2] and box[1] < box[3], 'selected mask is empty'
                    norm = (box / np.asarray([row['width'], row['height'], row['width'], row['height']]) * 1000).tolist()
                else:
                    norm = None
                rec.update(status='VALID', selected_mask_index=k, composition=comp, focus_box=norm)
            except Exception as e:
                rec.update(status='INVALID', error=repr(e))
                save(path, rec)
                raise
            save(path, rec)
            anchors.append(rec)
            plan['focus_boxes'].append(norm)
            plan['anchor_compositions'].append(comp)
            plan['selected_mask_indices'].append(k)
            del batch, gen
        save(dest, dict(status='VALID', plan=plan, anchors=anchors, sample_frames=picks.tolist(),
                        sample_seconds=clock[picks].tolist(), source_sha256=row['media_sha256'],
                        event_record=str(outdir / (rid + '_EVENT.json')), source_sha=sha(__file__),
                        grounder='Qwen numbered exact-frame SAM candidate choice; no free coordinate generation',
                        participant_identity_and_crop_GT=False))
        print('SEMANTIC_V2_PUBLIC_VALID', rid, sum(x is not None for x in plan['focus_boxes']), '/8 grounded', flush=True)
        gc.collect()
    q = receipt('PUBLIC_VQA_SI_AND_TIMESTAMP_BOUND_CANDIDATE_GROUNDING', counts, started)
    q.update(valid=4, total=4, image_grounding_calls=32, video_event_calls=4, schema_not_quality=True,
             previous_failure_directory='semantic; V1 untouched')
    save(BASE / 'SEMANTIC_V2_DONE.json', q)
    print('SEMANTIC_V2_DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(BASE / 'SEMANTIC_V2_FAILED.json', dict(error=repr(e)))
        raise
