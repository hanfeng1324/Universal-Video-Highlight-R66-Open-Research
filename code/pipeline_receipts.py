"""Evidence of executed inference; success never means semantic ground truth."""
import json
from pathlib import Path


def read(path): return Path(path).read_text(encoding='utf-8')


def record_capabilities(work, base, rows, counters, fresh_runtime=True):
    results=[]
    for row, count in zip(rows, counters):
        source=base/'visual'/row['id']
        semantic=json.loads(read(source/'SEMANTIC.json'))
        execution=json.loads(read(work/'sequence/PUBLIC/R64_ACTUAL_EXECUTION.json'))
        calls=semantic['anchors']
        evidence={
            'TG':dict(executed=all((source/n).is_file() for n in ['DSNET.npz','CG_EVENT.json','UV_EVENT.json','EVENT_POOL.json']), experts=['DSNet','CG-DETR','UVCOM']),
            'SG':dict(executed=execution['candidates']>0, candidates=execution['candidates'], scene_calls=execution['CLIP_scene_calls'], listwise=execution['listwise_boosted_trees_executed'], sequence_iou=execution['structured_sequence_executed'], transition_parameters=execution['sequence_parameters']),
            'Seg':dict(executed=count.get('Seg_anchors',0)>0, anchors=count.get('Seg_anchors',0), backend='SAM2 image masks at anchors'),
            'Tracking':dict(executed=count.get('Tracking_success',0)+count.get('Tracking_failure',0)>0, successful_frames=count.get('Tracking_success',0), failed_frames=count.get('Tracking_failure',0), backend='native forward/backward LK'),
            'STVG':dict(executed=(source/'EVENT_RAW.json').is_file() and len(calls)>0, event_parse_status=semantic['event_status'], anchor_grounding_calls=len(calls)),
            'VQA':dict(executed=bool(calls) and all('raw' in c for c in calls), calls=len(calls)+1, valid_grounding_answers=sum(c['status']=='VALID' for c in calls), unknown_grounding_answers=sum(c['status']!='VALID' for c in calls)),
            'SI':dict(executed=(source/'SPATIAL.npz').is_file() and execution['DINO_fresh_forward_frames']>0 and execution['CLIP_scene_calls']>0, backends=['UNISAL','CLIP','DINOv2'], fresh_dino_frames=execution['DINO_fresh_forward_frames']),
        }
        results.append(dict(video_id=row['video_id'],input_frames=row['frames'],evidence=evidence))
    return dict(all_seven_executed=all(v['executed'] for r in results for v in r['evidence'].values()), rows=results, fresh_work_directory=fresh_runtime,
                scope='Actual model/algorithm execution only; parse failures remain explicit, no semantic or event-quality certification.')
