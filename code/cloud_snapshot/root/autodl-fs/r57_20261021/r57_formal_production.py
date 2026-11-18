"""Release extraction: generic runtime only; historical batch entry removed."""
"""Frozen direct open-expert inference. No training, API, viewing or submission."""

import gc

import math

import os

import sys

import time

from collections import Counter

from pathlib import Path

BASE = Path(__file__).resolve().parent

sys.path[:0] = [str(BASE), str(BASE/'deps/sam2')]

from r57_cloud_visual import read, save, sha, CLONE

from r57_cloud_public_semantic import PROMPT, uv_predict

from r57_cloud_semantic_v3 import project_event

from r57_cloud_semantic_v2 import annotated, parse_choice, CHOICE_PROMPT

from r57_cloud_compose import flow, crop

import av

import cv2

import numpy as np

import torch

POLICY = dict(version='R57-OPEN-ENSEMBLE-V1', target_coverage=.90,
    selection='DSNet/CG/UVCOM median percentile and majority votes; whole proposals; 2s coverage backstop',
    semantic_interval='Retained as timestamp-bound STVG evidence, not a whole-video selection shortcut',
    SAM_anchor_seconds=.5, VLM_anchors=8, tracking='native-pixel forward-backward LK with shot reset',
    unknown='Keep explicit failed raw calls; generic query or saliency/quality fallback; never synthetic semantic evidence',
    formal_tuning=False, training=False, distillation=False, automatic_submission=False,
    hypothesis='High recall avoids the previous independently observed fixed-30pct event deletion risk; official benefit unknown')

def temporal_pool(folder):
    from helpers.bbox_helper import nms
    d=np.load(folder/'DSNET.npz');clock=d['clock'];n=len(clock);cells=len(d['scores'])
    scores,bb=nms(d['scores'],np.clip(d['boxes'],0,cells).round().astype(int),.5)
    candidates={};coverage=np.zeros(n,np.float32)
    def add(a,b,expert):
        a,b=max(0,int(a)),min(n,int(b))
        if a<b:candidates.setdefault((a,b),dict(start=a,end=b,experts=[]))['experts'].append(expert)
    for s,(a,b) in zip(scores,bb):
        lo,hi=int(d['picks'][a]),int(d['picks'][b]) if b<cells else n
        coverage[lo:hi]=np.maximum(coverage[lo:hi],s);add(lo,hi,'DSNet')
    native={'DSNet':coverage}
    for expert,name in [('CG_DETR','CG_EVENT.json'),('UVCOM','UV_EVENT.json')]:
        p=read(folder/name)
        for a,b,_ in p['pred_relevant_windows']:add(np.searchsorted(clock,a),np.searchsorted(clock,b),expert)
        s=np.asarray(p['pred_saliency_scores']);assert len(s) and np.isfinite(s).all()
        native[expert]=s[np.minimum((clock/2).astype(int),len(s)-1)]
    # The backstop spans the entire fresh native clock, including the last partial cell.
    bounds=sorted(set([0,n]+np.searchsorted(clock,np.arange(0,clock[-1]+2,2)).clip(0,n).tolist()))
    for a,b in zip(bounds,bounds[1:]):add(a,b,'COVERAGE_BACKSTOP')
    pool=list(candidates.values())
    for expert,values in native.items():
        s=np.array([values[c['start']:c['end']].mean() for c in pool])
        for c,v in zip(pool,s):c.setdefault('percentiles',{})[expert]=float(((s<v).sum()+.5*(s==v).sum())/len(s))
    for c in pool:
        p=list(c['percentiles'].values());c['votes']=sum(x>.5 for x in p);c['median']=float(np.median(p))
    pool.sort(key=lambda c:(-c['votes'],-c['median'],c['start'],c['end']))
    keep=np.zeros(n,bool);selected=[]
    for k,c in enumerate(pool):
        if keep.sum()>=math.ceil(n*POLICY['target_coverage']):break
        if keep[c['start']:c['end']].all():continue
        keep[c['start']:c['end']]=True;selected.append(k)
    assert keep.sum()>=math.ceil(n*POLICY['target_coverage'])
    save(folder/'EVENT_POOL.json',dict(policy=POLICY,candidates=pool,selected=selected,kept=int(keep.sum()),native=n,quality_GT=False))
    return np.flatnonzero(keep)

class Runtime:
    def __init__(self):
        from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import r22_lighthouse_runtime as original
        from lighthouse.common.uvcom import build_model
        assert torch.cuda.mem_get_info()[0]>22*2**30
        self.qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE/'models/qwen',local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval().requires_grad_(False)
        self.proc=AutoProcessor.from_pretrained(BASE/'models/qwen',local_files_only=True,use_fast=False)
        self.sam=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(BASE/'weights/sam2.1_hiera_small.pt'),device='cuda').eval().requires_grad_(False)
        self.predictor=SAM2ImagePredictor(self.sam)
        original.ENCODER=CLONE/'aic_highlight/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
        original.BIG=CLONE/'assets/r22_20261016';original.WEIGHT=original.BIG/'weights/clip_cg_detr_qvhighlight.ckpt'
        self.rt=original.Runtime()
        ck=torch.load(BASE/'weights/uvcom.ckpt',map_location='cpu',weights_only=False);opts=ck['opt'];opts.device='cuda'
        self.uv,_=build_model(opts);self.uv.load_state_dict(ck['model'],strict=True);self.uv.cuda().eval().requires_grad_(False)
        del ck
        counts=dict(Qwen2_5_VL=sum(p.numel() for p in self.qwen.parameters()),SAM2=sum(p.numel() for p in self.sam.parameters()),CLIP=sum(p.numel() for p in self.rt.clip.parameters()),CG_DETR=sum(p.numel() for p in self.rt.net.parameters()),UVCOM=sum(p.numel() for p in self.uv.parameters()))
        counts.update(DSNet=4328324,GoogLeNet=5599904,UNISAL=3714872)
        assert sum(counts.values())==3995217693
        self.counts=counts

    def generate(self,content,images=None,video=None,fps=None,max_tokens=80):
        text=self.proc.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        kw=dict(text=[text],padding=True,return_tensors='pt')
        if video is not None:kw.update(videos=[video],do_sample_frames=False,fps=fps)
        else:kw.update(images=images)
        batch=self.proc(**kw).to('cuda');assert batch.input_ids.shape[1]<4096
        with torch.inference_mode():gen=self.qwen.generate(**batch,max_new_tokens=max_tokens,do_sample=False)[:,batch.input_ids.shape[1]:]
        raw=self.proc.batch_decode(gen,skip_special_tokens=True)[0]
        q=dict(raw=raw,tokens=int(gen.shape[1]),input_tokens=int(batch.input_ids.shape[1]),truncated=gen.shape[1]>=max_tokens)
        del batch,gen
        return q

    def analyze(self,row,folder):
        dest=folder/'SEMANTIC.json'
        if dest.exists():return read(dest)
        spatial=np.load(folder/'SPATIAL.npz');clock=spatial['clock'];n=len(clock)
        targets=np.arange(0,float(clock[-1])+.001,.5)
        anchors=np.array(sorted(set(np.searchsorted(clock,targets).clip(0,n-1).tolist())|{n-1}))
        # Keep the public nearest-PTS anchor rule exactly, without an N x T matrix.
        anchors=np.array(sorted({min([max(0,min(n-1,int(np.searchsorted(clock,t))-1)),max(0,min(n-1,int(np.searchsorted(clock,t))))],key=lambda f:abs(clock[f]-t)) for t in targets}|{n-1}))
        slots=np.abs(clock[anchors,None]-np.linspace(0,clock[-1],8)[None]).argmin(0);picks=anchors[slots]
        event_picks=np.abs(clock[:,None]-np.linspace(0,clock[-1],8)[None]).argmin(0)
        wanted=set(anchors.tolist());image_picks=set(picks.tolist())|set(event_picks.tolist());images={};packed=[];boxes=[];qualities=[]
        assert len(anchors)*3*row['width']*row['height']<8*2**30,'bounded native SAM mask contract'
        with av.open(row['media']) as con:
            decoded=0
            for f,fr in enumerate(con.decode(video=0)):
                decoded+=1
                if f not in wanted and f not in image_picks:continue
                rgb=fr.to_ndarray(format='rgb24')
                if f in image_picks:images[f]=rgb
                if f not in wanted:continue
                point=np.clip(spatial['centers'][f],[0,0],[row['width']-1,row['height']-1]).astype(np.float32)[None]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    self.predictor.set_image(rgb)
                    masks,scores,_=self.predictor.predict(point_coords=point,point_labels=np.ones(1,dtype=np.int32),multimask_output=True)
                assert masks.shape==(3,row['height'],row['width'])
                bb=[]
                for mask in masks:
                    yy,xx=np.nonzero(mask);bb.append([int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else [0,0,0,0])
                packed.append(np.packbits(masks.astype(np.uint8).reshape(3,-1),axis=1));boxes.append(bb);qualities.append(scores.tolist())
        assert decoded==n==row['frames']
        packed=np.asarray(packed);boxes=np.asarray(boxes);qualities=np.asarray(qualities)
        np.savez_compressed(folder/'SAM_PACKED.npz',packed=packed,boxes=boxes,qualities=qualities,frames=anchors,shape=[row['height'],row['width']])
        arr=[]
        for f in event_picks:
            rgb=images[int(f)];scale=min(1,math.sqrt(65536/(row['width']*row['height'])))
            arr.append(cv2.resize(rgb,(max(28,round(row['width']*scale)),max(28,round(row['height']*scale)))))
        raw=self.generate([dict(type='video'),dict(type='text',text=PROMPT)],video=np.stack(arr),fps=7/max(.001,float(clock[-1])),max_tokens=640)
        try:
            plan=project_event(raw['raw']);event_status='VALID_TYPED_FIELDS'
        except Exception as e:
            plan=dict(event='',participants=[],intent='unknown',composition='unknown',event_samples=None)
            event_status='UNKNOWN_PARSE_FAILURE';raw['parse_error']=repr(e)
        save(folder/'EVENT_RAW.json',dict(raw,source_sha256=row['media_sha256'],status=event_status,sample_frames=event_picks.tolist(),prompt=PROMPT))
        query=('The complete visible event: '+plan['event']+' Include necessary preparation, interaction and outcome.') if plan['event'] else 'The main visible event, action or interaction unfolds, including preparation, the main moment and its outcome.'
        features=np.load(folder/'CLIP.npz')['features'];save(folder/'CG_EVENT.json',self.rt.predict(features,query))
        text,mask=self.rt.text(query);save(folder/'UV_EVENT.json',dict(uv_predict(self.uv,text,mask,features),query=query))
        plan.update(focus_boxes=[],selected_mask_indices=[],anchor_compositions=[]);records=[]
        for f,slot in zip(picks,slots):
            image=annotated(images[int(f)],boxes[slot]);prompt=CHOICE_PROMPT+(plan['event'] or 'the visible event or scene; uncertainty is allowed')
            rec=self.generate([dict(type='image'),dict(type='text',text=prompt)],images=[image],max_tokens=80)
            try:
                assert not rec['truncated'];k,composition=parse_choice(rec['raw'])
                if k is not None:
                    b=boxes[slot,k];assert b[0]<b[2] and b[1]<b[3]
                    box=(b/np.asarray([row['width'],row['height'],row['width'],row['height']])*1000).tolist()
                else:box=None
                rec['status']='VALID'
            except Exception as e:
                k,composition,box=None,'unknown',None;rec.update(status='UNKNOWN_PARSE_FAILURE',parse_error=repr(e))
            rec.update(frame=int(f),slot=int(slot),selected_mask_index=k,composition=composition,focus_box=box)
            plan['focus_boxes'].append(box);plan['selected_mask_indices'].append(k);plan['anchor_compositions'].append(composition);records.append(rec)
        q=dict(plan=plan,sample_frames=picks.tolist(),event_sample_frames=event_picks.tolist(),event_status=event_status,anchors=records,source_sha256=row['media_sha256'],SAM_cache_sha256=sha(folder/'SAM_PACKED.npz'),GT=False)
        save(dest,q);return q

def compose(row,folder,semantic,frames,public=False):
    spatial=np.load(folder/'SPATIAL.npz');sam=np.load(folder/'SAM_PACKED.npz');packed=sam['packed'];shape=tuple(sam['shape']);anchor={int(f):k for k,f in enumerate(sam['frames'])}
    cuts=set(spatial['cuts'].tolist());plan=semantic['plan'];wanted=set(frames.tolist());pred=[];counts=Counter();mask=None;previous=None
    ratios=[[16,9],[9,16]] if public else [row['targetRatioWH']];records=[[] for _ in ratios];writers=[]
    if public:
        fps=1/float(np.median(np.diff(spatial['clock'])))
        for k,size in enumerate([(640,360),(360,640)]):
            w=cv2.VideoWriter(str(folder/f'PUBLIC_{k}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,size);assert w.isOpened();writers.append(w)
    try:
        with av.open(row['media']) as con:
            decoded=0
            for f,fr in enumerate(con.decode(video=0)):
                rgb=fr.to_ndarray(format='rgb24');gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);decoded+=1
                if f in cuts:mask=None;previous=None
                nearest=int(np.abs(np.asarray(semantic['sample_frames'])-f).argmin());box=plan['focus_boxes'][nearest]
                qbox=np.asarray(box)/1000*[row['width'],row['height'],row['width'],row['height']] if box is not None else None
                if f in anchor:
                    from r57_cloud_compose import iou
                    slot=anchor[f];k=plan['selected_mask_indices'][nearest]
                    if f==semantic['sample_frames'][nearest] and k is not None:counts['SG_exact_frame_anchors']+=1
                    elif qbox is not None:k=max(range(3),key=lambda j:(iou(sam['boxes'][slot,j],qbox),sam['qualities'][slot,j]));counts['SG_coarse_event_link_anchors']+=1
                    else:k=int(sam['qualities'][slot].argmax());counts['SG_unknown_quality_fallback_anchors']+=1
                    mask=np.unpackbits(packed[slot,k],count=shape[0]*shape[1]).reshape(shape).astype(np.uint8);counts['Seg_anchors']+=1
                elif mask is not None and mask.any() and previous is not None:
                    mask=flow(previous,gray,mask);counts['Tracking_success' if mask is not None and mask.any() else 'Tracking_failure']+=1
                comp=plan['anchor_compositions'][nearest]
                if plan['intent']=='environment' or comp=='whole_scene':center=spatial['centers'][f];use=None;counts['SI_environment_frames']+=1
                elif mask is not None and mask.any():
                    yy,xx=np.nonzero(mask);center=[xx.mean(),yy.mean()];use=mask;counts['mask_center_frames']+=1
                elif qbox is not None:center=(qbox[:2]+qbox[2:])/2;use=None;counts['SG_fallback_frames']+=1
                else:center=spatial['centers'][f];use=None;counts['UNISAL_fallback_frames']+=1
                previous=gray
                if f not in wanted:continue
                for k,ratio in enumerate(ratios):
                    b,h=crop(center,use,ratio,row['width'],row['height'],comp);records[k].append(dict(frame=f,bboxes=b))
                    if public:
                        x,y,w=b;writers[k].write(cv2.resize(cv2.cvtColor(rgb[y:y+h,x:x+w],cv2.COLOR_RGB2BGR),[(640,360),(360,640)][k]))
        assert decoded==row['frames'] and all(len(p)==len(frames) for p in records)
    finally:
        for writer in writers:writer.release()
    result=[dict(video_id=row['video_id'],targetRatioWH=ratio,predictions=p) for ratio,p in zip(ratios,records)]
    for rec in result:
        fs=[p['frame'] for p in rec['predictions']];assert fs==sorted(set(fs)) and fs[-1]<row['frames']
        for p in rec['predictions']:
            x,y,w=p['bboxes'];assert x>=0 and y>=0 and w>0 and x+w<=row['width'] and y+math.ceil(w*rec['targetRatioWH'][1]/rec['targetRatioWH'][0])<=row['height']
    return result,dict(counts)
