"""Runtime definitions only; namespace supplied by run_inference.selected_module.
Original competition batch/validator/packaging workflow excluded.
"""
class IoUUtility:
 def __init__(self,trees):
  self.trees=trees;self.trees.n_jobs=2;assert len(trees.estimators_)==96;self.calls=0;self.candidates=0
 def __call__(self,x):
  self.calls+=1;self.candidates+=x.shape[1]
  q=self.trees.predict(features(x[0].detach().cpu().numpy())).astype('float32')
  save(ROOT/'ACTUAL_96_TREE_EXECUTION.json',dict(trees=len(self.trees.estimators_),calls=self.calls,candidates=self.candidates,utility_min=float(q.min()),utility_max=float(q.max()),sigmoid=False,expected_human_iou=True,checkpoint_sha256='65fa2a9632b973eb5f8c23941f430050ec2be78829df3b1a0561ffdad4bf09f7'))
  p=torch.from_numpy(q).to(x.device)[None]
  return p,torch.zeros_like(p)


def plans(row,source,semantic,model,expert,ratios,folder,variant='early'):
 path=folder/'SPATIAL_PATHS.json'
 if path.exists():return read(path)
 # Materialize once: NpzFile otherwise decompresses the complete mask archive
 # again on every anchor lookup. This is computation-equivalent caching.
 with np.load(source/'SPATIAL.npz') as z:sp={k:z[k] for k in z.files}
 with np.load(source/'SAM_PACKED.npz') as z:sam={k:z[k] for k in z.files}
 clock=sp['clock'];sf=np.asarray(sam['frames']);n=row['frames'];cuts=sp['cuts'].tolist();wanted=set(sf.tolist());paths=[dict(boxes=[],scores=[],references=[]) for _ in ratios];frames=[]
 # True RGB at each SAM temporal anchor, never synthetic replay as new inference.
 with av.open(row['media']) as con:
  for f,fr in enumerate(con.decode(video=0)):
   if f not in wanted:continue
   rgb=fr.to_ndarray(format='rgb24');h,w=rgb.shape[:2];slot=int(np.searchsorted(sf,f));near=int(np.abs(np.asarray(semantic['sample_frames'])-f).argmin());plan=semantic['plan'];subjects=[b for b in sam['boxes'][slot] if b[2]>b[0] and b[3]>b[1]]
   qb=plan['focus_boxes'][near]
   if qb is not None:subjects.append(np.asarray(qb)/1000*[w,h,w,h])
   if len(subjects)>1:
    # Every pair is a hypothesis; co-occurrence is not asserted as interaction GT.
    for i in range(min(3,len(subjects))):
     for j in range(i+1,min(4,len(subjects))):
      a,b=np.asarray(subjects[i]),np.asarray(subjects[j]);subjects.append(np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])])
   mi=int(np.clip(np.searchsorted(sp['sample_frames'],f,side='right')-1,0,len(sp['maps'])-1));sm=sp['maps'][mi];centers=[sp['centers'][f]]
   if 'raw_centers' in sp:centers.append(sp['raw_centers'][f])
   # Distinct saliency peaks add alternatives beyond inherited clustered focus.
   for _ in range(3):
    tmp=sm.copy() if _==0 else tmp;y,x=np.unravel_index(tmp.argmax(),tmp.shape);centers.append([x/tmp.shape[1]*w,y/tmp.shape[0]*h]);cv2.circle(tmp,(int(x),int(y)),max(2,min(tmp.shape)//8),0,-1)
   # Match the existing semantic SAM identity exactly for native residual motion.
   from r57_cloud_compose import iou
   selected=plan['selected_mask_indices'][near]
   if not (f==semantic['sample_frames'][near] and selected is not None):
    selected=max(range(3),key=lambda j:(iou(sam['boxes'][slot,j],np.asarray(qb)/1000*[w,h,w,h]),sam['qualities'][slot,j])) if qb is not None else int(sam['qualities'][slot].argmax())
   mask=np.unpackbits(sam['packed'][slot,selected],count=w*h).reshape(h,w);yy,xx=np.nonzero(mask);reference=[float(xx.mean()),float(yy.mean())] if len(xx) else np.asarray(centers[0]).tolist()
   for k,ratio in enumerate(ratios):
    b=proposals(w,h,ratio,centers,subjects)
    if row.get('video_id') in FORMAL_PARENTS:
     parent=FORMAL_PARENTS[row['video_id']];pp=min(parent['predictions'],key=lambda p:abs(p['frame']-f));px,py,pw=pp['bboxes'];pb=[px,py,px+pw,py+math.ceil(pw*ratio[1]/ratio[0])];b=np.unique(np.r_[b,[pb]],axis=0)
    rawpath=folder/'RAW_CANDIDATES'/f'{f:08d}_{k}.npz'
    if rawpath.exists():
     with np.load(rawpath) as raw:b=raw['boxes'];x=raw['x'];score=raw['u60'];original_count=int(raw['original_count'])
    else:
     oldraw=Path('/root/autodl-fs/r62_20261023/BASELINE/R62-R60-BASELINE-REPLAY')/folder.parent.parent.name/'work'/folder.name/'RAW_CANDIDATES'/f'{f:08d}_{k}.npz'
     oldx=None
     if oldraw.exists():
      with np.load(oldraw) as cached:
       assert np.array_equal(cached['boxes'],b);oldx=cached['x'].astype('float32')
     original_count=len(b)
     if R63_CHOICE.startswith('A'):
      inherited=[]
      if row.get('video_id') in R60_RECORDS:
       pp=min(R60_RECORDS[row['video_id']]['predictions'],key=lambda p:abs(p['frame']-f));px,py,pw=pp['bboxes'];inherited.append([px,py,px+pw,py+math.ceil(pw*ratio[1]/ratio[0])])
      hist=Path('/root/autodl-fs/r60_20261022/R60-HUMAN-CROP-IOU-V2')/folder.parent.parent.name/'work'/folder.name/'SPATIAL_PATHS.json'
      if hist.exists():
       hp=read(hist)[k];ii=int(np.abs(np.asarray(hp['frames'])-f).argmin());inherited.append(hp['boxes'][ii])
      previous=[]
      if paths[k]['boxes']:previous=[paths[k]['boxes'][-1][int(paths[k]['scores'][-1].argmax())]]
      b,score,x,cc=search_core.search(expert,model.trees,rgb,ratio,b,sm,raw=oldx,regions=subjects,inherited=inherited,previous=previous)
      if R63_CHOICE=='A_ADAPTED':
       if not hasattr(model,'adapter'):model.adapter=Adapter('/root/autodl-fs/r63_20261023/r63_candidate_residual.joblib')
       score=model.adapter.predict(x,score)
     else:
      assert oldx is not None,'B requires the complete existing real feature cache'
      x=oldx if oldx is not None else frozen_features(expert,rgb,b,sm)[0];score=search_core.score(model.trees,x)
     assert np.isfinite(score).all();rawpath.parent.mkdir(parents=True,exist_ok=True);pending=rawpath.with_suffix('.tmp.npz')
     np.savez(pending,x=x.astype('float32'),boxes=b,u60=score.astype('float32'),original_count=original_count,frame=f,ratio=ratio,width=w,height=h,reference=np.asarray(reference),cuts=np.asarray(cuts));pending.replace(rawpath)
    model.calls+=1;model.candidates+=len(b)
    save(ROOT/'R63_ACTUAL_EXECUTION.json',dict(calls=model.calls,candidates=model.candidates,real96trees=True,mechanism=R63_CHOICE,raw_feature_dim=x.shape[1],tree_feature_dim=156,tree_sha256=search_core.TREE_SHA,new_feature_forward=True))
    paths[k]['boxes'].append(b);paths[k]['scores'].append(score);paths[k]['references'].append(reference)
   frames.append(f)
 assert len(frames)==len(sf)
 result=[]
 for k,ratio in enumerate(ratios):
  q=paths[k]
  if R63_CHOICE=='B':
   if not hasattr(model,'path_adapter'):model.path_adapter=PathAdapter('/root/autodl-fs/r63_20261023/r63_path_context.joblib')
   # Reset context at scene boundaries as the original optimizer does.
   spans=[0]+[j for j in range(1,len(frames)) if any(frames[j-1]<c<=frames[j] for c in cuts)]+[len(frames)]
   corrected=[]
   for lo,hi in zip(spans[:-1],spans[1:]):corrected.extend(model.path_adapter.utilities(q['boxes'][lo:hi],q['scores'][lo:hi],row['width'],row['height']))
   q['scores']=corrected
  choices=path_optimize(q['boxes'],q['scores'],frames,cuts,row['width'],row['height']);b=np.array([bb[c] for bb,c in zip(q['boxes'],choices)])
  result.append(dict(ratio=ratio,frames=frames,boxes=b.tolist(),references=q['references'],candidate_counts=[len(v) for v in q['boxes']],selected_quality=[float(s[c]) for s,c in zip(q['scores'],choices)]))
 save(path,result);return result


def compose(row,source,semantic,frames,model,expert,ratios,folder,public,variant='early'):
 import r57_formal_production as production
 plan=plans(row,source,semantic,model,expert,ratios,folder,variant);count=[0];original=production.crop
 sp=np.load(source/'SPATIAL.npz');cuts=set(sp['cuts'].tolist());counts={'new_checkpoint_spatial_calls':0,'mask_motion_adjustments':0}
 def newcrop(center,mask,ratio,w,h,comp):
  i=count[0]//len(ratios);k=count[0]%len(ratios);count[0]+=1;f=int(frames[i]);p=plan[k];af=np.asarray(p['frames']);b=np.array([np.interp(f,af,np.asarray(p['boxes'])[:,d]) for d in range(4)])
  # Interpolate only within the current shot; use nearest anchor on boundary gaps.
  pos=int(np.clip(np.searchsorted(af,f,side='right')-1,0,len(af)-1))
  if pos+1<len(af) and any(af[pos]<c<=af[pos+1] for c in cuts):b=np.asarray(p['boxes'][pos if f<min(c for c in cuts if af[pos]<c<=af[pos+1]) else pos+1],float)
  # Native SAM/LK propagation supplies residual motion, never parent geometry veto.
  if mask is not None and mask.any() and comp!='whole_scene':
   yy,xx=np.nonzero(mask);now=np.array([xx.mean(),yy.mean()]);ref=np.array([np.interp(f,af,np.asarray(p['references'])[:,d]) for d in range(2)]);delta=np.clip(now-ref,[-.12*w,-.12*h],[.12*w,.12*h]);b[:2]+=delta;b[2:]+=delta;counts['mask_motion_adjustments']+=1
  cx,cy=(b[:2]+b[2:])/2;out=legal(cx,cy,b[2]-b[0],ratio,w,h);x,y,x1,y1=map(int,out);counts['new_checkpoint_spatial_calls']+=1
  return [x,y,x1-x],y1-y
 # Existing native pixel tracking, segmentation, semantic linkage and SI execute.
 # Only its original clustering/mask crop decision is replaced by learned path.
 production.crop=newcrop
 try:
  recs,ct=production.compose(row,folder,semantic,frames,public=public)
 finally:production.crop=original
 assert count[0]==len(frames)*len(ratios);ct.update(counts)
 return recs,ct
