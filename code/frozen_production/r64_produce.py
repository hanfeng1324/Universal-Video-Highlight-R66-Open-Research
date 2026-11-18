"""Runtime definitions only; namespace supplied by run_inference.selected_module.
Original competition batch/validator/packaging workflow excluded.
"""
def contract(record,row):
 fs=[p['frame'] for p in record['predictions']];assert fs and fs==sorted(set(fs)) and fs[0]>=0 and fs[-1]<row['frames'];rw,rh=record['targetRatioWH']
 for p in record['predictions']:
  assert set(p)=={'frame','bboxes'} and type(p['frame']) is int
  x,y,w=p['bboxes'];assert all(type(q) is int for q in [x,y,w]) and x>=0 and y>=0 and w>0 and x+w<=row['width'] and y+math.ceil(w*rh/rw)<=row['height']


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
