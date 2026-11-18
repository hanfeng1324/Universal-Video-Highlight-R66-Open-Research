"""Explicit frozen R66 learned structured sequence inference on real video anchor pixels.

Reuses the unchanged R63 candidate geometry and GAIC features. DINO is freshly
executed or reused with an explicit content-addressed receipt; never cache replay
presented as new visual inference. No labels or quality rules from formal inputs.
"""
from pathlib import Path
import sys,json,time
sys.path[:0]=['/root/autodl-fs/r64_20261023','/root/autodl-fs/r63_20261023','/root/autodl-fs/r59_20261021']
import numpy as np,torch,av
from transformers import AutoModel
from r64_semantic_extract import encode
from r64_semantic_ranker import candidate_features,Decision,sha
from r59_model import path_optimize
from r64_video_context import temporal_features
sys.path.insert(0,"/root/autodl-fs/r65_20261023/A")
sys.path.insert(0,'/root/autodl-fs/r65_20261023/A2')
from r65_a2_concepts import ConceptROI
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent/'source_snapshots'))
from r66_rank import features as rank_features, parent as a2_predict, gate_features
from r66_sequence import select_path
R=Path('/root/autodl-fs/r64_20261023')
CACHE=Path('/root/autodl-fs/r63_20261023/PRODUCTION_A_TREE/R63-A-CROP-SEARCH-V1')

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(path)

class SemanticUtility:
 def __init__(self,checkpoint):
  import joblib
  self.trees=joblib.load('/root/autodl-fs/r60_20261022/fcdb_ranker.joblib');self.trees.n_jobs=1;assert len(self.trees.estimators_)==96
  from r66_checkpoint import load as load_checkpoint
  self.checkpoint=Path(checkpoint);self.sha=sha(self.checkpoint);self.bundle=load_checkpoint(checkpoint);self.model=self.bundle['model'];self.model.n_jobs=1;self.transition=self.bundle['sequence_transition'];assert self.bundle['sequence_parameters']==52
  self.scenes=ConceptROI();self.ck=dict(input_dim=1222)
  self.encoder=AutoModel.from_pretrained(R/'weights/dinov2-base',local_files_only=True).eval().cuda().requires_grad_(False)
  self.projection=torch.load(R/'dino_projection.pt',weights_only=False)
  self.calls=self.candidates=self.forward_frames=self.reused_frames=0;self.gate_sum=0.;self.gate_count=0;self.scene_calls=0
 def scene_features(self,rgb):
  self.scene_calls+=1;return self.scenes.image(Image.fromarray(rgb))
 def score_features(self,x,u,scene,return_features=False):
  assert x.shape[1]==1222
  concept,soft=self.scenes.candidates(scene,x[:,1180:1184])
  xx=np.c_[x[:,1024:1193],concept,np.broadcast_to(soft,(len(x),6)),x[:,1193:],np.ones((len(x),1),np.float32)].astype('float32');assert xx.shape[1]==self.bundle['input_dim']
  q=dict(x=xx,u=u);v=self.model.predict(rank_features(xx),num_threads=1);a=a2_predict(q,self.bundle['parent']);z=gate_features(q,v,a)
  g=1/(1+np.exp(-np.clip(z@self.bundle['gate']['w']+self.bundle['gate']['b'],-40,40)));p=(g*v+(1-g)*a).astype('float32')
  assert np.isfinite(p).all();self.gate_sum+=float(g.sum());self.gate_count+=len(g);self.calls+=1;self.candidates+=len(x)
  return (p,xx) if return_features else p

def plans(row,source,semantic,model,expert,ratios,folder,variant='iou'):
 path=folder/'SPATIAL_PATHS.json';seal=folder/'R64_PATH_RECEIPT.json'
 if path.exists():
  assert seal.exists();q=json.loads(seal.read_text())
  if model is not None:assert q['checkpoint_sha256']==model.sha
  assert q['path_sha256']==sha(path)
  return json.loads(path.read_text())
 assert model is not None,'CPU tracking worker requires prepared GPU path'
 role=folder.parent.parent.name;assert role in ['PUBLIC','FORMAL']
 cache=CACHE/role/'work'/folder.name/'RAW_CANDIDATES';assert cache.exists()
 with np.load(source/'SAM_PACKED.npz') as z:frames=z['frames'].astype(int)
 with np.load(source/'SPATIAL.npz') as z:cuts=z['cuts'].tolist()
 wanted=set(frames.tolist());groups=[dict(boxes=[],scores=[],references=[],counts=[],features=[],semantic=[],scenes=[]) for _ in ratios];seen=[];raw_hashes={};start=time.time()
 with av.open(row['media']) as con:
  for f,frame in enumerate(con.decode(video=0)):
   if f not in wanted:continue
   rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2];assert (w,h)==(row['width'],row['height'])
   dest=R/'anchor_semantics'/role/folder.name/f'{f:08d}.npz'
   if not dest.exists():
    prior=Path('/root/autodl-tmp/r65_production/PRODUCTION_A2/R65-A2-LOCAL-CLIP-CONCEPT-IOU-V1')/role/'work'/folder.name/'DINO_CACHE'/f'{f:08d}.npz'
    dest=prior if prior.exists() else folder/'DINO_CACHE'/f'{f:08d}.npz'
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    with np.load(dest) as d:
     assert str(d['media_sha256'])==row['media_sha256'] and str(d['encoder_sha256'])=='d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841'
     patch=d['patch'];cls=d['cls']
    model.reused_frames+=1
   else:
    patch,cls=encode(model.encoder,rgb);pending=dest.with_suffix('.tmp.npz')
    np.savez_compressed(pending,patch=patch,cls=cls,media_sha256=row['media_sha256'],frame=f,encoder_sha256='d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841');pending.replace(dest);model.forward_frames+=1
   scene=model.scene_features(rgb)
   for k,ratio in enumerate(ratios):
    p=cache/f'{f:08d}_{k}.npz';raw_hashes[p.name]=sha(p)
    with np.load(p) as z:
     assert np.array_equal(z['ratio'],ratio) and int(z['frame'])==f and int(z['width'])==w and int(z['height'])==h
     b=z['boxes'];x=z['x'].astype('float32');reference=z['reference'].tolist();assert np.array_equal(z['cuts'],cuts)
    xx,u=candidate_features(patch,cls,b,x,w,h,ratio,model.projection,model.trees);g=groups[k]
    if model.ck['input_dim']==xx.shape[1]:score=model.score_features(xx,u)
    else:
     assert model.ck['input_dim']==xx.shape[1]+29
     score=u;g['features'].append(xx);g['scenes'].append(scene);g['semantic'].append(xx[:,:512].reshape(len(xx),128,4).mean(-1))
    g['boxes'].append(b);g['scores'].append(score);g['references'].append(reference);g['counts'].append(len(b))
   seen.append(f)
 assert seen==frames.tolist();result=[]
 for ratio,g in zip(ratios,groups):
  transition_groups=[]
  if g['features']:
   spans=[0]+[j for j in range(1,len(seen)) if any(seen[j-1]<c<=seen[j] for c in cuts)]+[len(seen)];context=[]
   for lo,hi in zip(spans[:-1],spans[1:]):context.extend(temporal_features(g['boxes'][lo:hi],g['scores'][lo:hi],g['semantic'][lo:hi],row['width'],row['height']))
   scores=[]
   for b,x,c,u,scene in zip(g['boxes'],g['features'],context,g['scores'],g['scenes']):
    score,rank_x=model.score_features(np.c_[x,c],u,scene,return_features=True)
    scores.append(score);transition_groups.append(dict(b=b,x=rank_x))
   g['scores']=scores
  assert len(transition_groups)==len(seen)
  choice=select_path(transition_groups,g['scores'],seen,cuts,row['width'],row['height'],model.transition)
  result.append(dict(ratio=ratio,frames=seen,boxes=[b[j].tolist() for b,j in zip(g['boxes'],choice)],references=g['references'],candidate_counts=g['counts'],selected_quality=[float(s[j]) for s,j in zip(g['scores'],choice)]))
 write(path,result);write(seal,dict(checkpoint_sha256=model.sha,path_sha256=sha(path),unchanged_R63_candidate_features=raw_hashes,anchors=len(seen),seconds=time.time()-start,labels_read=False,formal_training=False,parent_rank_sha256=model.bundle['parent_rank_sha256'],transition_sha256=model.bundle['sequence_transition_sha256'],sequence_parameters=52,cut_resets=True,mechanism='Frozen R66 scene-listwise emissions plus 52-parameter scene/context-conditioned trajectory CRF; per-shot resets, native SAM/LK; semantic signals are model predictions'))
 write(folder.parent.parent/'R64_ACTUAL_EXECUTION.json',dict(checkpoint_sha256=model.sha,calls=model.calls,candidates=model.candidates,DINO_fresh_forward_frames=model.forward_frames,DINO_explicit_cache_reused_frames=model.reused_frames,learned_neural_gate_mean=model.gate_sum/max(1,model.gate_count),CLIP_scene_calls=model.scene_calls,base_96_tree_executed=True,listwise_boosted_trees_executed=True,parent_A2_forest_executed=True,supervised_gate_executed=True,structured_sequence_executed=True,sequence_parameters=52,semantic_GT=False,relationship_GT=False,at=time.time()))
 return result
