"""Human-IoU temporal semantic quality features, no formal/test label access.

Neighbor crop semantics are matched across time. This represents visual
correspondence, not annotated subject identity or event relation ground truth.
"""
from pathlib import Path
import os,sys,json,time,hashlib,traceback
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(n,'2')
ROOT=Path(__file__).resolve().parent
sys.path[:0]=['/root/autodl-fs/r59_20261021','/root/autodl-fs/r60_20261022','/root/autodl-fs/r63_20261023']
import numpy as np,torch,cv2,joblib
from transformers import AutoModel
from sklearn.ensemble import HistGradientBoostingRegressor
from r64_semantic_extract import encode
from r64_semantic_ranker import candidate_features,Decision,sha
from r63_path_iou import context_features
from r59_model import path_optimize

def save(n,q):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)

def temporal_features(boxes,scores,semantic,width,height):
 base=context_features(boxes,scores,width,height);result=[]
 for t,(bb,u,roi,xx) in enumerate(zip(boxes,scores,semantic,base)):
  # Unit-normalized PCA crop semantics; temporal matches are unconstrained
  # by GT boxes and can represent moving animal/object/scene counterparts.
  a=roi/np.maximum(np.linalg.norm(roi,axis=1,keepdims=True),1e-6);extra=[]
  center=(bb[:,:2]+bb[:,2:])/2/np.array([width,height]);size=(bb[:,2:]-bb[:,:2])/np.array([width,height])
  for j in [max(0,t-1),min(len(boxes)-1,t+1)]:
   nb=boxes[j];nr=semantic[j];b=nr/np.maximum(np.linalg.norm(nr,axis=1,keepdims=True),1e-6);sim=a@b.T
   nc=(nb[:,:2]+nb[:,2:])/2/np.array([width,height]);ns=(nb[:,2:]-nb[:,:2])/np.array([width,height])
   dc=np.square(center[:,None]-nc[None]).sum(-1);ds=np.square(np.log(np.maximum(size[:,None],1e-6)/np.maximum(ns[None],1e-6))).sum(-1)
   best=(sim+.2*scores[j][None]-.1*dc).argmax(1);ids=np.arange(len(bb));top=int(scores[j].argmax())
   extra.extend([sim[ids,best,None],scores[j][best,None],dc[ids,best,None],ds[ids,best,None],sim[:,top,None]])
  result.append(np.c_[xx,*extra].astype('float32'))
 return result

class Context:
 def __init__(self,path):self.bundle=joblib.load(path)
 def utilities(self,boxes,scores,semantic,width,height):
  x=temporal_features(boxes,scores,semantic,width,height)
  return [np.clip(u+self.bundle['model'].predict(a),0,1).astype('float32') for a,u in zip(x,scores)]

def main():
 import fcntl
 lock=(ROOT/'VIDEO_CONTEXT_TRAIN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'VIDEO_CONTEXT_TRAIN_DONE.json').exists():return
 torch.set_num_threads(2);cv2.setNumThreads(1)
 config=json.loads((ROOT/'VIDEO_MODEL_SELECTION.json').read_text());ckpath=Path(config['checkpoint']);ck=torch.load(ckpath,map_location='cpu',weights_only=False)
 model=Decision(ck['input_dim']).cuda().eval();model.load_state_dict(ck['model']);encoder=AutoModel.from_pretrained(ROOT/'weights/dinov2-base',local_files_only=True).eval().cuda().requires_grad_(False)
 projection=torch.load(ROOT/'dino_projection.pt',weights_only=False);tree=joblib.load('/root/autodl-fs/r60_20261022/fcdb_ranker.joblib');tree.n_jobs=2
 parent=Path('/root/autodl-fs/r59_20261021');cfg=json.loads((parent/'FEATURES_FULL.json').read_text());rows=[r for r in cfg['rows'] if r['role']=='train' and r['kind']=='human_iou' and r['source'].startswith('retarget_')]
 sources=sorted({r['source'] for r in rows});assert len(sources)==20
 hold={s for s in sources if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%5==0};fit=set(sources)-hold
 trainconfig=dict(sources=sources,fit_sources=sorted(fit),hold_sources=sorted(hold),anchors=32,base_checkpoint=str(ckpath),base_sha256=sha(ckpath),features='19 adjacent candidate geometry/utility context +10 real DINO semantic temporal correspondence; not GT identity',supervision='six authentic native1:3/3:1 human crops at32 sampled public training anchors; no AIC or temporal highlight GT',objective='expected absoluteIoU residual; max64iter HGB7leaf learningrate.05; original path penalties unchanged .18/.035',source_sha256=sha(__file__),formal_inputs=False)
 save('VIDEO_CONTEXT_TRAIN_CONFIG.json',trainconfig);xx=[];yy=[];weights=[];held=[];start=time.time()
 for i,source in enumerate(sources):
  rr=sorted([r for r in rows if r['source']==source],key=lambda r:r['file']);rr=[rr[k] for k in np.unique(np.linspace(0,len(rr)-1,32).astype(int))];groups=[[],[]]
  for row in rr:
   assert sha(parent/row['file'])==row['sha256']
   z=np.load(parent/row['file']);f=int(z['frame']);dest=ROOT/'video_train_semantics'/source/f'{f:06d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    d=np.load(dest);patch=d['patch'];cls=d['cls']
   else:
    image=parent/'data/retarget'/source.removeprefix('retarget_')/f'{f:06d}.jpg';rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB);assert rgb.shape[:2]==(int(z['height']),int(z['width']))
    patch,cls=encode(encoder,rgb);np.savez_compressed(dest,patch=patch,cls=cls,image_sha256=sha(image))
   parts=np.cumsum([0]+z['lengths'].tolist())
   for k,(lo,hi) in enumerate(zip(parts[:-1],parts[1:])):
    b=z['boxes'][lo:hi];raw=z['x'][lo:hi].astype('float32');ratio=[[1,3],[3,1]][k]
    x,u=candidate_features(patch,cls,b,raw,int(z['width']),int(z['height']),ratio,projection,tree)
    with torch.inference_mode():p=model(torch.tensor((x-ck['mean'])/ck['std'],device='cuda')).cpu().numpy()
    a,c=ck['calibration'];p=np.clip(a*p+c,0,1);s=ck['alpha']*p+(1-ck['alpha'])*u
    # First128dim are first spatial ROI cell; average four cells to whole crop.
    semantic=x[:,:512].reshape(len(x),128,4).mean(-1)
    groups[k].append(dict(b=b,u=u,s=s,semantic=semantic,y=z['y'][lo:hi],frame=f,w=int(z['width']),h=int(z['height'])))
  for k,g in enumerate(groups):
   f=temporal_features([q['b'] for q in g],[q['s'] for q in g],[q['semantic'] for q in g],g[0]['w'],g[0]['h'])
   if source in fit:
    for a,q in zip(f,g):xx.append(a);yy.append(q['y']-q['s']);weights.append(np.full(len(a),1/len(a)))
   else:held.append(dict(source=source,ratio=k,frames=g,features=f))
  save('VIDEO_CONTEXT_PROGRESS.json',dict(stage='real_public_video_semantics',completed=i+1,total=20,heartbeat=time.time(),seconds=time.time()-start));print('VIDEO_CONTEXT',i+1,20,flush=True)
 reg=HistGradientBoostingRegressor(max_iter=64,max_leaf_nodes=7,max_depth=3,learning_rate=.05,min_samples_leaf=128,l2_regularization=10,early_stopping=False,random_state=6403)
 reg.fit(np.concatenate(xx),np.concatenate(yy),sample_weight=np.concatenate(weights));path=ROOT/'video_semantic_context.joblib';joblib.dump(dict(model=reg,config=trainconfig),path,compress=3)
 save('VIDEO_CONTEXT_BEFORE_HOLD.json',dict(sha256=sha(path),at=time.time()))
 records=[]
 for group in held:
  g=group['frames'];bs=[q['b'] for q in g];base=[q['s'] for q in g];scores=[np.clip(q['s']+reg.predict(x),0,1) for q,x in zip(g,group['features'])];frames=[q['frame'] for q in g];w,h=g[0]['w'],g[0]['h']
  a=path_optimize(bs,base,frames,[],w,h);b=path_optimize(bs,scores,frames,[],w,h)
  records.append(dict(source=group['source'],ratio=group['ratio'],base=float(np.mean([q['y'][j] for q,j in zip(g,a)])),context=float(np.mean([q['y'][j] for q,j in zip(g,b)]))))
 gain=float(np.mean([r['context']-r['base'] for r in records]))
 save('VIDEO_CONTEXT_TRAIN_DONE.json',dict(success=True,checkpoint=str(path),sha256=sha(path),parameters=sum(t.nodes.size*len(t.nodes.dtype.names) for predictors in reg._predictors for t in predictors),held_source_gain=gain,accepted_for_new_video_compare=gain>0,records=records,formal_inputs=False,training_sources=sorted(fit),held_sources=sorted(hold),limitation='small native-ratio public video source hold; not official quality GT'))

if __name__=='__main__':
 try:main()
 except BaseException:save('VIDEO_CONTEXT_TRAIN_FAILED.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
