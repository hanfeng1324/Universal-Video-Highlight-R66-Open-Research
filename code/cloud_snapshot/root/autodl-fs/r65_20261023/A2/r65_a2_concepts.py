"""Candidate-level CLIP concept retention, genuine cropIoU; no final/test labels.

Concepts and relation cues are model outputs, never human semantic annotation.
Reuses all completed A crop/GAIC/DINO/temporal features, adds actual CLIP ROI.
"""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,zipfile,io
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parent;A=ROOT.parent/'A';R=Path('/root/autodl-fs/r64_20261023')
sys.path[:0]=[str(A),str(R),'/root/autodl-fs/r59_20261021','/root/autodl-fs/r60_20261022']
import numpy as np,torch,joblib
from PIL import Image
from torchvision.ops import roi_align
from transformers import CLIPTokenizer
from sklearn.ensemble import ExtraTreesRegressor
from r65_a_scene import Scenes
from r59_model import path_optimize

CONCEPTS=['a person','a face','a person talking','two people interacting','a crowd','an animal','a bird','a dog or cat','two animals interacting','a person interacting with an animal','a landscape','the sky','water','trees','a natural phenomenon','a ball','a goal or net','an athlete','sports action','a racket','a vehicle','a building','food','an object being manipulated']
PAIRS=[(0,5),(0,15),(15,16),(17,15),(5,6),(0,23),(10,0),(10,5)]

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,q):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_text(json.dumps(q,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()));t.replace(p)

class ConceptROI:
 def __init__(self):
  self.scene=Scenes();self.model=self.scene.model
  tokenizer=CLIPTokenizer.from_pretrained('/root/autodl-fs/r57_20261021/clone_essentials/root/r53_autonomous_20261020/assets/r22_20261016/clip_tokenizer',local_files_only=True)
  with torch.inference_mode():text=self.model.get_text_features(**tokenizer(['a photo of '+s for s in CONCEPTS],padding=True,return_tensors='pt').to('cuda'))
  if not isinstance(text,torch.Tensor):text=text.pooler_output
  self.text=torch.nn.functional.normalize(text,dim=-1)
 def image(self,im):
  im=im.convert('RGB');w,h=im.size;scale=224/max(w,h);nw,nh=round(w*scale),round(h*scale);left,top=(224-nw)//2,(224-nh)//2
  canvas=Image.new('RGB',(224,224));canvas.paste(im.resize((nw,nh)),(left,top))
  pixel=self.scene.processor(images=canvas,do_resize=False,do_center_crop=False,return_tensors='pt').to('cuda')
  with torch.inference_mode():
   out=self.model.vision_model(pixel_values=pixel['pixel_values']);patch=self.model.visual_projection(self.model.vision_model.post_layernorm(out.last_hidden_state[:,1:]));patch=torch.nn.functional.normalize(patch,dim=-1)
   logits=(patch@self.text.T).reshape(1,7,7,len(CONCEPTS)).permute(0,3,1,2)
   globalv=torch.nn.functional.normalize(self.model.visual_projection(out.pooler_output),dim=-1);scene=(globalv@self.scene.text.T*10).softmax(-1)[0]
  return logits,scene.cpu().numpy(),(left,top,nw,nh)
 def candidates(self,encoded,norm):
  maps,scene,(left,top,nw,nh)=encoded;b=norm.copy();b[:,[0,2]]=b[:,[0,2]]*nw+left;b[:,[1,3]]=b[:,[1,3]]*nh+top;b*=7/224
  box=torch.tensor(np.c_[np.zeros(len(b)),b],device='cuda',dtype=torch.float32)
  with torch.inference_mode():
   roi=roi_align(maps,box,output_size=(2,2),sampling_ratio=2,aligned=True);inside=roi.mean((2,3));glob=maps.mean((2,3)).expand(len(b),-1)
   area=torch.tensor(np.maximum((norm[:,2]-norm[:,0])*(norm[:,3]-norm[:,1]),1e-4),device='cuda')[:,None]
   outside=((glob-area*inside)/(1-area).clamp_min(.05)).clamp(-1,1)
   positive=(maps*10).softmax(1);mass=roi_align(positive,box,output_size=(2,2),sampling_ratio=2,aligned=True).mean((2,3))*area
   pairs=torch.stack([mass[:,a]*mass[:,z] for a,z in PAIRS],1)
   v=torch.cat([roi.flatten(1),inside,outside,inside-outside,mass,pairs],1).cpu().numpy()
  return v.astype('float32'),scene

def prepare():
 encoder=ConceptROI();cache=ROOT/'features';cache.mkdir(exist_ok=True);out=[]
 fz=zipfile.ZipFile('/root/autodl-fs/r60_20261022/public_data/FCDB.zip');fn={Path(n).stem.split('_')[0]:n for n in fz.namelist() if n.lower().endswith('.jpg')}
 gz=zipfile.ZipFile(R/'GNMC/GNMC.zip');gn={Path(n).stem:n for n in gz.namelist() if '/train/' in n and '__MACOSX' not in n and n.lower().endswith(('.jpg','.jpeg','.png'))}
 files=sorted((A/'features').glob('*.joblib'))
 for i,p in enumerate(files):
  dest=cache/p.name
  if dest.exists():groups=joblib.load(dest)
  else:
   groups=joblib.load(p);encoded={}
   for q in groups:
    if q['domain']=='fcdb':key=0;im=lambda:Image.open(io.BytesIO(fz.read(fn[q['source'].removeprefix('fcdb_')])) )
    elif q['domain']=='gnmc':key=0;im=lambda:Image.open(io.BytesIO(gz.read(gn[q['source'].split('_')[-1]])))
    else:key=q['frame'];im=lambda:Image.open(Path('/root/autodl-fs/r59_20261021/data/retarget')/q['source'].split('_')[-1]/f'{q["frame"]:06d}.jpg')
    if key not in encoded:encoded[key]=encoder.image(im())
    old=q['x'];v,scene=encoder.candidates(encoded[key],old[:,1180:1184]);q['x']=np.c_[old[:,1024:1193],v,np.broadcast_to(scene,(len(v),6)),old[:,-30:]].astype('float32');q['scene']=scene
   joblib.dump(groups,dest,compress=3)
  out.extend(groups)
  if i%16==0:save('PROGRESS.json',dict(stage='actual_CLIP_candidate_ROI_concept_retention',completed=i+1,total=len(files),groups=len(out),at=time.time()))
 del encoder;torch.cuda.empty_cache();return out

def evaluate(model,groups,gate=None):
 rows=[];paths={}
 for q in groups:
  p=model.predict(q['x']).astype('float32')
  if gate is not None:
   with torch.inference_mode():g=torch.sigmoid(torch.tensor(q['x'])@gate['w']+gate['b']).numpy()
   p=g*p+(1-g)*q['u']
  rows.append(dict(source=q['source'],domain=q['domain'],ratio=str(q['ratio']),tree=float(q['y'][q['u'].argmax()]),new=float(q['y'][p.argmax()])))
  if q['domain']=='video':paths.setdefault((q['source'],q['ratio']),[]).append((q,p))
 pp=[]
 for key,rr in paths.items():
  rr.sort(key=lambda z:z[0]['frame']);qs=[q for q,p in rr];bb=[q['b'] for q in qs];ff=[q['frame'] for q in qs];w,h=qs[0]['w'],qs[0]['h'];a=path_optimize(bb,[q['u'] for q in qs],ff,[],w,h);b=path_optimize(bb,[p for q,p in rr],ff,[],w,h);pp.append(dict(source=key[0],ratio=key[1],tree=float(np.mean([q['y'][j] for q,j in zip(qs,a)])),new=float(np.mean([q['y'][j] for q,j in zip(qs,b)]))))
 domains={}
 for d in ['fcdb','gnmc','video']:
  rr=[r for r in rows if r['domain']==d];ds=[np.mean([r['new']-r['tree'] for r in rr if r['source']==s]) for s in sorted({r['source'] for r in rr})];domains[d]=dict(sources=len(ds),gain=float(np.mean(ds)),positive=int(sum(v>0 for v in ds)),degraded=int(sum(v<0 for v in ds)))
 return dict(domains=domains,path_gain=float(np.mean([r['new']-r['tree'] for r in pp])),paths=pp,records=rows)

def main():
 import fcntl
 lock=(ROOT/'RUN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/'DONE.json').exists();torch.set_num_threads(2);start=time.time()
 save('CONFIG.json',dict(mechanism='Candidate-local CLIP2x2 concept/retained and discarded mass/relation cues + existing visual169/context; absolute humanIoU forest + supervised conditional tree gate',concepts=CONCEPTS,pairs=PAIRS,semantic_GT=False,relationship_GT=False,source_split='Reuse original train-source fit/hold; hold split calibration and quality by source hash parity',new077_080_excluded=True,formal_inputs=False,trees=64,depth=18,min_leaf=32,source_sha256=sha(__file__)))
 data=prepare();fit=[q for q in data if q['role']=='fit'];cal=[q for q in data if q['role']=='hold' and int(hashlib.sha256(q['source'].encode()).hexdigest()[:8],16)%2==0];val=[q for q in data if q['role']=='hold' and int(hashlib.sha256(q['source'].encode()).hexdigest()[:8],16)%2!=0]
 totals={d:sum(q['domain']==d for q in fit) for d in ['fcdb','gnmc','video']};x=np.concatenate([q['x'] for q in fit]);y=np.concatenate([q['y'] for q in fit]);weights=np.concatenate([np.full(len(q['y']),1/3/totals[q['domain']]/len(q['y'])) for q in fit])
 save('PROGRESS.json',dict(stage='cross_domain_concept_absolute_IoU_training',samples=len(y),dimensions=x.shape[1],at=time.time()))
 model=ExtraTreesRegressor(n_estimators=64,max_depth=18,min_samples_leaf=32,max_features=.5,n_jobs=4,random_state=6503);model.fit(x,y,sample_weight=weights);joblib.dump(model,ROOT/'concept_forest.joblib',compress=3)
 # Gate learns expected IoU risk on disjoint calibration source families, not a
 # ratio constant or official-score-derived blend. Validation sources untouched.
 xx=np.concatenate([q['x'] for q in cal]);yy=np.concatenate([q['y'] for q in cal]);uu=np.concatenate([q['u'] for q in cal]);vv=model.predict(xx).astype('float32');mean=xx.mean(0);std=np.maximum(xx.std(0),.05)
 tx=torch.tensor((xx-mean)/std,device='cuda');ty=torch.tensor(yy,device='cuda');tu=torch.tensor(uu,device='cuda');tv=torch.tensor(vv,device='cuda');w=torch.zeros(tx.shape[1],device='cuda',requires_grad=True);b=torch.tensor(-1.,device='cuda',requires_grad=True);opt=torch.optim.Adam([w,b],lr=.005)
 for _ in range(160):
  g=torch.sigmoid(tx@w+b);pred=g*tv+(1-g)*tu;loss=(pred-ty).square().mean()+.01*w.square().mean();opt.zero_grad();loss.backward();opt.step()
 gw=w.detach().cpu()/torch.tensor(std);gb=b.detach().cpu()-gw@torch.tensor(mean);gate=dict(w=gw,b=gb)
 frozen=dict(model=model,gate=gate,config=json.loads((ROOT/'CONFIG.json').read_text()),input_dim=x.shape[1]);joblib.dump(frozen,ROOT/'decision.joblib',compress=3);save('BEFORE_EVAL_FREEZE.json',dict(checkpoint=str(ROOT/'decision.joblib'),sha256=sha(ROOT/'decision.joblib'),at=time.time()))
 metrics=evaluate(model,val,gate);save('DONE.json',dict(success=True,checkpoint=str(ROOT/'decision.joblib'),sha256=sha(ROOT/'decision.joblib'),metrics=metrics,seconds=time.time()-start,formal_inputs=False,quality_scope='public train-source disjoint gate calibration and validation; no final labels; not AIC GT'))

if __name__=='__main__':
 try:main()
 except BaseException:save('FAILED.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
