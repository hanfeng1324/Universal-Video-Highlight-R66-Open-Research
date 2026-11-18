"""Public crop-IoU supervision with soft CLIP scene routing and candidate context.

Scene/relationship scores are frozen-model features, never human labels.
Formal media and exposed independent video-final labels are never loaded.
"""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,zipfile,io
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
R=Path('/root/autodl-fs/r64_20261023');RUN=Path(__file__).resolve().parent
sys.path[:0]=[str(R),'/root/autodl-fs/r59_20261021','/root/autodl-fs/r60_20261022','/root/autodl-fs/r63_20261023']
import numpy as np,torch,joblib
from torch import nn
from PIL import Image
from transformers import CLIPModel,CLIPTokenizer,CLIPImageProcessor
from r64_semantic_ranker import candidate_features
from r64_video_context import temporal_features
from r59_model import path_optimize

PROMPTS=[
 ['a person talking to camera','a portrait of a single person'],
 ['people in a scenic landscape','a person and the surrounding environment'],
 ['an animal as the main subject','a close photograph of wildlife'],
 ['people interacting with each other','a person interacting with animals','animals interacting with each other'],
 ['a landscape with no people','a natural phenomenon in nature'],
 ['a sports action involving a ball and a goal','athletes interacting during a sporting event']]
NAMES=['person_talking_single','person_environment','animal','interaction','landscape_nature','sports_action']

def save(n,q):
 p=RUN/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_text(json.dumps(q,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()));t.replace(p)

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Scenes:
 def __init__(self):
  base=Path('/root/autodl-fs/r57_20261021/clone_essentials/root/r53_autonomous_20261020')
  loc=base/'aic_highlight/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
  self.model=CLIPModel.from_pretrained(loc,local_files_only=True).cuda().eval().requires_grad_(False)
  self.processor=CLIPImageProcessor.from_pretrained(loc,local_files_only=True)
  tok=CLIPTokenizer.from_pretrained(base/'assets/r22_20261016/clip_tokenizer',local_files_only=True)
  tt=tok([s for g in PROMPTS for s in g],padding=True,return_tensors='pt').to('cuda')
  with torch.inference_mode():v=self.model.get_text_features(**tt)
  if not isinstance(v,torch.Tensor):v=v.pooler_output
  v=nn.functional.normalize(v,dim=-1);parts=[];i=0
  for g in PROMPTS:parts.append(nn.functional.normalize(v[i:i+len(g)].mean(0),dim=-1));i+=len(g)
  self.text=torch.stack(parts)
 def __call__(self,im):
  with torch.inference_mode():
   v=self.model.get_image_features(**self.processor(images=im.convert('RGB'),return_tensors='pt').to('cuda'))
   if not isinstance(v,torch.Tensor):v=v.pooler_output
   s=(nn.functional.normalize(v,dim=-1)@self.text.T*10).softmax(-1)[0]
  return s.cpu().numpy().astype('float32')

class SceneQuality(nn.Module):
 def __init__(self,d=1223):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.LayerNorm(128))
  self.context=nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
  self.head=nn.Sequential(nn.Linear(256,96),nn.GELU(),nn.Linear(96,6))
  self.gate=nn.Sequential(nn.Linear(6+13,32),nn.GELU(),nn.Linear(32,1))
 def forward(self,x,u,scene):
  h=self.encoder(x);context,_=self.context(h,h,h,need_weights=False)
  expert=self.head(torch.cat([h,context],-1)).sigmoid()
  neural=(expert*scene[:,None,:]).sum(-1)
  gate=self.gate(torch.cat([scene[:,None,:].expand(-1,x.shape[1],-1),x[:,:,-43:-30]],-1)).sigmoid().squeeze(-1)
  return gate*neural+(1-gate)*u,gate,expert

def part(source):return 'hold' if int(hashlib.sha256(('r65_group_'+source).encode()).hexdigest()[:8],16)%5==0 else 'fit'

def build():
 scene=Scenes();groups=[];cache=RUN/'features';cache.mkdir(exist_ok=True)
 tree=joblib.load('/root/autodl-fs/r60_20261022/fcdb_ranker.joblib');tree.n_jobs=1
 projection=torch.load(R/'dino_projection.pt',weights_only=False)
 fzip=zipfile.ZipFile('/root/autodl-fs/r60_20261022/public_data/FCDB.zip');fn={Path(n).stem.split('_')[0]:n for n in fzip.namelist() if n.lower().endswith('.jpg')}
 gzip=zipfile.ZipFile(R/'GNMC/GNMC.zip');gn={Path(n).stem:n for n in gzip.namelist() if '/train/' in n and n.lower().endswith(('.jpg','.jpeg','.png')) and '__MACOSX' not in n}
 # Only original training sources enter either fitting or source-hold selection.
 manifests=[('fcdb',json.loads((R/'SEMANTIC_FEATURES.json').read_text())['rows']),('gnmc',json.loads((R/'GNMC_PHASE1_FEATURES.json').read_text())['rows'])]
 for domain,rows in manifests:
  rows=[r for r in rows if r['role']=='train']
  for i,r in enumerate(rows):
   source=r['source'];p=cache/(source+'.joblib')
   if p.exists():gg=joblib.load(p)
   else:
    if domain=='gnmc':
     z=np.load(r['file']);identity=source.split('_')[-1];s=scene(Image.open(io.BytesIO(gzip.read(gn[identity]))));gg=[]
     for j,tag in enumerate(z['tags']):
      lo,hi=z['parts'][j:j+2];ids=np.unique(np.linspace(lo,hi-1,min(128,hi-lo)).astype(int));x=z['x'][ids].astype('float32');gg.append(dict(source=source,domain=domain,role=part(source),ratio=str(tag),x=np.c_[x,np.zeros((len(x),30),np.float32)],u=z['u'][ids],y=z['y'][ids],scene=s))
    else:
     d=np.load(r['file']);old=np.load(r['parent_feature_file']);expanded=Path('/root/autodl-fs/r63_20261023/train_features')/(source+'.npz');z=np.load(expanded) if expanded.exists() else old
     b=z['boxes'];ids=np.unique(np.linspace(0,len(b)-1,min(128,len(b))).astype(int));ratio=z['ratio'] if 'ratio' in z.files else old['native_ratio']
     x,u=candidate_features(d['patch'],d['cls'],b[ids],z['x'][ids].astype('float32'),int(d['width']),int(d['height']),ratio,projection,tree)
     s=scene(Image.open(io.BytesIO(fzip.read(fn[source.removeprefix('fcdb_')]))));gg=[dict(source=source,domain=domain,role=part(source),ratio=str(ratio.tolist()),x=np.c_[x,np.zeros((len(x),30),np.float32)],u=u,y=z['y'][ids],scene=s)]
    joblib.dump(gg,p,compress=3)
   groups.extend(gg)
   if i%16==0:save('PROGRESS.json',dict(stage='CLIP_scene_and_content_features',domain=domain,processed=i+1,total=len(rows),groups=len(groups),at=time.time(),GPU=True))
 # Video context combines candidate content correspondence, geometry and tree utility.
 rows=json.loads(Path('/root/autodl-fs/r59_20261021/FEATURES_FULL.json').read_text())['rows'];rows=[r for r in rows if r['role']=='train' and r['kind']=='human_iou' and r['source'].startswith('retarget_')]
 for source in sorted({r['source'] for r in rows}):
  p=cache/(source+'.joblib')
  if p.exists():gg=joblib.load(p)
  else:
   rr=sorted([r for r in rows if r['source']==source],key=lambda r:r['file']);rr=[rr[k] for k in np.unique(np.linspace(0,len(rr)-1,32).astype(int))];parts=[[],[]]
   for r in rr:
    z=np.load(Path('/root/autodl-fs/r59_20261021')/r['file']);frame=int(z['frame']);d=np.load(R/'video_train_semantics'/source/f'{frame:06d}.npz');s=scene(Image.open(Path('/root/autodl-fs/r59_20261021/data/retarget')/source.split('_')[-1]/f'{frame:06d}.jpg'));ends=np.cumsum([0]+z['lengths'].tolist())
    for k,(lo,hi) in enumerate(zip(ends[:-1],ends[1:])):
     ids=np.unique(np.linspace(lo,hi-1,min(128,hi-lo)).astype(int));b=z['boxes'][ids];x,u=candidate_features(d['patch'],d['cls'],b,z['x'][ids].astype('float32'),int(z['width']),int(z['height']),[[1,3],[3,1]][k],projection,tree)
     parts[k].append(dict(source=source,domain='video',role=part(source),ratio=k,b=b,x=x,u=u,y=z['y'][ids],scene=s,frame=frame,w=int(z['width']),h=int(z['height'])))
   gg=[]
   for k,g in enumerate(parts):
    ctx=temporal_features([q['b'] for q in g],[q['u'] for q in g],[q['x'][:,:512].reshape(len(q['x']),128,4).mean(-1) for q in g],g[0]['w'],g[0]['h'])
    for q,c in zip(g,ctx):q['x']=np.c_[q['x'],c,np.ones(len(c),np.float32)];gg.append(q)
   joblib.dump(gg,p,compress=3)
  groups.extend(gg);save('PROGRESS.json',dict(stage='video_CLIP_semantics_context_features',source=source,groups=len(groups),at=time.time()))
 del scene;torch.cuda.empty_cache();return groups

def evaluate(model,data,mean,std):
 model.eval();records=[];paths={}
 with torch.inference_mode():
  for q in data:
   x=torch.tensor((q['x']-mean)/std,device='cuda')[None];u=torch.tensor(q['u'],device='cuda')[None];s=torch.tensor(q['scene'],device='cuda')[None];pred,gate,_=model(x,u,s);p=pred[0].cpu().numpy()
   records.append(dict(source=q['source'],domain=q['domain'],ratio=str(q['ratio']),scene_proxy=NAMES[int(q['scene'].argmax())],tree=float(q['y'][q['u'].argmax()]),new=float(q['y'][p.argmax()]),gate=float(gate.mean().cpu())))
   if q['domain']=='video':paths.setdefault((q['source'],q['ratio']),[]).append((q,p))
 path_rows=[]
 for key,rows in paths.items():
  rows.sort(key=lambda z:z[0]['frame']);qs=[x[0] for x in rows];bb=[q['b'] for q in qs];ff=[q['frame'] for q in qs];w,h=qs[0]['w'],qs[0]['h'];a=path_optimize(bb,[q['u'] for q in qs],ff,[],w,h);b=path_optimize(bb,[x[1] for x in rows],ff,[],w,h)
  path_rows.append(dict(source=key[0],ratio=key[1],tree=float(np.mean([q['y'][j] for q,j in zip(qs,a)])),new=float(np.mean([q['y'][j] for q,j in zip(qs,b)]))))
 result={'records':records,'paths':path_rows,'domains':{}}
 for domain in ['fcdb','gnmc','video']:
  rr=[q for q in records if q['domain']==domain];src={s:np.mean([r['new']-r['tree'] for r in rr if r['source']==s]) for s in {r['source'] for r in rr}}
  result['domains'][domain]=dict(sources=len(src),gain=float(np.mean(list(src.values()))) if src else None,positive=int(sum(v>0 for v in src.values())),degraded=int(sum(v<0 for v in src.values())))
 result['path_gain']=float(np.mean([q['new']-q['tree'] for q in path_rows])) if path_rows else None
 result['scene_strata']={s:dict(groups=len(rr),gain=float(np.mean([r['new']-r['tree'] for r in rr]))) for s in NAMES if (rr:=[r for r in records if r['scene_proxy']==s])}
 return result

def main():
 import fcntl
 lock=(RUN/'RUN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (RUN/'DONE.json').exists();torch.set_num_threads(2);torch.manual_seed(6501);np.random.seed(6501);start=time.time()
 save('CONFIG.json',dict(architecture='Frozen CLIP six soft overlapping scene features + DINO candidate ROI/discard contrast + GAIC +29 temporal matching -> candidate multihead attention +6 absoluteIoU experts + humanIoU-learned tree/expert gate',prompts=PROMPTS,scene_human_GT=False,relationship_human_GT=False,supervision='unchanged genuine FCDB/GNMC/Retarget cropIoU only; no MOS or formal labels',split='original train sources hash20percent hold; no old061-076 final for fit or selection',loss='three-domain balanced absoluteIoU MSE + list KL + scene worst-group penalty',epochs=32,checkpoints=[8,16,32],candidate_budget=128,formal_inputs=False,parameter_limit=9000000000))
 data=build();fit=[q for q in data if q['role']=='fit'];hold=[q for q in data if q['role']=='hold'];mean=np.concatenate([q['x'] for q in fit]).mean(0);std=np.maximum(np.concatenate([q['x'] for q in fit]).std(0),.05)
 save('DATA_READY.json',dict(groups=len(data),fit_groups=len(fit),hold_groups=len(hold),fit_sources={d:len({q['source'] for q in fit if q['domain']==d}) for d in ['fcdb','gnmc','video']},processed_candidates=sum(len(q['y']) for q in data),at=time.time()))
 model=SceneQuality(len(mean)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.04);bydomain=[[q for q in fit if q['domain']==d] for d in ['fcdb','gnmc','video']];rng=np.random.default_rng(6501);choices=[];start_epoch=0
 if (RUN/'last.pt').exists():
  ck=torch.load(RUN/'last.pt',map_location='cpu',weights_only=False);assert np.array_equal(ck['mean'],mean) and np.array_equal(ck['std'],std)
  model.load_state_dict(ck['model']);start_epoch=int(ck['epoch'])
  if 'optimizer' in ck:opt.load_state_dict(ck['optimizer'])
  if 'numpy_rng' in ck:rng.bit_generator.state=ck['numpy_rng']
  if 'torch_rng' in ck:torch.set_rng_state(ck['torch_rng'])
  if 'cuda_rng' in ck:torch.cuda.set_rng_state_all(ck['cuda_rng'])
  if (RUN/'CHECKPOINTS.json').exists():choices=json.loads((RUN/'CHECKPOINTS.json').read_text())['choices']
  save('RESUMED_CHECKPOINT.json',dict(epoch=start_epoch,optimizer_restored='optimizer' in ck,at=time.time()))
 for epoch in range(start_epoch+1,33):
  model.train();losses=[]
  for _ in range(160):
   batch=[g[int(rng.integers(len(g)))] for g in bydomain];n=min(96,min(len(q['y']) for q in batch));ids=[rng.choice(len(q['y']),n,replace=False) for q in batch]
   x=torch.tensor(np.stack([(q['x'][i]-mean)/std for q,i in zip(batch,ids)]),device='cuda');y=torch.tensor(np.stack([q['y'][i] for q,i in zip(batch,ids)]),device='cuda');u=torch.tensor(np.stack([q['u'][i] for q,i in zip(batch,ids)]),device='cuda');scene=torch.tensor(np.stack([q['scene'] for q in batch]),device='cuda')
   pred,gate,experts=model(x,u,scene);mse=(pred-y).square().mean(1);loss=mse.mean()+.15*mse.max()+.02*nn.functional.kl_div(nn.functional.log_softmax(pred/.1,dim=1),nn.functional.softmax(y/.1,dim=1),reduction='batchmean')
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
  checkpoint=RUN/'last.pt';pending=RUN/'last.pending.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),mean=mean,std=std,epoch=epoch,input_dim=len(mean),prompts=PROMPTS,scene_labels_are_predictions=True),pending);pending.replace(checkpoint)
  save('PROGRESS.json',dict(stage='scene_semantic_IoU_training',epoch=epoch,total_epochs=32,loss=float(np.mean(losses)),groups_processed=epoch*160*3,candidates_processed=epoch*160*3*96,checkpoint=str(checkpoint),seconds=time.time()-start,at=time.time()))
  if epoch in [8,16,32]:
   metrics=evaluate(model,hold,mean,std);path=RUN/f'epoch{epoch:02d}.pt';torch.save(dict(model=model.state_dict(),mean=mean,std=std,epoch=epoch,input_dim=len(mean),prompts=PROMPTS),path);choices.append(dict(epoch=epoch,path=str(path),sha256=sha(path),metrics=metrics));save('CHECKPOINTS.json',dict(choices=choices,formal_inputs=False))
 best=max(choices,key=lambda q:np.mean([v['gain'] for v in q['metrics']['domains'].values()])+q['metrics']['path_gain']);save('DONE.json',dict(success=True,selected=best,seconds=time.time()-start,parameters=sum(p.numel() for p in model.parameters()),formal_inputs=False,quality_scope='train-source hash hold; scene strata proxy not human sceneGT; no independent final or AIC quality claim'))

if __name__=='__main__':
 try:main()
 except BaseException:save('FAILED.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
