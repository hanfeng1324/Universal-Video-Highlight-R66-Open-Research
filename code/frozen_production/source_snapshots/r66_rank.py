"""Source-isolated scene-conditioned, listwise human crop-IoU learning.

Only original public training caches enter fit/calibration/selection. Exposed
independent evaluations and formal inputs never enter this program.
"""
from pathlib import Path
import os,sys,json,time,hashlib,io,tarfile,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import numpy as np,joblib,torch
import lightgbm as lgb
R=Path('/root/autodl-fs/r66_20261023');T=Path('/root/autodl-tmp/r66_20261023')
A=Path('/root/autodl-fs/r65_20261023/A2')
sys.path.insert(0,'/root/autodl-fs/r59_20261021')
from r59_model import path_optimize

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,q):
 p=T/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(q,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()));tmp.replace(p)
def caches():
 audit=json.loads((A/'TRAIN_SOURCE_SPLIT_AUDIT.json').read_text());assert audit['success'] and not audit['source_overlap'];allowed=set().union(*[set(v) for v in audit['source_roles'].values()])
 seen=set();out=[]
 for name in ['FEATURE_SNAPSHOT.tar','FEATURE_COMPLETION_DELTA35.tar']:
  with tarfile.open(A/name) as tar:
   for m in tar:
    if not m.isfile() or not m.name.endswith('.joblib'):continue
    key=Path(m.name).name;assert key not in seen;seen.add(key)
    gg=joblib.load(io.BytesIO(tar.extractfile(m).read()))
    for q in gg:
     assert q['x'].shape[1]==405 and np.isfinite(q['x']).all() and q['role'] in ['fit','hold']
     assert q['domain'] in ['fcdb','gnmc','video']
     if q['domain']=='video':assert 3<=int(q['source'].split('_')[-1])<=22
     assert q['source'] in allowed
    out.extend(gg)
    if len(seen)%128==0:save('LOAD_PROGRESS.json',dict(files=len(seen),groups=len(out),at=time.time()))
 assert len(seen)==1767
 return out
def features(x):
 # Scene scores are predicted features, not semantic or relationship GT.
 s=x[:,369:375];v=np.c_[x[:,156:169],x[:,337:369]]
 return np.c_[x,(s[:,:,None]*v[:,None,:]).reshape(len(x),-1)].astype('float32')
def parent(q,b):
 g=1/(1+np.exp(-np.clip(q['x']@b['gate']['w'].numpy()+float(b['gate']['b']),-40,40)))
 return (g*b['model'].predict(q['x'])+(1-g)*q['u']).astype('float32')
def gate_features(q,p,a):
 x=q['x'];return np.c_[x[:,156:169],x[:,369:375],x[:,375:],p,a,p-a,np.broadcast_to([p.std(),a.std(),p.max()-np.median(p),a.max()-np.median(a)],(len(p),4))].astype('float32')
def predict(q,bundle):
 p=bundle['model'].predict(features(q['x']),num_threads=1);a=parent(q,bundle['parent']);z=gate_features(q,p,a)
 g=1/(1+np.exp(-np.clip(z@bundle['gate']['w']+bundle['gate']['b'],-40,40)))
 return (g*p+(1-g)*a).astype('float32')
def assess(groups,bundle):
 rows=[];paths={}
 for q in groups:
  p=predict(q,bundle);a=parent(q,bundle['parent']);rows.append(dict(source=q['source'],domain=q['domain'],new=float(q['y'][p.argmax()]),a2=float(q['y'][a.argmax()]),r63=float(q['y'][q['u'].argmax()])))
  if q['domain']=='video':paths.setdefault((q['source'],str(q['ratio'])),[]).append((q,p,a))
 pp=[]
 for key,rr in paths.items():
  rr.sort(key=lambda r:r[0]['frame']);qs=[r[0] for r in rr];bb=[q['b'] for q in qs];ff=[q['frame'] for q in qs];w,h=qs[0]['w'],qs[0]['h'];rec=dict(source=key[0],ratio=key[1])
  for name,ss in [('new',[r[1] for r in rr]),('a2',[r[2] for r in rr]),('r63',[q['u'] for q in qs])]:
   choice=path_optimize(bb,ss,ff,[],w,h);rec[name]=float(np.mean([q['y'][i] for q,i in zip(qs,choice)]))
  pp.append(rec)
 domains={}
 for d in ['fcdb','gnmc','video']:
  rr=[r for r in rows if r['domain']==d];ds=[float(np.mean([r['new']-r['a2'] for r in rr if r['source']==s])) for s in sorted({r['source'] for r in rr})];domains[d]=dict(sources=len(ds),gain_vs_a2=float(np.mean(ds)),positive=sum(v>0 for v in ds),negative=sum(v<0 for v in ds))
 gain=float(np.mean([r['new']-r['a2'] for r in pp]));return dict(domains=domains,path_gain_vs_a2=gain,paths=pp,records=rows)
def calibrate(model,groups,b):
 ps=[model.predict(features(q['x']),num_threads=2) for q in groups];aa=[parent(q,b) for q in groups];z=np.concatenate([gate_features(q,p,a) for q,p,a in zip(groups,ps,aa)]);y=np.concatenate([q['y'] for q in groups]);p=np.concatenate(ps);a=np.concatenate(aa)
 mean=z.mean(0);std=np.maximum(z.std(0),.05);zz=torch.tensor((z-mean)/std);yy=torch.tensor(y);pp=torch.tensor(p);aa=torch.tensor(a);w=torch.zeros(z.shape[1],requires_grad=True);bias=torch.tensor(-1.,requires_grad=True);opt=torch.optim.Adam([w,bias],lr=.01)
 # Calibration families are disjoint from both fit and model-selection families.
 counts={d:len({q['source'] for q in groups if q['domain']==d}) for d in ['fcdb','gnmc','video']};gs={s:sum(q['source']==s for q in groups) for s in {q['source'] for q in groups}};wt=torch.tensor(np.concatenate([np.full(len(q['y']),1/(counts[q['domain']]*gs[q['source']]*len(q['y']))) for q in groups]),dtype=torch.float32);wt/=wt.sum()
 for _ in range(200):
  g=torch.sigmoid(zz@w+bias);pred=g*pp+(1-g)*aa;loss=((pred-yy).square()*wt).sum()+.002*w.square().mean();opt.zero_grad();loss.backward();opt.step()
 ww=w.detach().numpy()/std;return dict(w=ww,b=float(bias.detach())-float(ww@mean))
def main():
 import fcntl
 T.mkdir(exist_ok=True);lock=(T/'RANK.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (T/'DONE.json').exists();torch.set_num_threads(2);torch.manual_seed(6601);start=time.time()
 cfg=dict(mechanism='Soft scene x geometry/concept-retention interactions; group-listwise human-IoU boosting with calibrated absolute-IoU; source-disjoint supervised parent-A2 risk gate',parent_sha256=sha(A/'decision.joblib'),source_sha256=sha(__file__),formal_inputs=False,semantic_human_GT=False,relationship_human_GT=False,objective='source/domain-balanced MSE + continuous-IoU softmax listwise loss',temperature=.08,listwise_weight=.02,checkpoints=[64,128,256],trees_max=256,num_leaves=31,split='original R65 fit sources; hold even=calibration odd=selection; new sources excluded',new_source_ids=[85,86,87,88])
 save('CONFIG.json',cfg);data=caches();fit=[q for q in data if q['role']=='fit'];cal=[q for q in data if q['role']=='hold' and int(hashlib.sha256(q['source'].encode()).hexdigest()[:8],16)%2==0];val=[q for q in data if q['role']=='hold' and int(hashlib.sha256(q['source'].encode()).hexdigest()[:8],16)%2!=0];assert not ({q['source'] for q in fit}&{q['source'] for q in cal}) and not ({q['source'] for q in val}&{q['source'] for q in cal})
 par=joblib.load(A/'decision.joblib');par['model'].n_jobs=1
 x=np.concatenate([features(q['x']) for q in fit]);y=np.concatenate([q['y'] for q in fit]);lengths=np.array([len(q['y']) for q in fit]);starts=np.r_[0,np.cumsum(lengths)[:-1]];ids=np.repeat(np.arange(len(fit)),lengths)
 dc={d:len({q['source'] for q in fit if q['domain']==d}) for d in ['fcdb','gnmc','video']};sc={s:sum(q['source']==s for q in fit) for s in {q['source'] for q in fit}};gw=np.array([1/(dc[q['domain']]*sc[q['source']]) for q in fit]);gw/=gw.mean();cw=np.repeat(gw/lengths,lengths)*128
 target=np.exp((y-np.maximum.reduceat(y,starts)[ids])/.08);target/=np.add.reduceat(target,starts)[ids]
 save('DATA_READY.json',dict(at=time.time(),fit_sources=dc,fit_groups=len(fit),cal_sources=len({q['source'] for q in cal}),val_sources=len({q['source'] for q in val}),samples=len(y),dimensions=x.shape[1],data_role='public_original_training',cuda_used=False))
 ds=lgb.Dataset(x,label=y,free_raw_data=False)
 def objective(pred,train):
  prob=np.exp(np.clip((pred-np.maximum.reduceat(pred,starts)[ids])/.08,-80,0));prob/=np.add.reduceat(prob,starts)[ids]
  grad=(pred-y+.02*(prob-target)/.08)*cw
  hess=(1+.02*prob*(1-prob)/(.08**2))*cw
  return grad,hess
 params=dict(objective=objective,num_leaves=31,learning_rate=.04,min_data_in_leaf=128,feature_fraction=.85,lambda_l2=8.,max_bin=127,num_threads=4,seed=6601,verbosity=-1)
 choices=[]
 def callback(env):
  epoch=env.iteration+1
  if epoch%16==0:save('PROGRESS.json',dict(stage='scene_conditioned_listwise_IoU_training',iteration=epoch,total=256,seconds=time.time()-start,at=time.time()))
  if epoch not in [64,128,256]:return
  text=env.model.model_to_string(num_iteration=epoch);model=lgb.Booster(model_str=text);gate=calibrate(model,cal,par);bundle=dict(model=model,gate=gate,parent=par,input_dim=405,config=cfg,iterations=epoch);path=T/f'epoch{epoch:03d}.joblib';joblib.dump(bundle,path,compress=3);metrics=assess(val,bundle);score=float(np.mean([d['gain_vs_a2'] for d in metrics['domains'].values()]))+metrics['path_gain_vs_a2'];choices.append(dict(iteration=epoch,path=str(path),sha256=sha(path),score=score,metrics=metrics));save('CHECKPOINTS.json',dict(choices=choices,at=time.time()))
 callback.order=10;lgb.train(params,ds,num_boost_round=256,callbacks=[callback]);best=max(choices,key=lambda q:q['score']);selected=T/'decision.joblib';selected.write_bytes(Path(best['path']).read_bytes());save('BEFORE_NEW_EVAL_FREEZE.json',dict(checkpoint=str(selected),sha256=sha(selected),selection='original training hold only',at=time.time()));save('DONE.json',dict(success=True,checkpoint=str(selected),sha256=sha(selected),selected=best,seconds=time.time()-start,formal_inputs=False,official_score=None))
 # Essential immutable backup only; no material packaging.
 for n in ['CONFIG.json','DATA_READY.json','DONE.json','BEFORE_NEW_EVAL_FREEZE.json','decision.joblib']:
  (R/'RANK').mkdir(exist_ok=True);__import__('shutil').copyfile(T/n,R/'RANK'/n)
if __name__=='__main__':
 try:main()
 except BaseException:save('FAILED.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
