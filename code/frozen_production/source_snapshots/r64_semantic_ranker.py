"""R64 calibrated human-IoU semantic decision model, public-only training.

New capacity: frozen DINOv2 patch ROI, discarded-region, full-scene and
contrast features. Absolute IoU and source-local ranking are learned jointly.
The neural predictor may replace R60, or be calibrated with it on train-hold.
No old final/test, formal video, MOS or rewritten ratio supervision is read.
"""
from pathlib import Path
import sys,os,json,time,hashlib,traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'4')
import numpy as np,torch,joblib
from torch import nn
from torchvision.ops import roi_align
ROOT=Path(__file__).resolve().parent
OLD=Path('/root/autodl-fs/r60_20261022');R63=Path('/root/autodl-fs/r63_20261023')
sys.path.insert(0,str(OLD))
from r60_fcdb_train import features as tree_features
SEED=6401

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,q):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)

def candidate_features(patch,cls,boxes,raw,width,height,ratio,projection,tree):
    """Same explicit-ratio extractor used for training and fresh inference."""
    device='cuda';mu=projection['mean'].to(device);comp=projection['components'].to(device)
    patch=torch.as_tensor(np.array(patch),device=device,dtype=torch.float32)
    cls=torch.as_tensor(np.array(cls),device=device,dtype=torch.float32)
    v=(patch-mu)@comp;glob=v.mean((0,1));c=(cls-mu)@comp
    gh,gw=v.shape[:2];b=np.asarray(boxes,np.float32);n=len(b)
    bt=torch.tensor(np.c_[np.zeros(n),b*[gw/width,gh/height,gw/width,gh/height]],device=device,dtype=torch.float32)
    roi=roi_align(v.permute(2,0,1)[None],bt,output_size=(2,2),spatial_scale=1.,sampling_ratio=2,aligned=True).flatten(1)
    inside=roi.reshape(n,-1,4).mean(-1)
    area=torch.tensor(((b[:,2]-b[:,0])*(b[:,3]-b[:,1])/(width*height)).clip(0,.999),device=device)[:,None]
    outside=(glob[None]-area*inside)/(1-area).clamp_min(.05)
    outside=outside.clamp(-20,20)
    semantic=torch.cat([roi,inside-outside,outside,glob.expand(n,-1),c.expand(n,-1)],1).cpu().numpy()
    old=tree_features(raw);u=tree.predict(old).astype('float32')
    norm=b/np.array([width,height,width,height],np.float32);sz=norm[:,2:]-norm[:,:2];center=(norm[:,2:]+norm[:,:2])/2
    explicit=np.array([np.log(width/height),np.log(ratio[0]/ratio[1])],np.float32)
    geometry=np.c_[norm,center,sz,sz.prod(1),np.broadcast_to(explicit,(n,2)),u,u-u.max()]
    x=np.c_[semantic,old,geometry].astype('float32');assert np.isfinite(x).all()
    return x,u

class Decision(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.LayerNorm(256),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,x):return self.net(x).squeeze(-1).sigmoid()

def summarize(records):
    keys=['r60','semantic','selected','oracle'];means={k:float(np.mean([r[k] for r in records])) for k in keys}
    d=np.array([r['selected']-r['r60'] for r in records]);boot=d[np.random.default_rng(SEED).integers(len(d),size=(5000,len(d)))].mean(1)
    return dict(n=len(records),means=means,gain=float(d.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),positive_sources=int((d>0).sum()),records=records)

def main():
    import fcntl
    lock=(ROOT/'SEMANTIC_TRAIN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'SEMANTIC_TRAIN_DONE.json').exists():return
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    cfg=json.loads((ROOT/'SEMANTIC_FEATURES.json').read_text());assert cfg['formal_inputs'] is False
    rows=cfg['rows'];train=[r for r in rows if r['role']=='train'];dev=[r for r in rows if r['role']=='dev']
    hold=[r for r in train if int(hashlib.sha256(r['source'].encode()).hexdigest()[:8],16)%7==0]
    fit=[r for r in train if r not in hold]
    config=dict(seed=SEED,architecture='DINOv2-base frozen aspect-preserving patches -> train-only PCA128 -> ROI2x2 / discarded-region / contrast / scene / CLS + R60 visual156 + explicit ratio -> MLP256/128 sigmoid absoluteIoU',loss='MSE absolute human IoU + .02 source-list KL at fixed temperature .10',train_sources=len(fit),hold_sources=len(hold),dev_sources=len(dev),epochs=32,checkpoints=[8,16,32],candidate_budget=128,selection='train-hash-hold only; select expected true crop IoU, then calibrated mixture from fixed [0,.5,1] with R60; exposed FCDB80 dev read after frozen choice; no new final read',supervision='unchanged actual human native crop IoU; R63 extended candidates exact IoU to unchanged annotations; no transformed-ratio GT',formal_inputs=False,ratio_gt='native FCDB only, not AIC/video GT',tree_sha256=sha(OLD/'fcdb_ranker.joblib'),encoder_sha256=sha(ROOT/'ENCODER_READY.json'),source_sha256=sha(__file__))
    save('SEMANTIC_TRAIN_CONFIG.json',config)
    tree=joblib.load(OLD/'fcdb_ranker.joblib');tree.n_jobs=2
    projpath=ROOT/'dino_projection.pt'
    if not projpath.exists():
        chunks=[]
        for r in fit:
            z=np.load(r['file']);a=z['patch'].reshape(-1,768);ids=np.linspace(0,len(a)-1,min(32,len(a))).astype(int);chunks.append(a[ids].astype('float32'))
        x=torch.tensor(np.concatenate(chunks),device='cuda');mean=x.mean(0);_,_,components=torch.pca_lowrank(x-mean,q=128,center=False,niter=3)
        torch.save(dict(mean=mean.cpu(),components=components.cpu(),fit_sources=[r['source'] for r in fit],formal_inputs=False),projpath);del chunks,x
    projection=torch.load(projpath,weights_only=False);data={};started=time.time()
    for i,r in enumerate(rows):
        d=np.load(r['file']);oldpath=Path(r['parent_feature_file']);assert sha(oldpath)==r['parent_sha256']
        expanded=R63/('train_features' if r['role']=='train' else 'public_features')/(r['source']+'.npz')
        old=np.load(oldpath);z=np.load(expanded) if expanded.exists() else old
        ratio=z['ratio'] if 'ratio' in z.files else old['native_ratio'];b=z['boxes'];raw=z['x'].astype('float32');y=z['y']
        x,u=candidate_features(d['patch'],d['cls'],b,raw,int(d['width']),int(d['height']),ratio,projection,tree)
        if r in fit:
            # Sampling uses geometry and frozen utility only, never target IoU.
            chosen=list(dict.fromkeys(np.r_[np.linspace(0,len(y)-1,min(80,len(y))).astype(int),np.argsort(-u,kind='stable')[:48]].tolist()))
            if len(chosen)<128:chosen+=(np.linspace(0,len(y)-1,128-len(chosen)).astype(int)).tolist()
            chosen=np.array(chosen[:128]);x=x[chosen];y=y[chosen];u=u[chosen]
        data[r['source']]=dict(x=x,y=y.astype('float32'),u=u,ratio=ratio.tolist())
        if i%25==0:save('SEMANTIC_TRAIN_PROGRESS.json',dict(stage='candidate_semantic_pooling',completed=i+1,total=len(rows),heartbeat=time.time()))
    fitx=np.concatenate([data[r['source']]['x'] for r in fit]);mean=fitx.mean(0);std=np.maximum(fitx.std(0),.05);del fitx
    xx=torch.tensor(np.stack([(data[r['source']]['x']-mean)/std for r in fit]),device='cuda');yy=torch.tensor(np.stack([data[r['source']]['y'] for r in fit]),device='cuda')
    model=Decision(xx.shape[-1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.02)
    candidates=[]
    for epoch in range(1,33):
        model.train();losses=[]
        for ids in torch.randperm(len(fit),device='cuda').split(12):
            p=model(xx[ids]);y=yy[ids]
            reg=(p-y).square().mean()
            rank=torch.nn.functional.kl_div(torch.log_softmax(p/.10,dim=1),torch.softmax(y/.10,dim=1),reduction='batchmean')
            loss=reg+.02*rank;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        save('SEMANTIC_TRAIN_PROGRESS.json',dict(stage='training',epoch=epoch,total_epochs=32,loss=float(np.mean(losses)),heartbeat=time.time(),elapsed_seconds=time.time()-started))
        print('TRAIN',epoch,float(np.mean(losses)),flush=True)
        if epoch in [8,16,32]:
            model.eval();allpred=[];ally=[];recs=[]
            with torch.inference_mode():
                for r in hold:
                    q=data[r['source']];p=model(torch.tensor((q['x']-mean)/std,device='cuda')).cpu().numpy();allpred.append(p);ally.append(q['y']);recs.append((r,q,p))
            # Source-balanced affine calibration keeps utility in IoU units.
            weights=np.concatenate([np.full(len(y),1/len(y)) for y in ally]);a=np.concatenate(allpred);target=np.concatenate(ally)
            design=np.c_[a,np.ones(len(a))];coef=np.linalg.lstsq(design*np.sqrt(weights[:,None]),target*np.sqrt(weights),rcond=None)[0];coef[0]=max(.01,float(coef[0]))
            options=[]
            for alpha in [0.,.5,1.]:
                vals=[float(q['y'][np.argmax(alpha*np.clip(coef[0]*p+coef[1],0,1)+(1-alpha)*q['u'])]) for _,q,p in recs]
                options.append(dict(alpha=alpha,selected_iou=float(np.mean(vals))))
            best=max(options,key=lambda q:q['selected_iou'])
            path=ROOT/f'semantic_epoch{epoch:02d}.pt'
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,input_dim=xx.shape[-1],epoch=epoch,calibration=coef,alpha=best['alpha'],config=config,projection_sha256=sha(projpath)),path)
            candidates.append(dict(epoch=epoch,path=str(path),hold_options=options,selected_alpha=best['alpha'],hold_iou=best['selected_iou'],sha256=sha(path)))
            save('SEMANTIC_HOLD_CHECKPOINTS.json',dict(checkpoints=candidates,formal_inputs=False,not_new_blind=True))
    best=max(candidates,key=lambda q:q['hold_iou']);ck=torch.load(best['path'],weights_only=False);model.load_state_dict(ck['model']);model.eval()
    frozen=ROOT/'semantic_decision.pt';torch.save(ck,frozen)
    save('SEMANTIC_BEFORE_DEV_FREEZE.json',dict(selected=best,checkpoint_sha256=sha(frozen),projection_sha256=sha(projpath),at=time.time()))
    records=[]
    with torch.inference_mode():
        for r in dev:
            q=data[r['source']];p=model(torch.tensor((q['x']-mean)/std,device='cuda')).cpu().numpy();p=np.clip(ck['calibration'][0]*p+ck['calibration'][1],0,1);s=ck['alpha']*p+(1-ck['alpha'])*q['u'];y=q['y']
            records.append(dict(source=r['source'],r60=float(y[q['u'].argmax()]),semantic=float(y[p.argmax()]),selected=float(y[s.argmax()]),oracle=float(y.max()),native_ratio=q['ratio']))
    summary=summarize(records);save('SEMANTIC_DEV_EVAL.json',summary)
    save('SEMANTIC_TRAIN_DONE.json',dict(success=True,checkpoint=str(frozen),sha256=sha(frozen),parameters=sum(p.numel() for p in model.parameters()),alpha=ck['alpha'],selected_epoch=ck['epoch'],dev={k:v for k,v in summary.items() if k!='records'},accepted_for_video_test=ck['alpha']>0 and summary['gain']>0,formal_production_authorized_by_this_script=False,formal_inputs=False,baseline='96tree on identical available expanded candidate pool; r60 metric key is not a formal R60 reproduction',limitation='static native-ratio image development, not evidence for full video path or official improvement'))

class Predictor:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.ck=torch.load(self.root/'semantic_decision.pt',map_location='cpu',weights_only=False)
        self.projection=torch.load(self.root/'dino_projection.pt',map_location='cpu',weights_only=False)
        assert sha(self.root/'dino_projection.pt')==self.ck['projection_sha256']
        self.model=Decision(self.ck['input_dim']).cuda().eval();self.model.load_state_dict(self.ck['model'])
        self.tree=joblib.load(OLD/'fcdb_ranker.joblib');self.tree.n_jobs=2
    def predict(self,patch,cls,boxes,raw,width,height,ratio):
        x,u=candidate_features(patch,cls,boxes,raw,width,height,ratio,self.projection,self.tree)
        with torch.inference_mode():p=self.model(torch.tensor((x-self.ck['mean'])/self.ck['std'],device='cuda')).cpu().numpy()
        a,b=self.ck['calibration'];p=np.clip(a*p+b,0,1);alpha=self.ck['alpha']
        return (alpha*p+(1-alpha)*u).astype('float32')

if __name__=='__main__':
    try:main()
    except BaseException:save('SEMANTIC_TRAIN_FAILED.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
