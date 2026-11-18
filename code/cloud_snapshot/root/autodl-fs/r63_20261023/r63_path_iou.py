"""Train a path-context IoU residual on existing public training trajectories."""
from pathlib import Path
import os,sys,json,time,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
import r63_crop_search as core
from r59_model import path_optimize
R=Path(__file__).resolve().parent

def save(n,q):
    p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)

def context_features(boxes,scores,w,h):
    out=[];scale=np.array([w,h,w,h])
    for t,(bb,u) in enumerate(zip(boxes,scores)):
        b=bb/scale;c=(b[:,:2]+b[:,2:])/2;size=np.maximum(b[:,2:]-b[:,:2],1e-6)
        f=[u[:,None],(u-u.max())[:,None],c,size,(size[:,0]*size[:,1])[:,None]]
        for j in [max(0,t-1),min(len(boxes)-1,t+1)]:
            p=boxes[j]/scale;pc=(p[:,:2]+p[:,2:])/2;ps=np.maximum(p[:,2:]-p[:,:2],1e-6)
            dc=np.square(c[:,None]-pc[None]).sum(-1);ds=np.square(np.log(size[:,None]/ps[None])).sum(-1)
            z=scores[j][None]-.18*dc-.035*ds;idx=z.argmax(1);a=np.arange(len(b))
            best=int(scores[j].argmax())
            f.extend([(z[a,idx]-scores[j].max())[:,None],scores[j][idx,None],
                      dc[a,idx,None],ds[a,idx,None],dc[:,best,None],ds[:,best,None]])
        out.append(np.concatenate(f,axis=1).astype(np.float32))
    return out

class PathAdapter:
    def __init__(self,path):self.model=joblib.load(path)
    def utilities(self,boxes,scores,w,h):
        xx=context_features(boxes,scores,w,h)
        return [s+self.model.predict(x).astype(np.float32) for s,x in zip(scores,xx)]

def main():
    if (R/'PATH_TRAIN_DONE.json').exists():print('PATH_ALREADY_DONE');return
    root=Path('/root/autodl-fs/r59_20261021');cfg=json.loads((root/'FEATURES_FULL.json').read_text())
    rows=[r for r in cfg['rows'] if r['role']=='train' and r['kind']=='human_iou' and r['source'].startswith('retarget_')]
    sources=sorted({r['source'] for r in rows});assert len(sources)==20
    config=dict(mechanism='learn human IoU residual from adjacent candidate support, geometry and frozen96-tree utility',
                sources=sources,anchors_per_source=32,original_penalties=[.18,.035],
                max_iter=64,max_leaf_nodes=7,max_depth=3,learning_rate=.05,l2_regularization=10,
                min_samples_leaf=128,early_stopping=False,random_state=6302,
                supervision='six native1:3/3:1 human crops at sampled frames; not dense AIC or identity GT',
                no_formal_data=True,tree_sha256=core.sha(core.TREE),code_sha256=core.sha(__file__))
    save('PATH_TRAIN_CONFIG.json',config);tree=core.load_tree();xx=[];yy=[];ww=[];started=time.time()
    for i,source in enumerate(sources):
        rr=sorted([r for r in rows if r['source']==source],key=lambda r:r['file']);rr=[rr[j] for j in np.unique(np.linspace(0,len(rr)-1,32).astype(int))]
        groups=[[],[]]
        for row in rr:
            p=root/row['file'];assert core.sha(p)==row['sha256'];z=np.load(p);parts=np.cumsum([0]+z['lengths'].tolist());u=core.score(tree,z['x'].astype(np.float32))
            for k,(lo,hi) in enumerate(zip(parts[:-1],parts[1:])):
                groups[k].append(dict(b=z['boxes'][lo:hi],u=u[lo:hi],y=z['y'][lo:hi],w=int(z['width']),h=int(z['height']),frame=int(z['frame'])))
        for gg in groups:
            ff=context_features([g['b'] for g in gg],[g['u'] for g in gg],gg[0]['w'],gg[0]['h'])
            for f,g in zip(ff,gg):xx.append(f);yy.append(g['y']-g['u']);ww.append(np.full(len(f),1/len(f)/len(gg)))
        save('PATH_TRAIN_PROGRESS.json',dict(completed=i+1,total=20,heartbeat=time.time(),candidates=sum(len(y) for y in yy),elapsed_seconds=time.time()-started))
        print('R63_PATH_TRAIN',i+1,20,flush=True)
    x=np.concatenate(xx);y=np.concatenate(yy);weights=np.concatenate(ww);weights*=len(weights)/weights.sum()
    model=HistGradientBoostingRegressor(**{k:config[k] for k in ['max_iter','max_leaf_nodes','max_depth','learning_rate','l2_regularization','min_samples_leaf','early_stopping','random_state']})
    model.fit(x,y,sample_weight=weights);ck=R/'r63_path_context.joblib';tmp=ck.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(ck)
    q=dict(success=True,checkpoint=str(ck),checkpoint_sha256=core.sha(ck),config=config,training_sources=20,train_candidates=len(y),
           feature_dim=x.shape[1],parameters=sum(len(p.nodes)*8 for it in model._predictors for p in it),
           source_sha256=core.sha(__file__),elapsed_seconds=time.time()-started,public_validation='PENDING_NEW_SOURCES',formal_started=False)
    save('PATH_TRAIN_DONE.json',q);print(json.dumps(q),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:save('PATH_TRAIN_FAILED.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
