"""Fixed IoU-only multiratio candidate ranker, no MOS/pseudo crop supervision."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,time,hashlib
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,q):
    p=ROOT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(q,indent=2),'utf8');t.replace(p)

def features(x):
    v=x[:,:1296].reshape(-1,16,3,3,3,3).mean((3,5)).reshape(-1,144)
    # Exclude every saliency/MOS descriptor and uniform-map layout. Inputs
    # remain numerically identical when real UNISAL is present at inference.
    g=x[:,1296:]
    return np.c_[v,g[:,:10],g[:,17:19]].astype('float32')

def evaluate(model,rows):
    records=[]
    for r in rows:
        p=ROOT/r['file'];assert sha(p)==r['sha256'];z=np.load(p);y=z['y'];s=model.predict(features(z['x'].astype('float32')))
        k=int(s.argmax());base=int(z['baseline']);g=int(z['gaic'].argmax());b=z['boxes'];w,h=int(z['width']),int(z['height'])
        records.append(dict(source=r['source'],selected=float(y[k]),center_max=float(y[base]),oracle=float(y.max()),gaic=float(y[g]),native_ratio=z['native_ratio'].tolist(),
                            selected_width_fraction=float((b[k,2]-b[k,0])/w),baseline_width_fraction=float((b[base,2]-b[base,0])/w)))
    gains=np.array([r['selected']-r['center_max'] for r in records]);rng=np.random.default_rng(6023)
    ci=np.quantile(np.mean(gains[rng.integers(len(gains),size=(5000,len(gains)))],axis=1),[.025,.975]).tolist()
    return dict(n=len(records),selected=float(np.mean([r['selected'] for r in records])),center_max=float(np.mean([r['center_max'] for r in records])),oracle=float(np.mean([r['oracle'] for r in records])),gaic=float(np.mean([r['gaic'] for r in records])),gain=float(gains.mean()),gain_bootstrap95=ci,positive_sources=int((gains>0).sum()),records=records,
                baseline_is_smartvidcrop=False,actual_human_native_crop_GT=True,target_ratio_GT=False,formal_inputs=False)

def retarget_diagnostic(model):
    import resource
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    # Reuse actual frozen human IoU labels; no fitting or checkpoint selection.
    _,_,_,rows=joblib.load(ROOT/'PUBLIC_IOU_FEATURES.joblib',mmap_mode='r')
    rec=[]
    for r in rows:
        pred=model.predict(features(r['raw_x']))
        for k,(a,b) in enumerate(zip(r['parts'][:-1],r['parts'][1:])):
            y=r['y'][a:b];j=int(pred[a:b].argmax())
            rec.append(dict(source=r['source'],ratio=k,selected=float(y[j]),smartvid=float(y[int(r['baseline'][k])]),oracle=float(y.max())))
    per=[]
    for s in sorted({r['source'] for r in rec}):
        rr=[r for r in rec if r['source']==s];per.append(dict(source=s,**{k:float(np.mean([r[k] for r in rr])) for k in ['selected','smartvid','oracle']}))
    return dict(source_macro={k:float(np.mean([r[k] for r in per])) for k in ['selected','smartvid','oracle']},per_source=per,records=rec,dev_exposed=True,
                GT='native1:3/3:1 only; no competition-ratio claim',use='diagnosis only after fixed FCDB checkpoint; never train or retune on these labels')

def main():
    if (ROOT/'FCDB_TRAIN_DONE.json').exists():print('ALREADY_COMPLETE');return
    cfg=read(ROOT/'FEATURES_FCDB.json');assert cfg['formal_inputs'] is False and not cfg['smoke_limit']
    train=[r for r in cfg['rows'] if r['role']=='train'];dev=[r for r in cfg['rows'] if r['role']=='dev'];test=[r for r in cfg['rows'] if r['role']=='test']
    assert len(train)>500 and len(dev)>30 and len(test)>30
    xx=[];yy=[];ww=[]
    for r in train:
        p=ROOT/r['file'];assert sha(p)==r['sha256'];z=np.load(p);x=features(z['x'].astype('float32'));y=z['y']
        ids=np.unique(np.linspace(0,len(y)-1,min(len(y),96)).astype(int))
        xx.append(x[ids]);yy.append(y[ids]);ww.append(np.full(len(ids),1/len(ids),np.float32))
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww)
    config=dict(architecture='Expected true human native crop IoU ExtraTrees utility; spatial3x3 ROI/ROD144+geometry12; no saliency/MOS inputs',trees=96,depth=18,min_leaf=24,max_features=.8,seed=6023,
        train_sources=len(train),train_candidates=len(y),dev_sources=len(dev),once_test_sources=len(test),parameters_limit=9000000000,
        labels='Actual author xywh unchanged; native aspect only; no MOS, detection, pseudo crops or target-ratio rewrites',
        selection='fixed96 trees; dev reported, not used for checkpoint selection; once-test after checkpoint SHA freeze; positive mean gain and lower bootstrap95>0 for candidate eligibility',
        manifest_sha256=sha(ROOT/'FEATURES_FCDB.json'),code_sha256=sha(__file__),formal_inputs=False)
    save('FCDB_CONFIG.json',config)
    checkpoint=ROOT/'fcdb_ranker.joblib'
    # User GPU clone has 62GiB RAM; tree parallelism changes execution only.
    model=joblib.load(checkpoint) if checkpoint.exists() else ExtraTreesRegressor(n_estimators=0,max_depth=18,min_samples_leaf=24,max_features=.8,n_jobs=8,random_state=6023,warm_start=True)
    for count in [24,48,72,96]:
        if len(getattr(model,'estimators_',[]))>=count:continue
        model.n_estimators=count;model.fit(x,y,sample_weight=w)
        tmp=checkpoint.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(checkpoint)
        save('FCDB_TRAIN_PROGRESS.json',dict(trees=count,total=96,heartbeat=time.time(),checkpoint_sha256=sha(checkpoint)))
    save('FCDB_BEFORE_TEST_FREEZE.json',dict(checkpoint_sha256=sha(checkpoint),config_sha256=sha(ROOT/'FCDB_CONFIG.json'),at=time.time(),fixed_final_trees=96))
    devq=evaluate(model,dev);save('FCDB_DEV_EVAL.json',devq)
    # Resume is a read of frozen test metrics, never a new fitted model or scan.
    testq=read(ROOT/'FCDB_ONCE_TEST.json') if (ROOT/'FCDB_ONCE_TEST.json').exists() else evaluate(model,test)
    save('FCDB_ONCE_TEST.json',testq)
    diagnostic=retarget_diagnostic(model)
    save('FCDB_RETARGET_DIAGNOSTIC.json',diagnostic)
    nparams=sum(t.tree_.threshold.size+t.tree_.value.size+t.tree_.feature.size for t in model.estimators_)
    result=dict(success=True,accepted=testq['gain']>.002 and testq['gain_bootstrap95'][0]>0,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),parameters=nparams,
        dev={k:v for k,v in devq.items() if k!='records'},once_test={k:v for k,v in testq.items() if k!='records'},official_score=None,
        smartvidcrop_comparison=diagnostic['source_macro'],competition_ratio_GT=False,backbone_pretraining_overlap='not established; once-test is unseen by new ranker, not proven independent of GAIC pretraining',formal_production='requires GPU and actual public video full decode + two validators')
    save('FCDB_TRAIN_DONE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
