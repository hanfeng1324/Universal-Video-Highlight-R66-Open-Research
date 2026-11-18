"""Candidate-distribution adaptation: frozen96-tree plus learned IoU residual."""
from pathlib import Path
import os,sys,json,time,traceback
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
import r63_crop_search as core
R=Path(__file__).resolve().parent

def save(name,q):
    p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)

def inputs(raw,u60):
    return np.c_[core.features(np.asarray(raw,np.float32)),np.asarray(u60,np.float32)].astype(np.float32)

class Adapter:
    def __init__(self,path):self.model=joblib.load(path)
    def predict(self,raw,u60):
        return np.asarray(u60,np.float32)+self.model.predict(inputs(raw,u60)).astype(np.float32)

def main():
    ck=R/'r63_candidate_residual.joblib'
    if (R/'ADAPTER_DONE.json').exists():print('ADAPTER_ALREADY_DONE');return
    train=json.loads((R/'TRAIN_SEARCH_DONE.json').read_text())
    assert train['success'] and not train['formal_inputs'] and len(train['records'])==256
    config=dict(target='true human IoU minus actual frozen96-tree prediction on new expanded crops',
                train_sources=256,dev_sources=80,max_iter=96,max_leaf_nodes=15,max_depth=5,
                learning_rate=.06,l2_regularization=10,min_samples_leaf=64,early_stopping=False,
                random_state=6301,tree_sha256=core.sha(core.TREE),candidate_code_sha256=core.sha(R/'r63_crop_search.py'),
                new_supervision='actual expanded candidate distribution, no expert fusion or R62 weights',
                no_blend_search=True,formal_inputs=False,test_and_exposed_final_excluded=True)
    save('ADAPTER_CONFIG.json',config);xx=[];yy=[];ww=[]
    for r in train['records']:
        p=R/'train_features'/(r['source']+'.npz');assert core.sha(p)==r['feature_sha256'];z=np.load(p)
        ids=np.unique(np.linspace(0,len(z['y'])-1,min(384,len(z['y']))).astype(int))
        xx.append(inputs(z['x'][ids],z['u60'][ids]));yy.append(z['y'][ids]-z['u60'][ids]);ww.append(np.full(len(ids),1/len(ids)))
    x=np.concatenate(xx);y=np.concatenate(yy);weights=np.concatenate(ww);weights*=len(weights)/weights.sum()
    model=HistGradientBoostingRegressor(**{k:config[k] for k in ['max_iter','max_leaf_nodes','max_depth','learning_rate','l2_regularization','min_samples_leaf','early_stopping','random_state']})
    model.fit(x,y,sample_weight=weights);tmp=ck.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(ck)
    save('ADAPTER_BEFORE_DEV.json',dict(checkpoint_sha256=core.sha(ck),config_sha256=core.sha(R/'ADAPTER_CONFIG.json'),at=time.time(),train_candidates=len(y)))
    adapter=Adapter(ck);recs=[]
    dev=json.loads((R/'PUBLIC_SEARCH_DONE.json').read_text())
    for r in dev['records']:
        p=R/'public_features'/(r['source']+'.npz');assert core.sha(p)==r['feature_sha256'];z=np.load(p)
        s=adapter.predict(z['x'],z['u60']);n=int(z['original_count']);j=int(s.argmax())
        recs.append(dict(source=r['source'],ratio=r['ratio'],old_actual=r['old_actual'],expanded_tree_actual=r['new_actual'],
                         adapted_actual=float(z['y'][j]),adapted_old_pool_actual=float(z['y'][int(s[:n].argmax())]),
                         selected_new=j>=n,oracle=r['new_oracle']))
    d=np.array([r['adapted_actual']-r['old_actual'] for r in recs]);boot=d[np.random.default_rng(6301).integers(len(d),size=(5000,len(d)))].mean(1)
    means={k:float(np.mean([r[k] for r in recs])) for k in ['old_actual','expanded_tree_actual','adapted_actual','adapted_old_pool_actual','oracle']}
    # Original tree remains in every utility; this is a documented distribution
    # adapter, never represented as an unchanged tree-only candidate.
    result=dict(success=True,checkpoint=str(ck),checkpoint_sha256=core.sha(ck),config=config,means=means,
                actual_iou_delta=float(d.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),nonnegative_sources=int((d>=0).sum()),
                records=recs,train_candidates=len(y),dev_sources=len(recs),formal_inputs=False,
                production_eligible=bool(d.mean()>0),qualification='development evidence; independent video path evaluation still required',
                parameters=sum(len(p.nodes)*8 for it in model._predictors for p in it))
    save('ADAPTER_DONE.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:save('ADAPTER_FAILED.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
