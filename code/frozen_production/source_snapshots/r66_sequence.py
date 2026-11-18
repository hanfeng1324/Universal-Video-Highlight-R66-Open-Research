"""Human-IoU supervised structured video-path learning, original public sources.

Frozen R66 emissions; learn scene/context-conditioned transition energies with
full-sequence CRF supervision. No formal media or exposed evaluation sources.
CPU only, so current dual-GPU production keeps priority.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time, hashlib, tarfile, io, traceback
import numpy as np
import torch
import joblib

R = Path('/root/autodl-tmp/r66_20261023')
OUT = R / 'SEQUENCE'
A = Path('/root/autodl-fs/r65_20261023/A2')
PARENT = R / 'PRODUCTION_RANK/weights/decision.joblib'
sys.path[:0] = [str(R/'PRODUCTION_RANK/source_snapshots'), str(R), '/root/autodl-fs/r59_20261021']
from r66_rank import predict, parent
from r59_model import path_optimize

TEMP = .06
TOPK = 64
BASE = np.array([.18, .035, .005, .002], np.float32)
CAP = np.array([2., .5, 2., .5], np.float32)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    pending = p.with_suffix('.pending')
    pending.write_text(json.dumps(data, indent=2))
    pending.replace(p)

def softmax(values):
    values = np.exp((values-values.max())/.08)
    return values / values.sum()

def prepare(groups, scores, frames, w, h, reference=None):
    """Label-independent pruning and edge features, shared by fit/inference."""
    if reference is None:
        reference = path_optimize([q['b'] for q in groups], scores, frames, [], w, h)
    seq = dict(groups=groups, scores=scores, frames=frames, selections=[], unary=[],
               features=[], context=[], teacher_y=[])
    states=[]
    for q, score, ref in zip(groups, scores, reference):
        idx = np.unique(np.r_[np.argsort(score)[-TOPK:], int(ref)])
        seq['selections'].append(idx)
        b = np.asarray(q['b'][idx], np.float32)/[w,h,w,h]
        center=(b[:,:2]+b[:,2:])/2
        size=np.maximum(b[:,2:]-b[:,:2],1e-6)
        concept=np.asarray(q['x'][idx,265:289],np.float32)
        concept/=np.maximum(np.linalg.norm(concept,axis=1,keepdims=True),1e-6)
        prob=softmax(np.asarray(score[idx],np.float32))
        scene=np.asarray(q['x'][idx,369:375],np.float32).mean(0)
        entropy=float(-(prob*np.log(np.maximum(prob,1e-9))).sum()/max(np.log(len(prob)),1))
        gap=float(np.sort(score)[-1]-np.sort(score)[-2]) if len(score)>1 else 0.
        states.append((center,size,concept,prob@center,scene,entropy,gap))
        seq['unary'].append(torch.tensor(np.asarray(score[idx],np.float32)))
        if 'y' in q:seq['teacher_y'].append(torch.tensor(np.asarray(q['y'][idx],np.float32)))
    median=max(float(np.median(np.diff(frames))),1.) if len(frames)>1 else 1.
    for t in range(1,len(groups)):
        pc,ps,pv,pm,p_scene,pe,pg=states[t-1]
        c,s,v,m,scene,e,g=states[t]
        delta=c[None,:,:]-pc[:,None,:]
        movement=np.square(delta).sum(-1)
        size_change=np.square(np.log(ps[:,None,:]/s[None,:,:])).sum(-1)
        reference_motion=m-pm
        compensated=np.square(delta-reference_motion).sum(-1)
        concept_drift=np.clip(1-pv@v.T,0,2)
        edge=np.stack([movement,size_change,compensated,concept_drift],-1).astype('float32')
        context=np.r_[1.,(p_scene+scene)/2,(pe+e)/2,
                      np.clip((pg+g)*5,0,2),reference_motion,
                      np.clip(np.log(max(frames[t]-frames[t-1],1)/median),-2,2)]
        seq['features'].append(torch.tensor(edge))
        seq['context'].append(torch.tensor(context,dtype=torch.float32))
    seq['context']=torch.stack(seq['context'])
    return seq

def teacher(seq):
    """Soft human-IoU oracle over complete trajectories, with original prior."""
    with torch.no_grad():
        y=seq['teacher_y']; edges=seq['features']
        penalties=[e[:,:,0]*.18+e[:,:,1]*.035 for e in edges]
        alpha=[y[0]/TEMP]
        for u,p in zip(y[1:],penalties):
            alpha.append(u/TEMP+torch.logsumexp(alpha[-1][:,None]-p/TEMP,0))
        z=torch.logsumexp(alpha[-1],0)
        beta=[None]*len(y);beta[-1]=torch.zeros_like(y[-1])
        for t in range(len(y)-2,-1,-1):
            beta[t]=torch.logsumexp(-penalties[t]/TEMP+(y[t+1]/TEMP+beta[t+1])[None,:],1)
        expected=[]
        for t,e in enumerate(edges):
            prob=(alpha[t][:,None]-penalties[t]/TEMP+(y[t+1]/TEMP+beta[t+1])[None,:]-z).exp()
            assert abs(float(prob.sum())-1)<.01
            expected.append((e*prob[:,:,None]).sum((0,1)))
        seq['expected_edges']=torch.stack(expected)
        seq['expected_unary']=sum((u*(a+b-z).exp()).sum() for u,a,b in zip(seq['unary'],alpha,beta))

class Transition(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.zeros(dimensions,4))
        self.bias=torch.nn.Parameter(torch.log(torch.tensor(BASE)/(torch.tensor(CAP)-torch.tensor(BASE))))
        self.register_buffer('initial_bias',self.bias.detach().clone())
        self.register_buffer('caps',torch.tensor(CAP))
    def forward(self, context):
        return (context@self.weight+self.bias).sigmoid()*self.caps

def objective(seq, model):
    coef=model(seq['context'])
    alpha=seq['unary'][0]/TEMP
    for t,u in enumerate(seq['unary'][1:]):
        penalty=(seq['features'][t]*coef[t]).sum(-1)
        alpha=u/TEMP+torch.logsumexp(alpha[:,None]-penalty/TEMP,0)
    return (torch.logsumexp(alpha,0)-seq['expected_unary']/TEMP+
            (coef*seq['expected_edges']).sum()/TEMP)/len(seq['unary'])

def decode(seq, weight, bias):
    context=seq['context'].numpy()
    coefficients=CAP/(1+np.exp(-np.clip(context@weight+bias,-40,40)))
    values=seq['unary'][0].numpy().copy();back=[]
    for t,u in enumerate(seq['unary'][1:]):
        penalty=(seq['features'][t].numpy()*coefficients[t]).sum(-1)
        value=values[:,None]-penalty;idx=value.argmax(0)
        values=u.numpy()+value[idx,np.arange(value.shape[1])]
        values-=values.max();back.append(idx)
    chosen=[int(values.argmax())]
    for idx in reversed(back):chosen.append(int(idx[chosen[-1]]))
    chosen.reverse()
    return [int(indices[i]) for indices,i in zip(seq['selections'],chosen)]

def select_path(groups, scores, frames, cuts, w, h, checkpoint):
    """Complete CPU inference with cut resets and no quality labels required."""
    spans=[0]+[j for j in range(1,len(frames)) if any(frames[j-1]<c<=frames[j] for c in cuts)]+[len(frames)]
    result=[]
    for lo,hi in zip(spans[:-1],spans[1:]):
        if hi-lo==1:result.append(int(np.argmax(scores[lo])));continue
        seq=prepare(groups[lo:hi],scores[lo:hi],frames[lo:hi],w,h)
        result.extend(decode(seq,checkpoint['weight'],checkpoint['bias']))
    return result

def assess(sequences, weight, bias):
    rows=[]
    for seq in sequences:
        chosen=decode(seq,weight,bias)
        q=seq['groups']; scores=seq['scores']; frames=seq['frames']
        rank=path_optimize([r['b'] for r in q],scores,frames,[],q[0]['w'],q[0]['h'])
        a2=path_optimize([r['b'] for r in q],seq['a2_scores'],frames,[],q[0]['w'],q[0]['h'])
        mean=lambda indices:float(np.mean([r['y'][j] for r,j in zip(q,indices)]))
        rows.append(dict(source=q[0]['source'],ratio=str(q[0]['ratio']),sequence=mean(chosen),rank=mean(rank),a2=mean(a2)))
    sources=[]
    for source in sorted({r['source'] for r in rows}):
        group=[r for r in rows if r['source']==source]
        sources.append(dict(source=source,**{k:float(np.mean([r[k] for r in group])) for k in ['sequence','rank','a2']}))
    return dict(gain_vs_rank=float(np.mean([r['sequence']-r['rank'] for r in sources])),
                gain_vs_a2=float(np.mean([r['sequence']-r['a2'] for r in sources])),sources=sources,rows=rows)

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'CONFIG.json').exists(),'Preserve previous run; use a distinct research directory'
    os.nice(10);torch.set_num_threads(1);torch.manual_seed(6603);start=time.time()
    audit=json.loads((A/'TRAIN_SOURCE_SPLIT_AUDIT.json').read_text());assert audit['success'] and not audit['source_overlap']
    config=dict(mechanism='Scene/context-conditioned trajectory CRF; full-sequence human-IoU teacher marginal supervision; semantic continuity and reference-motion compensated edges',
                parent_sha256=sha(PARENT),source_sha256=sha(__file__),formal_inputs=False,
                training_sources='original retarget003-022 only; original source fit14/cal3/val3',
                checkpoint_selection='cal3 only; val3 diagnostic once after freeze; neither is new independent evidence',
                temperature=TEMP,states=TOPK,include_frozen_parent_path=True,epochs=64,
                checkpoint_epochs=[16,32,64],new_evaluation_sources=[97,98,99,100],cuda_used=False,
                semantic_GT=False,relationship_GT=False,supervision='six-human native crop IoU; native1:3 and3:1, not competition ratio GT')
    save('CONFIG.json',config)
    bundle=joblib.load(PARENT);bundle['parent']['model'].n_jobs=1
    grouped={}
    with tarfile.open(A/'FEATURE_COMPLETION_DELTA35.tar') as archive:
        for member in archive:
            if not member.isfile() or not Path(member.name).name.startswith('retarget_'):continue
            groups=joblib.load(io.BytesIO(archive.extractfile(member).read()))
            for q in groups:
                assert q['domain']=='video' and 3<=int(q['source'].split('_')[-1])<=22
                assert q['x'].shape[1]==405
                grouped.setdefault((q['source'],str(q['ratio'])),[]).append(q)
    assert len({s for s,r in grouped})==20
    roles={};sequences=[]
    for (source,ratio),groups in sorted(grouped.items()):
        groups.sort(key=lambda q:q['frame']);roles[source]=next(k for k,v in audit['source_roles'].items() if source in v)
        scores=[predict(q,bundle) for q in groups]
        seq=prepare(groups,scores,[q['frame'] for q in groups],groups[0]['w'],groups[0]['h'])
        seq['a2_scores']=[parent(q,bundle['parent']) for q in groups]
        seq['role']=roles[source];sequences.append(seq)
        if seq['role']=='fit':teacher(seq)
        save('PROGRESS.json',dict(stage='public_sequence_cache_and_teacher',sequences=len(sequences),total=40,seconds=time.time()-start))
    fit=[s for s in sequences if s['role']=='fit'];cal=[s for s in sequences if s['role']=='cal'];val=[s for s in sequences if s['role']=='val']
    assert [len(fit),len(cal),len(val)]==[28,6,6]
    save('DATA_READY.json',dict(source_roles=roles,fit_sequences=len(fit),cal_sequences=len(cal),val_sequences=len(val),candidate_groups=sum(len(s['groups']) for s in sequences),formal_inputs=False))
    model=Transition(sequences[0]['context'].shape[1]);opt=torch.optim.Adam(model.parameters(),lr=.025)
    rng=np.random.default_rng(6603);choices=[]
    for epoch in range(1,65):
        losses=[]
        for index in rng.permutation(len(fit)):
            loss=objective(fit[int(index)],model)+.002*model.weight.square().mean()+.002*(model.bias-model.initial_bias).square().mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        save('PROGRESS.json',dict(stage='structured_trajectory_training',epoch=epoch,total=64,loss=float(np.mean(losses)),seconds=time.time()-start))
        if epoch not in [16,32,64]:continue
        checkpoint=dict(weight=model.weight.detach().numpy().copy(),bias=model.bias.detach().numpy().copy(),config=config,epoch=epoch)
        path=OUT/f'epoch{epoch:03d}.joblib';joblib.dump(checkpoint,path,compress=3)
        metrics=assess(cal,checkpoint['weight'],checkpoint['bias'])
        choices.append(dict(epoch=epoch,path=str(path),sha256=sha(path),cal=metrics));save('CHECKPOINTS.json',dict(choices=choices))
    best=max(choices,key=lambda q:q['cal']['gain_vs_rank'])
    final=OUT/'transition.joblib';final.write_bytes(Path(best['path']).read_bytes())
    save('BEFORE_NEW_EVAL_FREEZE.json',dict(at=time.time(),checkpoint=str(final),sha256=sha(final),selection='original cal3 only',parent_sha256=sha(PARENT)))
    selected=joblib.load(final);validation=assess(val,selected['weight'],selected['bias'])
    report=dict(success=True,checkpoint=str(final),sha256=sha(final),selected=best,validation=validation,parameters=int(model.weight.numel()+model.bias.numel()),
                seconds=time.time()-start,formal_inputs=False,official_score=None,
                eligible_for_new_public_evaluation=best['cal']['gain_vs_rank']>0 and validation['gain_vs_rank']>=0,
                eligible_for_production=False,limitation='Original training-source held development only; fresh source comparison and native public gate still required')
    save('DONE.json',report)
    shared=Path('/root/autodl-fs/r66_20261023/SEQUENCE');shared.mkdir(exist_ok=True)
    for name in ['CONFIG.json','DATA_READY.json','BEFORE_NEW_EVAL_FREEZE.json','transition.joblib','DONE.json']:
        __import__('shutil').copyfile(OUT/name,shared/name)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        save('FAILED.json',dict(at=time.time(),traceback=traceback.format_exc(),formal_inputs=False));raise
