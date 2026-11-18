"""Frozen DINOv2 patch features from public FCDB train/dev only.

No formal input path or old final/test access. Images are resized preserving
aspect ratio without cropping. Annotation extent is never an encoder input.
"""
from pathlib import Path
import os, json, time, hashlib, zipfile, traceback
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '4')
import numpy as np
import cv2, torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
PARENT = Path('/root/autodl-fs/r60_20261022')

def save(name, value):
    p=ROOT/name; p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(value, indent=2)); t.replace(p)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()

def encode(model, rgb):
    h,w=rgb.shape[:2]
    scale=518/max(h,w); nh=max(14,round(h*scale/14)*14); nw=max(14,round(w*scale/14)*14)
    a=cv2.resize(rgb,(nw,nh)).astype('float32')/255
    a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    x=torch.from_numpy(a.transpose(2,0,1).copy())[None].cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        v=model(pixel_values=x).last_hidden_state.float()
    gh,gw=nh//14,nw//14
    assert v.shape[1]==1+gh*gw
    return v[0,1:].reshape(gh,gw,-1).cpu().numpy().astype('float16'),v[0,0].cpu().numpy().astype('float16')

def main():
    import fcntl
    lock=(ROOT/'SEMANTIC_EXTRACT.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'SEMANTIC_EXTRACT_DONE.json').exists():return
    torch.set_num_threads(4);cv2.setNumThreads(1)
    cfg=json.loads((PARENT/'FEATURES_FCDB.json').read_text());assert not cfg['formal_inputs']
    rows=[r for r in cfg['rows'] if r['role'] in ['train','dev']]
    plan=json.loads((PARENT/'FCDB_METADATA/SOURCE_PLAN.json').read_text())
    meta={str(r['source_id']):r for r in plan['eligible_official_train_sources']}
    modeldir=ROOT/'weights/dinov2-base';modeldir.mkdir(parents=True,exist_ok=True)
    save('SEMANTIC_PROGRESS.json',dict(stage='fetch_pretrained',model='facebook/dinov2-base',heartbeat=time.time(),formal_inputs=False))
    if not (modeldir/'config.json').exists():
        # Pin the immutable HF revision on first fetch and save it in receipt.
        from huggingface_hub import HfApi, snapshot_download
        info=HfApi().model_info('facebook/dinov2-base')
        save('PRETRAINED_SOURCE.json',dict(repo='facebook/dinov2-base',revision=info.sha,url='https://huggingface.co/facebook/dinov2-base',license='apache-2.0; verify model card saved with weights',formal_inputs=False))
        snapshot_download('facebook/dinov2-base',revision=info.sha,local_dir=modeldir,allow_patterns=['config.json','model.safetensors','pytorch_model.bin','README.md','preprocessor_config.json','.gitattributes'])
    model=AutoModel.from_pretrained(modeldir,local_files_only=True).eval().cuda().requires_grad_(False)
    save('ENCODER_READY.json',dict(parameters=sum(p.numel() for p in model.parameters()),files={p.name:sha(p) for p in modeldir.iterdir() if p.is_file()},formal_inputs=False))
    start=time.time();records=[]
    with zipfile.ZipFile(PARENT/'public_data/FCDB.zip') as archive:
        names={Path(n).name:n for n in archive.namelist() if n.lower().endswith('.jpg')}
        for i,row in enumerate(rows):
            dst=ROOT/'semantic_fcdb'/(row['source']+'.npz');dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                old=np.load(PARENT/row['file']);m=meta[row['source'].removeprefix('fcdb_')]
                data=archive.read(names[m['archive_basename']]);digest=hashlib.sha256(data).hexdigest();assert digest==str(old['image_sha256'])
                rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                assert rgb.shape[:2]==(int(old['height']),int(old['width']))
                patch,cls=encode(model,rgb)
                temp=dst.with_suffix('.tmp.npz');np.savez_compressed(temp,patch=patch,cls=cls,width=rgb.shape[1],height=rgb.shape[0],image_sha256=digest);temp.replace(dst)
            records.append(dict(source=row['source'],role=row['role'],file=str(dst),sha256=sha(dst),parent_feature_file=str(PARENT/row['file']),parent_sha256=row['sha256']))
            if i%10==0 or i+1==len(rows):
                q=dict(stage='extract_public_fcdb',completed=i+1,total=len(rows),elapsed_seconds=time.time()-start,heartbeat=time.time(),formal_inputs=False)
                save('SEMANTIC_PROGRESS.json',q);print(json.dumps(q),flush=True)
    save('SEMANTIC_FEATURES.json',dict(rows=records,formal_inputs=False,old_test_access=False,model='frozen DINOv2-base',labels='unchanged genuine FCDB native crop IoU, inherited identity audit; not target-ratio GT'))
    save('SEMANTIC_EXTRACT_DONE.json',dict(success=True,sources=len(records),seconds=time.time()-start,formal_inputs=False))

if __name__=='__main__':
    try:main()
    except BaseException:save('SEMANTIC_EXTRACT_FAILED.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
