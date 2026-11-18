"""Real GNMC multi-aspect crop supervision, frozen semantic/GAIC features.

No annotation is changed. Invalid or ratio-inconsistent labels are quarantined.
Original image families stay together, including all five ratio annotations.
"""
from pathlib import Path
import os,sys,json,time,hashlib,zipfile,traceback
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'4')
ROOT=Path(__file__).resolve().parent
sys.path[:0]=['/root/autodl-fs/r59_20261021','/root/autodl-fs/r60_20261022']
import numpy as np,cv2,torch,joblib
from transformers import AutoModel
from r64_semantic_extract import encode
from r64_semantic_ranker import candidate_features
from r59_model import proposals,frozen_features
import r59_gaic as gaic
RATIOS={'16:9':[16,9],'3:4':[3,4],'4:3':[4,3],'2:1':[2,1],'1:1':[1,1]}

def save(n,q):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def phash(rgb):
 a=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);v=cv2.dct(cv2.resize(a,(32,32)).astype('float32'))[:8,:8].flatten()[1:]
 bits=v>np.median(v);return sum(int(x)<<i for i,x in enumerate(bits))
def bitdistance(a,b):return (a^b).bit_count()

def audit(archive,metadata):
 """Train/dev exclusion uses hashes, not crop quality or official predictions."""
 out=ROOT/'GNMC/IMAGE_IDENTITY_AUDIT.json'
 if out.exists():return json.loads(out.read_text())
 names={Path(n).name:n for n in archive.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))}
 image_rows=[];buckets={};matches=[]
 for role in ['test','validation','train']:
  for r in sorted(metadata[role],key=lambda q:q['filename']):
   name=r['filename'];data=archive.read(names[name]);rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
   h,w=rgb.shape[:2];p=phash(rgb);digest=hashlib.sha256(data).hexdigest();possible=set()
   for k in range(5):possible.update(buckets.get((k,(p>>(13*k))&8191),[]))
   near=[i for i in possible if bitdistance(p,image_rows[i]['phash'])<=4]
   record=dict(filename=name,role=role,sha256=digest,phash=p,width=w,height=h,excluded=bool(near),duplicates=[image_rows[i]['filename'] for i in near])
   image_rows.append(record);index=len(image_rows)-1
   for k in range(5):buckets.setdefault((k,(p>>(13*k))&8191),[]).append(index)
   if near:matches.append(record)
  save('GNMC/AUDIT_PROGRESS.json',dict(stage='source_family_exact_and_phash',done=len(image_rows),total=10000,heartbeat=time.time()))
 # Existing FCDB labels need not be opened: match only public JPEG identities.
 fcdb=Path('/root/autodl-fs/r60_20261022/public_data/FCDB.zip')
 with zipfile.ZipFile(fcdb) as z:
  for name in z.namelist():
   if not name.lower().endswith('.jpg'):continue
   data=z.read(name);im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
   if im is None:continue
   p=phash(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));possible=set()
   for k in range(5):possible.update(buckets.get((k,(p>>(13*k))&8191),[]))
   for i in possible:
    if bitdistance(p,image_rows[i]['phash'])<=4:
     image_rows[i]['excluded']=True;image_rows[i]['existing_fcdb_overlap']=Path(name).name
 q=dict(rows=image_rows,quarantine=matches,rule='pHash63 distance<=4 or exact duplicate: earlier protected split wins; existing FCDB image overlaps excluded from all new GNMC model roles; image-family all ratios together',formal_inputs=False,backbone_pretraining_overlap='UNKNOWN',at=time.time())
 save('GNMC/IMAGE_IDENTITY_AUDIT.json',q);return q

def main():
 import fcntl
 lock=(ROOT/'GNMC_FEATURE.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'GNMC_FEATURES_DONE.json').exists():return
 torch.set_num_threads(4);cv2.setNumThreads(1)
 archivepath=ROOT/'GNMC/GNMC.zip';model=AutoModel.from_pretrained(ROOT/'weights/dinov2-base',local_files_only=True).eval().cuda().requires_grad_(False)
 expert=gaic.load();tree=joblib.load('/root/autodl-fs/r60_20261022/fcdb_ranker.joblib');tree.n_jobs=2
 projection=torch.load(ROOT/'dino_projection.pt',weights_only=False)
 with zipfile.ZipFile(archivepath) as archive:
  metanames={Path(n).name:n for n in archive.namelist() if n.endswith('.json')}
  metadata={role:json.loads(archive.read(metanames[role+'.json'])) for role in ['train','validation','test']}
  assert [len(metadata[r]) for r in ['train','validation','test']]==[7000,1000,2000]
  identity=audit(archive,metadata);ids={r['filename']:r for r in identity['rows']}
  # Fixed source order independent of labels/scores enables useful early outputs.
  selected={role:sorted([r for r in metadata[role] if not ids[r['filename']]['excluded']],key=lambda r:hashlib.sha256(('r64-gnmc:'+r['filename']).encode()).hexdigest())[:limit] for role,limit in [('train',2048),('validation',256)]}
  config=dict(archive_sha256=sha(archivepath),ratios=RATIOS,train_source_budget=2048,validation_source_budget=256,phase1_train_sources=512,phase1_validation_sources=128,selection='fixed image-hash order; no crop-quality selection',candidate_policy='original generic image-midpoint lattice; fixed evenly spaced128 candidates per training ratio; full generic pool for development; no human crop seed, no GT perturbation, no semantic score filtering',supervision='exact normalized human xyxy times actual original image width/height; labels not clipped, resized or aspect rewritten',quarantine='any coordinate outside [0,1], nonpositive extent, actual native-ratio relative error>2%; skip label only; preserve original',formal_inputs=False,test_quality_access=False,projection_source='FCDB train-only unsupervised PCA; encoder frozen DINOv2-base',license='PolyForm Noncommercial1.0.0; retain GNMC/LICENSE; noncommercial research only',source_sha256=sha(__file__))
  save('GNMC_FEATURE_CONFIG.json',config)
  byname={Path(n).name:n for n in archive.namelist() if n.lower().endswith(('.jpeg','.jpg','.png'))};rows=[];invalid=[];start=time.time()
  # Early phase is a complete source-disjoint train/dev cohort, not a final test.
  sequence=[]
  for role,lo,hi in [('train',0,512),('validation',0,128),('train',512,2048),('validation',128,256)]:sequence += [(role,r) for r in selected[role][lo:hi]]
  for j,(role,r) in enumerate(sequence):
   ident=ids[r['filename']];dest=ROOT/'gnmc_features'/role/(Path(r['filename']).stem+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():
    data=archive.read(byname[r['filename']]);assert hashlib.sha256(data).hexdigest()==ident['sha256']
    rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
    patch,cls=encode(model,rgb);xx=[];yy=[];uu=[];bb=[];parts=[0];tags=[]
    for tag,ratio in RATIOS.items():
     norm=np.asarray(r['crop_bboxes'][tag],np.float64);gt=norm*np.array([w,h,w,h]);size=gt[2:]-gt[:2]
     valid=np.isfinite(norm).all() and (norm>=0).all() and (norm<=1).all() and (size>0).all()
     error=abs(size[0]/max(size[1],1e-6)/(ratio[0]/ratio[1])-1)
     if not valid or error>.02:
      invalid.append(dict(filename=r['filename'],ratio=tag,reason='invalid_bounds_or_native_ratio_error_gt2percent',relative_ratio_error=float(error)));continue
     b=proposals(w,h,ratio,[[w/2,h/2]])
     if role=='train':b=b[np.unique(np.linspace(0,len(b)-1,min(128,len(b))).astype(int))]
     raw,_=frozen_features(expert,rgb,b,np.ones((32,32),np.float32))
     x,u=candidate_features(patch,cls,b,raw,w,h,ratio,projection,tree)
     inter=np.maximum(0,np.minimum(b[:,2:],gt[2:])-np.maximum(b[:,:2],gt[:2])).prod(1)
     y=inter/np.maximum((b[:,2:]-b[:,:2]).prod(1)+size.prod()-inter,1e-9)
     xx.append(x.astype('float16'));yy.append(y.astype('float32'));uu.append(u);bb.append(b);parts.append(parts[-1]+len(b));tags.append(tag)
    if not xx:continue
    tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,x=np.concatenate(xx),y=np.concatenate(yy),u=np.concatenate(uu),boxes=np.concatenate(bb),parts=parts,tags=tags,width=w,height=h,source=r['filename'],image_sha256=ident['sha256']);tmp.replace(dest)
   rows.append(dict(source='gnmc_'+Path(r['filename']).stem,role=role,file=str(dest),sha256=sha(dest),image_sha256=ident['sha256']))
   if j%8==0 or j==639 or j+1==len(sequence):
    save('GNMC_FEATURE_PROGRESS.json',dict(completed=j+1,total=len(sequence),train_sources=sum(q['role']=='train' for q in rows),dev_sources=sum(q['role']=='validation' for q in rows),seconds=time.time()-start,heartbeat=time.time(),formal_inputs=False))
    print('GNMC_FEATURE',j+1,len(sequence),time.time()-start,flush=True)
   if j==639:
    save('GNMC_PHASE1_FEATURES.json',dict(rows=rows.copy(),formal_inputs=False,labels='genuine native human multi-ratio crops',config=config,invalid_labels=invalid.copy()))
    wait=time.time()
    while not (ROOT/'GNMC_phase1/TRAIN_DONE.json').exists():
     if (ROOT/'GNMC_TRAIN_FAILED.json').exists():raise RuntimeError('Phase1 training failed; do not spend full feature budget')
     if time.time()-wait>1200:raise TimeoutError('20min phase1 signal deadline')
     time.sleep(10)
    signal=json.loads((ROOT/'GNMC_phase1/TRAIN_DONE.json').read_text());ratio=signal['dev']['ratio_means'].get('16:9',{})
    if not signal['accepted_for_video_test'] and ratio.get('neural',0)-ratio.get('tree',0)<.015:
     save('GNMC_STOPPED_AFTER_PHASE1.json',dict(reason='No positive early native-IoU signal; preserve all phase1 artifacts and switch mechanism',evidence=signal,at=time.time(),formal_inputs=False));return
  save('GNMC_FEATURES.json',dict(rows=rows,formal_inputs=False,labels='genuine native human multi-ratio crops',config=config,invalid_labels=invalid))
  save('GNMC_FEATURES_DONE.json',dict(success=True,sources=len(rows),seconds=time.time()-start,formal_inputs=False))

if __name__=='__main__':
 try:main()
 except BaseException:save('GNMC_FEATURE_FAILED.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
