"""Raw video -> R57 expert selection -> R60/R63 rebuild -> frozen R66 sequence.

This repair never reads a frozen submission. It builds every input-dependent
cache below a new work directory. Production snapshots remain unchanged.
Default: Linux, CUDA >=24GiB; --low-memory uses Accelerate CPU offload with
the same BF16 Qwen checkpoint and is separately reported.
"""
from pathlib import Path
import argparse,ast,json,sys,hashlib,shutil,os,time,types,gc,traceback
from temporal_segments import TemporalSegmenter, add_backend_arguments, backend_options
from model_assets import configuration, resolve as resolve_asset, model_rows
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,q):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
def link(src,dst):
 dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():
  if os.path.samefile(src,dst):return
  assert sha(dst)==sha(src),(str(src),str(dst));return
 try:os.link(src,dst)
 except OSError:shutil.copyfile(src,dst)
def prepare(package,work,low,config=None):
 config=config or {}
 snapshot=package/'code/cloud_snapshot';runtime=work/'runtime'
 prefix={'/root/autodl-fs/r57_20261021/clone_essentials/root/r53_autonomous_20261020':str(runtime/'r53_autonomous_20261020').replace('\\','/'),'/root/autodl-fs':str(runtime/'autodl-fs').replace('\\','/'),'/autodl-fs/data':str(runtime/'autodl-fs').replace('\\','/'),'/root/autodl-tmp':str(runtime/'autodl-tmp').replace('\\','/'),'/root/r53_autonomous_20261020':str(runtime/'r53_autonomous_20261020').replace('\\','/')}
 def target(original):
  for a,b in prefix.items():
   if original.startswith(a):return Path(b+original[len(a):])
  raise ValueError(original)
 transforms=[]
 for src in snapshot.rglob('*'):
  if not src.is_file() or 'validator' in src.relative_to(snapshot).parts or '__pycache__' in src.parts:continue
  actual=resolve_asset(package,src.relative_to(package).as_posix(),config)
  orig='/'+src.relative_to(snapshot).as_posix()
  if not any(orig.startswith(x) for x in prefix):continue
  dst=target(orig);dst.parent.mkdir(parents=True,exist_ok=True)
  if src.suffix=='.py':
   s=actual.read_text('utf8')
   for a,b in prefix.items():s=s.replace(a,b)
   if src.name=='r57_formal_production.py' and low:
    s=s.replace("assert torch.cuda.mem_get_info()[0]>22*2**30","assert torch.cuda.is_available()")
    s=s.replace("device_map={'':0},attn_implementation='sdpa'","device_map='auto',max_memory={0:'4GiB','cpu':'20GiB'},offload_folder=str(BASE/'offload'),attn_implementation='sdpa'")
   compile(s,str(dst),'exec');dst.write_text(s,encoding='utf8')
   transforms.append(dict(source=src.relative_to(package).as_posix(),original_sha256=sha(src),runtime_sha256=sha(dst)))
  elif not dst.exists():shutil.copyfile(actual,dst)
 for row in model_rows(package):
  if row['path'].endswith('/r66/decision.joblib'):continue
  link(resolve_asset(package,row['path'],config),target(row['original']))
 b=runtime/'autodl-fs/r57_20261021';clone=runtime/'r53_autonomous_20261020'
 for src in clone.rglob('*'):
  if src.is_file():link(src,b/'clone_essentials/root/r53_autonomous_20261020'/src.relative_to(clone))
 link(b/'weights/googlenet-1378be20.pth',b/'torchhub/checkpoints/googlenet-1378be20.pth')
 # The production variant's exact inference functions have priority over research.
 frozen=package/'code/frozen_production'
 prod=runtime/'sequence';prod.mkdir(parents=True,exist_ok=True)
 for src in list(frozen.glob('*.py'))+list((frozen/'source_snapshots').glob('*.py')):
  s=src.read_text('utf8')
  for a,c in prefix.items():s=s.replace(a,c)
  dst=prod/src.name;compile(s,str(dst),'exec');dst.write_text(s,encoding='utf8')
  transforms.append(dict(source=src.relative_to(package).as_posix(),original_sha256=sha(src),runtime_sha256=sha(dst)))
 checkpoint=resolve_asset(package,'local_assets/models/r66/decision.joblib',config)
 if checkpoint.is_dir() or checkpoint.suffix=='.json':
  from r66_checkpoint import inspect
  index,parts=inspect(checkpoint)
  link(index,prod/'weights/checkpoint.json')
  for item in parts['parts'].values():link(index.parent/item['file'],prod/'weights'/item['file'])
 else:link(checkpoint,prod/'weights/decision.joblib')
 save(work/'SOURCE_TRANSFORMS.json',dict(prefix=prefix,low_memory_offload=low,rows=transforms,frozen_submissions_read=False,official_validator_excluded=True,models_configured=bool(config)))
 return runtime,b,prod
def selected_module(name,path,namespace,names):
 """Load named frozen functions/classes, avoiding historical import-time JSONL reads."""
 tree=ast.parse(path.read_text('utf8'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 m=types.ModuleType(name);m.__dict__.update(namespace);m.__file__=str(path)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),m.__dict__)
 return m
def run(args):
 package=Path(args.package).resolve();work=Path(args.work_dir).resolve()
 assert not (work/'RUN_DONE.json').exists(),'Choose a new work directory; no successful run overwrite'
 work.mkdir(parents=True,exist_ok=True);started=time.time()
 if (work/'RUN_FAILED.json').exists():shutil.copy2(work/'RUN_FAILED.json',work/f'RUN_FAILED_{time.time_ns()}.json')
 fresh_runtime=not (work/'runtime').exists()
 runtime,base,prod=prepare(package,work,args.low_memory,configuration(args.models))
 if args.prepare_only:
  save(work/'PREPARED.json',dict(success=True,seconds=time.time()-started));print('PREPARE_PASS');return
 segmenter=TemporalSegmenter(args.temporal_backend,args.legacy_kts_dir)
 import torch,numpy as np,av,cv2,joblib
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(5701)
 assert torch.cuda.is_available(),'CUDA required by unchanged expert extraction'
 for p in [runtime/'r53_autonomous_20261020/tools',base,base/'deps/dsnet',base/'deps/sam2',runtime/'r53_autonomous_20261020/assets/r22_20261016/lighthouse',runtime/'autodl-fs/r59_20261021',runtime/'autodl-fs/r60_20261022',runtime/'autodl-fs/r63_20261023',runtime/'autodl-fs/r64_20261023',runtime/'autodl-fs/r65_20261023/A',runtime/'autodl-fs/r65_20261023/A2',prod]:sys.path.insert(0,str(p))
 import r57_cloud_visual as visual
 import r57_formal_production as upstream
 import r22_lighthouse_runtime as lighthouse
 lighthouse.ROOT=runtime/'r53_autonomous_20261020'
 lighthouse.BIG=lighthouse.ROOT/'assets/r22_20261016'
 lighthouse.ENCODER=lighthouse.ROOT/'aic_highlight/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
 lighthouse.WEIGHT=lighthouse.BIG/'weights/clip_cg_detr_qvhighlight.ckpt'
 visual.CLONE=lighthouse.ROOT
 visual.BASE=base
 entries=[json.loads(x) for x in Path(args.index).read_text('utf-8-sig').splitlines() if x.strip()]
 assert entries and len({r['video_id'] for r in entries})==len(entries)
 rows=[]
 for i,e in enumerate(entries):
  media=(Path(args.video_root)/e.get('video',e.get('video_path',e.get('filename','')))).resolve();assert media.is_file()
  meta=lighthouse.video_info(media)
  with av.open(str(media)) as c:count=sum(1 for f in c.decode(video=0))
  assert e['targetRatioWH'] in [[16,9],[9,16]]
  rows.append(dict(id=f'INPUT_{i:04d}',video_id=e['video_id'],media=str(media),media_sha256=sha(media),width=meta['width'],height=meta['height'],frames=count,targetRatioWH=e['targetRatioWH'],role='PUBLIC_REPRO' if args.public else 'AUTHORIZED_LOCAL_INFERENCE'))
 save(work/'INPUT_IDENTITY.json',dict(rows=rows,public=args.public,source=sha(__file__)))
 print('STAGE_TEMPORAL_EXTRACT',flush=True);visual.temporal(rows)
 segment_calls=[]
 for row in rows:
  dsnet=base/'visual'/row['id']/'DSNET.npz'
  with np.load(dsnet,allow_pickle=False) as samples:
   result=segmenter.segment(samples['features'],samples['picks'],row['frames'])
  segment_calls.append(dict(video_id=row['video_id'],media_sha256=row['media_sha256'],
                            dsnet_sha256=sha(dsnet),**result))
 save(work/'TEMPORAL_SEGMENTS.json',dict(backend=args.temporal_backend,
      role='diagnostic partition; R66 temporal proposal selection is independent',
      legacy_modules_loaded=any(n=='kts' or n.startswith('kts.') for n in sys.modules),
      calls=segment_calls))
 print('STAGE_SPATIAL_EXTRACT',flush=True);visual.spatial(rows)
 gc.collect();torch.cuda.empty_cache();print('STAGE_SEMANTIC_MODELS',flush=True)
 rt=upstream.Runtime();save(work/'UPSTREAM_PARAMETERS.json',dict(counts=rt.counts,total=sum(rt.counts.values())))
 for row in rows:
  print('STAGE_SEMANTICS',row['id'],flush=True);rt.analyze(row,base/'visual'/row['id'])
 del rt;gc.collect();torch.cuda.empty_cache()
 import r59_gaic as gaic,r59_model as quality,r60_fcdb_train as train,r63_crop_search as search
 expert=gaic.load();gaic_count=sum(p.numel() for p in expert.parameters());tree=joblib.load(runtime/'autodl-fs/r60_20261022/fcdb_ranker.joblib');tree.n_jobs=1
 common=dict(sys=sys,time=time,math=__import__('math'),json=json,Path=Path,np=np,torch=torch,cv2=cv2,av=av,joblib=joblib,read=visual.read,save=visual.save,sha=sha,features=train.features,proposals=quality.proposals,frozen_features=quality.frozen_features,path_optimize=quality.path_optimize,legal=quality.legal,FORMAL_PARENTS={},R60_RECORDS={},gaic=gaic)
 r60=selected_module('r60_rebuild',prod/'r62_baseline_produce.py',common,['plans','compose','IoUUtility'])
 r63src=runtime/'autodl-fs/r63_20261023/PRODUCTION_A_TREE/r63_produce.py'
 r63=selected_module('r63_rebuild',r63src,dict(common,search_core=search,R63_CHOICE='A_TREE'),['plans','compose','IoUUtility'])
 from r66_rank_inference import SemanticUtility,plans
 import r66_rank_inference as seq
 r66=selected_module('r66_rebuild',prod/'r64_produce.py',dict(common,plans=plans),['compose','contract'])
 contexts=[]
 for row in rows:
  source=base/'visual'/row['id'];semantic=visual.read(source/'SEMANTIC.json');frames=upstream.temporal_pool(source);ratios=[row['targetRatioWH']]
  parent,counts=upstream.compose(row,source,semantic,frames,public=False)
  r60.FORMAL_PARENTS[row['video_id']]=parent[0];r63.FORMAL_PARENTS[row['video_id']]=parent[0]
  d60=runtime/'autodl-fs/r60_20261022/R60-HUMAN-CROP-IOU-V2/PUBLIC/work'/row['id'];d60.mkdir(parents=True,exist_ok=True);r60.ROOT=d60
  for n in ['SPATIAL.npz','SAM_PACKED.npz']:link(source/n,d60/n)
  print('STAGE_R60_REBUILD',row['id'],flush=True)
  records,c=r60.compose(row,source,semantic,frames,r60.IoUUtility(tree),expert,ratios,d60,False,'iou')
  r63.R60_RECORDS[row['video_id']]=records[0]
  d63=work/'r63/PUBLIC/work'/row['id'];d63.mkdir(parents=True,exist_ok=True);r63.ROOT=d63
  print('STAGE_R63_REBUILD',row['id'],flush=True)
  r63.plans(row,source,semantic,r63.IoUUtility(tree),expert,ratios,d63,'iou')
  contexts.append((row,source,semantic,frames))
 del expert;gc.collect();torch.cuda.empty_cache()
 checkpoint=prod/'weights/checkpoint.json' if (prod/'weights/checkpoint.json').exists() else prod/'weights/decision.joblib'
 print('STAGE_R66_MODELS',flush=True);seq.CACHE=work/'r63';model=SemanticUtility(checkpoint)
 save(work/'CURRENT_PARAMETERS.json',dict(GAIC=gaic_count,DINOv2=sum(p.numel() for p in model.encoder.parameters()),CLIP=sum(p.numel() for p in model.scenes.model.parameters()),upstream=json.loads((work/'UPSTREAM_PARAMETERS.json').read_text('utf8')),temporal=visual.read(base/'TEMPORAL_DONE.json')['parameter_counts'],spatial=visual.read(base/'SPATIAL_DONE.json')['parameter_counts']))
 output=[];counters=[]
 for row,source,semantic,frames in contexts:
  folder=work/'sequence/PUBLIC/work'/row['id'];folder.mkdir(parents=True,exist_ok=True)
  for n in ['SPATIAL.npz','SAM_PACKED.npz']:link(source/n,folder/n)
  print('STAGE_R66_SEQUENCE_TRACKING',row['id'],flush=True)
  records,c=r66.compose(row,source,semantic,frames,model,None,[row['targetRatioWH']],folder,False,'iou')
  r66.contract(records[0],row);output.extend(records);counters.append(c)
 # Declared complete-system conservative upper bound in billions; no recount.
 for record in output:
  record['model_params_b']=4.098220951
 target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(''.join(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n' for r in output),encoding='utf8')
 from validate_independent import validate
 val=validate(target,work/'INPUT_IDENTITY.json')
 assert val['success'];save(work/'INDEPENDENT_VALIDATOR.json',val)
 from pipeline_receipts import record_capabilities
 capabilities=record_capabilities(work,base,rows,counters,fresh_runtime=fresh_runtime)
 save(work/'CAPABILITIES.json',capabilities)
 # Individual inputs may legitimately have no LK attempts (e.g. one frame or
 # empty foreground). Record that explicitly without failing a valid crop.
 loaded=[]
 for name,module in list(sys.modules.items()):
  filename=getattr(module,'__file__',None)
  if filename and Path(filename).resolve().is_relative_to(runtime):loaded.append(dict(module=name,path=Path(filename).resolve().relative_to(runtime).as_posix()))
 save(work/'LOADED_IMPLEMENTATIONS.json',dict(rows=loaded))
 save(work/'RUN_DONE.json',dict(success=True,rows=len(rows),predictions=sum(len(r['predictions']) for r in output),seconds=time.time()-started,output_sha256=sha(target),checkpoint_sha256=sha(checkpoint),checkpoint_format='split' if checkpoint.suffix=='.json' else 'combined',official_validator_used=False,independent_contract='PASS',capabilities='CAPABILITIES.json',temporal_backend=args.temporal_backend,temporal_segments='TEMPORAL_SEGMENTS.json',counters=counters,public=args.public,low_memory_offload=args.low_memory,frozen_jsonl_read=False,clean_environment=False))
 print('RAW_INPUT_END_TO_END_PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--package',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--models',help='JSON model component and implementation paths');p.add_argument('--work-dir',required=True);p.add_argument('--index');p.add_argument('--video-root');p.add_argument('--output');p.add_argument('--public',action='store_true');p.add_argument('--low-memory',action='store_true');p.add_argument('--prepare-only',action='store_true')
 add_backend_arguments(p);args=p.parse_args();backend_options(args,p)
 try:run(args)
 except BaseException:
  save(Path(args.work_dir)/'RUN_FAILED.json',dict(at=time.time(),traceback=traceback.format_exc(),frozen_jsonl_read=False));raise
