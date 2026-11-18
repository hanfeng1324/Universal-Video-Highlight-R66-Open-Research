# Modified locally: PyAV loader, HF CLIP, normalization fix and chunked inference.
# Inherits Lighthouse Apache-2.0 and CG/QD/Moment-DETR MIT; see THIRD_PARTY_NOTICES.md.
"""Local CG-DETR adapter. Full official backbone, CLIP B/32, native FFmpeg filters.

Uses checkpoint-training normalization (the upstream demo computes but discards it).
No older video features are reused. HF/OpenAI CLIP parameter mapping checked separately.
"""
import os,sys,json,hashlib,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BIG=Path('/root/autodl-tmp/r53_autonomous_20261020/assets/r22_20261016');OUT=ROOT/'outputs/r22_20261016'
sys.path.insert(0,str(ROOT/'aic_highlight/outputs/OPEN_SOURCE_SMALL_MODEL_AUDIT_R0/runtime_deps'))
sys.path.insert(0,str(BIG/'lighthouse'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
import av
from transformers import CLIPModel,CLIPTokenizerFast
from lighthouse.common.cg_detr import build_model
ENCODER=ROOT/'aic_highlight/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
WEIGHT=BIG/'weights/clip_cg_detr_qvhighlight.ckpt'
GENERIC_QUERIES=[
 'The main visible event, action or interaction unfolds, including preparation, the main moment and its outcome.',
 'The most interesting visible movement, behavior, interaction or change in the scene, including natural events.',
]
def event_query(plan):
 event=str(plan.get('event','')).strip()
 return ('The complete visible event: '+event+' Include necessary preparation, interaction and outcome.') if event else GENERIC_QUERIES[0]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def video_info(path):
 with av.open(str(path)) as c:
  s=c.streams.video[0]
  return dict(width=s.width,height=s.height,duration=float(s.duration*s.time_base) if s.duration else float(c.duration/1e6),frame_count=s.frames,fps=float(s.average_rate),start_time=float(s.start_time*s.time_base) if s.start_time else 0.)

def sampled_rgb(path):
 """Same fps/scale/center-crop sequence as upstream CLIPLoader via PyAV."""
 meta=video_info(path);h,w=meta['height'],meta['width'];size=224
 hh,ww=(int(h*size/w),size) if h>=w else (size,int(w*size/h))
 g=av.filter.Graph();nodes=None;images=[];times=[]
 def drain():
  while True:
   try:
    out=g.pull();images.append(out.to_ndarray(format='rgb24'));times.append(float(out.time))
   except (av.error.BlockingIOError,av.error.EOFError):break
 with av.open(str(path)) as c:
  st=c.streams.video[0]
  for frame in c.decode(video=0):
   if nodes is None:
    nodes=[g.add_buffer(template=frame),g.add('fps','fps=1/2'),g.add('scale',f'{ww}:{hh}'),g.add('crop',f'224:224:{int((ww-size)/2)}:{int((hh-size)/2)}'),g.add('buffersink')]
    g.link_nodes(*nodes);g.configure()
   g.push(frame);drain()
  g.push(None);drain()
 assert images and len(images)==len(times)
 return np.stack(images),np.array(times),meta

class Runtime:
 def __init__(self,load_cg=True):
  torch.set_num_threads(4);torch.manual_seed(22);self.start=time.time()
  assert torch.cuda.is_available();self.device='cuda'
  if load_cg:
   ckpt=torch.load(WEIGHT,map_location='cpu',weights_only=False);args=ckpt['opt'];args.device=self.device
   self.args=args;assert args.v_feat_dim==514 and args.t_feat_dim==512 and args.clip_length==2
   self.net,_=build_model(args);self.net.load_state_dict(ckpt['model'],strict=True)
   self.net=self.net.to(self.device).eval().requires_grad_(False)
  self.clip=CLIPModel.from_pretrained(ENCODER,local_files_only=True,attn_implementation='eager').to(self.device).eval().requires_grad_(False)
  self.tok=CLIPTokenizerFast.from_pretrained(BIG/'clip_tokenizer',local_files_only=True) if load_cg else None
  self.text_cache={}
  self.mean=torch.tensor([.48145466,.4578275,.40821073],device=self.device).view(1,3,1,1)
  self.std=torch.tensor([.26862954,.26130258,.27577711],device=self.device).view(1,3,1,1)
  print('MODEL_LOADED',round(time.time()-self.start,2),flush=True)
 @torch.inference_mode()
 def features(self,path,dest):
  if dest.exists():
   v=np.load(dest);return v['features'],v['sample_seconds'],read(dest.with_suffix('.json'))
  start=time.time();images,seconds,meta=sampled_rgb(path);parts=[]
  for k in range(0,len(images),24):
   x=torch.from_numpy(images[k:k+24]).permute(0,3,1,2).to(self.device,dtype=torch.float32)/255
   y=self.clip.get_image_features(pixel_values=(x-self.mean)/self.std)
   if not isinstance(y,torch.Tensor):y=y.pooler_output
   parts.append(y.cpu().float().numpy())
  feats=np.concatenate(parts)
  dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,features=feats,sample_seconds=seconds)
  meta.update(source_sha256=sha(path),source_path=str(path),feature_seconds=time.time()-start,feature_contract='CLIP_B32_full_image_projection512; PyAV_native_ffmpeg_fps1/2_scale224_center_crop; cells[2i,2i+2); L2_before_CG; HF_checkpoint_same_OpenAI_B32; no_old_features')
  save(dest.with_suffix('.json'),meta);return feats,seconds,meta
 @torch.inference_mode()
 def text(self,query):
  if query not in self.text_cache:
   ids=self.tok(query,truncation=True,max_length=77,add_special_tokens=True)['input_ids']
   # OpenAI tokenize pads with zero, unlike the HF default EOS padding.
   ids=torch.tensor([ids+[0]*(77-len(ids))],device=self.device)
   txt=self.clip.text_model(input_ids=ids).last_hidden_state.float()
   self.text_cache[query]=(F.normalize(txt,dim=-1,eps=1e-5),(ids!=0).float())
  return self.text_cache[query]
 @torch.inference_mode()
 def predict(self,features,query):
  # Independent chunks avoid exceeding learned 75-clip positional range.
  results=[];scores=[]
  txt,mask=self.text(query)
  for off in range(0,len(features),75):
   v=F.normalize(torch.as_tensor(features[off:off+75],device=self.device).float(),dim=-1,eps=1e-5)
   n=len(v);tef=torch.stack([torch.arange(n,device=self.device)/n,(torch.arange(n,device=self.device)+1)/n],-1)
   v=torch.cat([v,tef],-1).unsqueeze(0)
   o=self.net(src_vid=v,src_vid_mask=torch.ones(1,n,device=self.device),src_txt=txt,src_txt_mask=mask,src_aud=None,vid=None,qid=None)
   sal=o['saliency_scores'][0].cpu().numpy();assert len(sal)==n and np.isfinite(sal).all();scores.extend(sal.tolist())
   spans=o['pred_spans'][0];ab=torch.stack([spans[:,0]-spans[:,1]/2,spans[:,0]+spans[:,1]/2],-1).clamp(0,1)*n*2+off*2
   confidence=o['pred_logits'][0].softmax(-1)[:,0]
   results.extend(torch.cat([ab,confidence[:,None]],-1).cpu().tolist())
  return dict(query=query,pred_saliency_scores=scores,pred_relevant_windows=sorted(results,key=lambda x:-x[2]),cell_seconds=2,chunk_max_cells=75,query_free=False)
 def accounting(self):
  return dict(CG_DETR_parameters=sum(p.numel() for p in self.net.parameters()),CLIP_parameters=sum(p.numel() for p in self.clip.parameters()),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),torch=torch.__version__,weight_sha256=sha(WEIGHT),checkpoint_options=dict(self.args),load_to_now_seconds=time.time()-self.start)
