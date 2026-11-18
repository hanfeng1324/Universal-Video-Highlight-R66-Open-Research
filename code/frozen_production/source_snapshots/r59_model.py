"""Frozen published GAIC ROI/ROD features -> genuinely trained crop quality."""
import math
import numpy as np
import torch
from torch import nn

DIM=1296+31
class Quality(nn.Module):
 def __init__(self,variant='early'):
  super().__init__();self.variant=variant
  self.encoder=nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,384),nn.GELU(),nn.Dropout(.12),nn.Linear(384,128),nn.GELU())
  self.context=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,384,.1,batch_first=True,norm_first=True),2,enable_nested_tensor=False) if variant=='context' else nn.Identity()
  self.geometry=nn.Sequential(nn.Linear(31,128),nn.GELU(),nn.Linear(128,128)) if variant=='context' else None
  self.iou=nn.Linear(128,1);self.mos=nn.Linear(128,1)
 def forward(self,x):
  z=self.encoder(x)
  if self.geometry is not None:z=z+self.geometry(x[...,-31:])
  z=self.context(z)
  return self.iou(z).squeeze(-1).sigmoid(),self.mos(z).squeeze(-1)

def legal(cx,cy,width,ratio,w,h):
 cw=max(2,min(int(round(width)),w,int(h*ratio[0]//ratio[1])));ch=math.ceil(cw*ratio[1]/ratio[0]);x=int(np.clip(round(cx-cw/2),0,w-cw));y=int(np.clip(round(cy-ch/2),0,h-ch))
 return [x,y,x+cw,y+ch]

def proposals(w,h,ratio,centers=(),subjects=()):
 mx=min(w,int(h*ratio[0]//ratio[1]));bb=[]
 # Broad and tight real legal crops, no parent displacement/scale floor.
 for scale in [1.,.86,.70,.55]:
  cw=max(2,round(mx*scale));ch=math.ceil(cw*ratio[1]/ratio[0])
  for x in np.linspace(cw/2,w-cw/2,9 if cw<w else 1):
   for y in np.linspace(ch/2,h-ch/2,5 if ch<h else 1):bb.append(legal(x,y,cw,ratio,w,h))
  for x,y in list(centers)+[(w/2,h/2)]:bb.append(legal(x,y,cw,ratio,w,h))
 for b in subjects:
  a=np.asarray(b);cx,cy=(a[:2]+a[2:])/2;cw=max(a[2]-a[0],(a[3]-a[1])*ratio[0]/ratio[1])
  for scale in [1.1,1.4,1.8]:bb.append(legal(cx,cy,cw*scale,ratio,w,h))
 return np.unique(np.asarray(bb,dtype=np.int32),axis=0)

def target_iou(bb,gt):
 # RetargetVid evaluation uses inclusive endpoints, retained for comparisons.
 inter=np.maximum(0,np.minimum(bb[:,None,2:],gt[None,:,2:])-np.maximum(bb[:,None,:2],gt[None,:,:2])+1).prod(-1)
 a=(bb[:,2:]-bb[:,:2]+1).prod(-1)[:,None];g=(gt[:,2:]-gt[:,:2]+1).prod(-1)
 return (inter/np.maximum(a+g-inter,1e-8)).mean(-1).astype(np.float32)

def descriptors(boxes,sm,w,h,gaic_score):
 import cv2
 sm=sm.astype(np.float32);sm/=max(float(sm.sum()),1e-8);mh,mw=sm.shape;out=[]
 yy,xx=np.mgrid[:mh,:mw];com=np.array([(xx*sm).sum()/mw,(yy*sm).sum()/mh]);entropy=float(-(sm*np.log(sm+1e-12)).sum()/np.log(sm.size))
 for b,score in zip(boxes,gaic_score):
  x0,y0,x1,y1=b/[w,h,w,h];cx,cy=(x0+x1)/2,(y0+y1)/2;bw,bh=x1-x0,y1-y0
  a,b0=max(0,int(x0*mw)),max(0,int(y0*mh));c,d=min(mw,max(a+1,int(np.ceil(x1*mw)))),min(mh,max(b0+1,int(np.ceil(y1*mh))))
  inside=sm[b0:d,a:c];grid=cv2.resize(inside,(3,3),interpolation=cv2.INTER_AREA).flatten()*inside.size
  mass=float(inside.sum());border=float(inside[:1].sum()+inside[-1:].sum()+inside[:,:1].sum()+inside[:,-1:].sum())
  geom=[x0,y0,x1,y1,cx,cy,bw,bh,bw*bh,np.log(max(bw*w/(bh*h),1e-8))/3,mass,border,entropy,com[0],com[1],abs(cx-com[0]),abs(cy-com[1]),abs(cx-.5),abs(cy-.5),float(score)/5,0.,1.]
  out.append(np.r_[geom,grid])
 return np.asarray(out,np.float32)

@torch.inference_mode()
def frozen_features(model,rgb,boxes,sm):
 import r59_gaic as gaic
 x,s=gaic.preprocess(rgb);bb=np.asarray(boxes,dtype=float)*s;bb[:,:2]=np.floor(bb[:,:2]);bb[:,2:]=np.ceil(bb[:,2:]);f=model.encode(x)
 b=torch.tensor(np.c_[np.zeros(len(bb)),bb],device='cuda',dtype=torch.float32);out=[];scores=[]
 for i in range(0,len(b),48):
  q=b[i:i+48];v=torch.cat((gaic.align(f,q),gaic.align(f,q,True)),1)
  out.append(v.flatten(1).cpu().numpy());scores.append(model.FC_layers(v).flatten().cpu().numpy())
 score=np.concatenate(scores);v=np.concatenate(out);h,w=rgb.shape[:2]
 # Geometry normalized to image; original image coords must match boxes.
 return np.c_[v,descriptors(boxes,sm,w,h,score)].astype(np.float32),score

def path_optimize(boxes,scores,frames,cuts,w,h):
 """Soft continuous trajectory cost; cuts and missing-frame gaps reset path."""
 values=[];back=[]
 for t,(b,s) in enumerate(zip(boxes,scores)):
  b=b/np.array([w,h,w,h]);center=(b[:,:2]+b[:,2:])/2;size=np.maximum(b[:,2:]-b[:,:2],1e-6)
  reset=t==0 or any(frames[t-1]<c<=frames[t] for c in cuts)
  if reset:values.append(np.asarray(s));back.append(None)
  else:
   p=boxes[t-1]/np.array([w,h,w,h]);pc=(p[:,:2]+p[:,2:])/2;ps=np.maximum(p[:,2:]-p[:,:2],1e-6)
   penalty=.18*np.square(pc[:,None]-center[None]).sum(-1)+.035*np.square(np.log(ps[:,None]/size[None])).sum(-1)
   z=values[-1][:,None]-penalty;idx=z.argmax(0);values.append(np.asarray(s)+z[idx,np.arange(len(b))]);back.append(idx)
  values[-1]-=values[-1].max()
 chosen=np.zeros(len(boxes),int);k=int(values[-1].argmax())
 for t in range(len(boxes)-1,-1,-1):
  chosen[t]=k;k=int(back[t][k]) if back[t] is not None else int(values[t-1].argmax()) if t else 0
 return chosen
