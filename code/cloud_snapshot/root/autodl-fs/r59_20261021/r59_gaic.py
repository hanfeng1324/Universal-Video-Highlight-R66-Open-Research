# Local faithful PyTorch inference port of GAIC ROI/ROD; upstream MIT retained.
# This file is excluded from the project MIT proposal; see THIRD_PARTY_NOTICES.md.
"""GAIC published checkpoint, faithful inference port of legacy CUDA ROI/ROD.

No torchvision ROIAlign substitution: original inclusive extent, endpoint grid,
edge extrapolation, outside mask, then 2x2 average are preserved.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

B=Path(__file__).resolve().parent/'data/gaic'
sys.path.insert(0,str(B))
from mobilenetv2 import MobileNetV2

def align(features, boxes, outside=False, size=9, scale=1/16):
    n=len(boxes); _,c,h,w=features.shape; s=size+1
    b=boxes[:,1:]*scale
    t=torch.linspace(0,1,s,device=features.device)
    if outside:
        yy=(t*(h-1.001))[None,:,None].expand(n,s,s)
        xx=(t*(w-1.001))[None,None,:].expand(n,s,s)
    else:
        yy=b[:,1,None,None]+t[None,:,None]*(b[:,3]-b[:,1]+1).clamp_min(0)[:,None,None]
        xx=b[:,0,None,None]+t[None,None,:]*(b[:,2]-b[:,0]+1).clamp_min(0)[:,None,None]
        yy=yy.expand(n,s,s);xx=xx.expand(n,s,s)
    # Original kernel uses min(floor(coord),dim-2), including extrapolation
    # in the final pixel; grid_sample(border) would not reproduce that.
    y0=yy.floor().clamp(0,h-2).long();x0=xx.floor().clamp(0,w-2).long()
    dy=(yy-y0)[:,None];dx=(xx-x0)[:,None]
    f=features[boxes[:,0].long()].flatten(2)
    def at(y,x):return f.gather(2,(y*w+x).flatten(1)[:,None].expand(n,c,s*s)).reshape(n,c,s,s)
    out=at(y0,x0)*(1-dy)*(1-dx)+at(y0,x0+1)*(1-dy)*dx+at(y0+1,x0)*dy*(1-dx)+at(y0+1,x0+1)*dy*dx
    valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
    if outside:valid &= ~((yy>=b[:,1,None,None])&(yy<=b[:,3,None,None])&(xx>=b[:,0,None,None])&(xx<=b[:,2,None,None]))
    return F.avg_pool2d(out*valid[:,None],2,1)

class Base(nn.Module):
    def __init__(self):
        super().__init__();m=MobileNetV2(width_mult=1.)
        self.feature3=nn.Sequential(m.features[:7]);self.feature4=nn.Sequential(m.features[7:14]);self.feature5=nn.Sequential(m.features[14:])
    def forward(self,x):
        a=self.feature3(x);b=self.feature4(a);return a,b,self.feature5(b)

class Ranker(nn.Module):
    def __init__(self):
        super().__init__();self.Feat_ext=Base();self.DimRed=nn.Conv2d(448,8,1)
        self.FC_layers=nn.Sequential(nn.Sequential(nn.Conv2d(16,768,9),nn.BatchNorm2d(768),nn.ReLU(True)),nn.Sequential(nn.Conv2d(768,128,1),nn.BatchNorm2d(128),nn.ReLU(True)),nn.Dropout(.5),nn.Conv2d(128,1,1))
    def encode(self,x):
        a,b,c=self.Feat_ext(x)
        return self.DimRed(torch.cat((F.interpolate(a,scale_factor=.5,mode='bilinear',align_corners=True),b,.5*F.interpolate(c,scale_factor=2.,mode='bilinear',align_corners=True)),1))
    def score(self,f,boxes):
        return self.FC_layers(torch.cat((align(f,boxes),align(f,boxes,True)),1)).flatten()

def load():
    m=Ranker().eval().cuda().requires_grad_(False)
    m.load_state_dict(torch.load(B/'weights.pth',map_location='cpu',weights_only=True),strict=True)
    return m

def preprocess(rgb):
    import cv2
    h,w=rgb.shape[:2];s=256/min(h,w);nh,nw=[round(d*s/32)*32 for d in [h,w]]
    a=cv2.resize(rgb,(nw,nh)).astype(np.float32)/256
    a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(a.transpose(2,0,1).copy())[None].cuda(),np.array([nw/w,nh/h,nw/w,nh/h])

def scores(m,rgb,boxes):
    x,s=preprocess(rgb);bb=np.asarray(boxes)*s
    bb[:,:2]=np.floor(bb[:,:2]);bb[:,2:]=np.ceil(bb[:,2:])
    with torch.inference_mode():
        f=m.encode(x);b=torch.tensor(np.c_[np.zeros(len(bb)),bb],device='cuda',dtype=torch.float32)
        return torch.cat([m.score(f,b[i:i+64]) for i in range(0,len(b),64)]).cpu().numpy()

def verify():
    # Independent scalar transcription of published CUDA, both operators.
    rng=np.random.default_rng(39);a=rng.normal(size=(1,3,13,17)).astype('float32')
    boxes=np.array([[0,0,0,16,12],[0,2.3,1.4,12.1,10.6],[0,10,8,16,12]],'float32');errs=[]
    for outside in [False,True]:
        expected=np.zeros((3,3,10,10),np.float32)
        for k,b in enumerate(boxes):
            for y in range(10):
                for x in range(10):
                    yy=y*(13-1.001)/9 if outside else b[2]+y*max(b[4]-b[2]+1,0)/9
                    xx=x*(17-1.001)/9 if outside else b[1]+x*max(b[3]-b[1]+1,0)/9
                    if yy<0 or yy>=13 or xx<0 or xx>=17 or (outside and b[2]<=yy<=b[4] and b[1]<=xx<=b[3]):continue
                    y0=min(int(np.floor(yy)),11);x0=min(int(np.floor(xx)),15);dy=yy-y0;dx=xx-x0
                    expected[k,:,y,x]=a[0,:,y0,x0]*(1-dy)*(1-dx)+a[0,:,y0,x0+1]*(1-dy)*dx+a[0,:,y0+1,x0]*dy*(1-dx)+a[0,:,y0+1,x0+1]*dy*dx
        expected=(expected[:,:,:-1,:-1]+expected[:,:,1:,:-1]+expected[:,:,:-1,1:]+expected[:,:,1:,1:])/4
        actual=align(torch.tensor(a),torch.tensor(boxes),outside,scale=1).numpy();err=float(abs(expected-actual).max());assert err<1e-5;errs.append(err)
    m=load();return dict(ROI_max_error=errs[0],ROD_max_error=errs[1],strict_checkpoint_load=True,parameters=sum(p.numel() for p in m.parameters()),full_R37_plus_ranker=8706039635+sum(p.numel() for p in m.parameters()),port_not_original_binary=True)

if __name__=='__main__':print(verify())
