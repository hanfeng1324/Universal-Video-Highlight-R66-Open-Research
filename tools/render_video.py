"""Render selected native frame crops to an H.264 MP4 (silent, CFR)."""
from pathlib import Path
import argparse, json, hashlib, math
import av
import cv2
from fractions import Fraction

def render(video,predictions,output,video_id=None):
 rows=[json.loads(x) for x in Path(predictions).read_text('utf8').splitlines() if x.strip()]
 if video_id is not None:rows=[r for r in rows if r['video_id']==video_id]
 if len(rows)!=1:raise ValueError('Select exactly one video record')
 row=rows[0];rw,rh=row['targetRatioWH'];ps=row['predictions']
 frames=[q['frame'] for q in ps]
 if not frames or frames!=sorted(set(frames)):raise ValueError('Invalid selected frame sequence')
 selected={q['frame']:q['bboxes'] for q in ps};target=Path(output)
 if target.exists():raise FileExistsError(target)
 target.parent.mkdir(parents=True,exist_ok=True)
 width,height=(360,640) if rw<rh else (640,360)
 count=0;decoded=0
 with av.open(str(video)) as src:
  rate=src.streams.video[0].average_rate or Fraction(25,1)
  with av.open(str(target),'w') as dst:
   stream=dst.add_stream('libx264',rate=rate);stream.width=width;stream.height=height;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'}
   for i,frame in enumerate(src.decode(video=0)):
    decoded+=1
    if i not in selected:continue
    image=frame.to_ndarray(format='bgr24');h,w=image.shape[:2]
    x,y,cw=map(int,selected[i]);ch=math.ceil(cw*rh/rw)
    if not (cw>0 and ch>0 and x>=0 and y>=0 and x+cw<=w and y+ch<=h):raise ValueError(('Invalid crop',i,x,y,cw,ch,w,h))
    crop=cv2.resize(image[y:y+ch,x:x+cw],(width,height),interpolation=cv2.INTER_AREA)
    out=av.VideoFrame.from_ndarray(crop,format='bgr24');out.pts=count
    for packet in stream.encode(out):dst.mux(packet)
    count+=1
   for packet in stream.encode():dst.mux(packet)
 if count!=len(selected):raise ValueError('Predicted frame outside input')
 with av.open(str(target)) as check:verified=sum(1 for f in check.decode(video=0))
 assert verified==count
 receipt=dict(success=True,input_frames=decoded,selected_frames=count,decoded_output_frames=verified,width=width,height=height,fps=float(rate),seconds=count/float(rate),audio=False,timing='CFR at source average rate; selected native frames concatenated',sha256=hashlib.sha256(target.read_bytes()).hexdigest())
 target.with_suffix('.verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt))
 return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--video-id');a=p.parse_args();render(a.video,a.predictions,a.output,a.video_id)
