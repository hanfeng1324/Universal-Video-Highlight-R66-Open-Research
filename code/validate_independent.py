"""Independent public/new-output contract checker, no model imports."""
import argparse,json,math,hashlib
from pathlib import Path
def validate(output,identity):
 rows=json.loads(Path(identity).read_text('utf8'))['rows'];meta={r['video_id']:r for r in rows}
 seen=set();count=0
 for line in Path(output).read_text('utf8').splitlines():
  r=json.loads(line);vid=r['video_id'];assert vid in meta and vid not in seen;seen.add(vid);m=meta[vid]
  assert r['targetRatioWH']==m['targetRatioWH'];rw,rh=r['targetRatioWH'];last=-1
  assert isinstance(r['predictions'],list)
  for p in r['predictions']:
   assert set(p)=={'frame','bboxes'};f=p['frame'];assert type(f) is int and last<f<m['frames'];last=f
   b=p['bboxes'];assert isinstance(b,list) and len(b)==3 and all(type(v) is int for v in b)
   x,y,w=b;assert x>=0 and y>=0 and w>0 and x+w<=m['width'] and y+math.ceil(w*rh/rw)<=m['height'];count+=1
 assert seen==set(meta)
 return dict(success=True,videos=len(seen),predictions=count,output_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(),checks=['all_expected_unique_ids','exact_input_target_ratio','strictly_increasing_unique_native_frames','integer_xyw','positive_width','full_aspect_converted_bounds'],model_imports=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--identity',required=True);p.add_argument('--report',required=True);a=p.parse_args();q=validate(a.output,a.identity);Path(a.report).write_text(json.dumps(q,indent=2),encoding='utf8');print(q)
