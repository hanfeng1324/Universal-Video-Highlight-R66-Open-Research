"""Video -> complete R66 temporal/spatial decisions -> cropped MP4.

Requires real, locally authorized R66 components; no substitute model mode.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--ratio',choices=['9:16','16:9'],default='9:16')
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--models',help='JSON model paths; default: local_assets in this checkout')
    parser.add_argument('--low-memory',action='store_true',help='Same BF16 Qwen checkpoint with CPU offload for 8GB GPUs')
    parser.add_argument('--public',action='store_true',help='Input is a lawful public demonstration video')
    parser.add_argument('--render',action='store_true',help='Compatibility flag; MP4 export is now the default')
    parser.add_argument('--no-render',action='store_true',help='Only write decisions and execution evidence')
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'code'))
    from temporal_segments import add_backend_arguments, backend_options, TemporalSegmenter
    add_backend_arguments(parser)
    args=parser.parse_args()
    backend_options(args, parser)
    if not __debug__: parser.error('Assertions are part of the frozen contracts; do not use python -O or PYTHONOPTIMIZE')
    video,work=Path(args.video).resolve(),Path(args.work_dir).resolve()
    if not video.is_file(): parser.error('Video does not exist: '+str(video))
    if work.exists(): parser.error('Choose a new work directory; existing results are preserved')
    from model_assets import check, configuration
    try:
        result=check(root,configuration(args.models))
    except (OSError,ValueError,KeyError) as exc:
        parser.error('Invalid model configuration: '+str(exc))
    problems=[r for r in result['files'] if r['status']!='OK']
    if problems:
        for row in problems[:15]: print(row['status'],row.get('component'),row['resolved'],file=sys.stderr)
        parser.error('Missing/incompatible R66 assets. Use tools/fetch_models.py --list, configure --models, and see README.md.')
    try:
        TemporalSegmenter(args.temporal_backend, args.legacy_kts_dir)
    except (ImportError, OSError, ValueError) as exc:
        parser.error('Temporal backend unavailable: '+str(exc))
    try:
        import torch
    except ImportError:
        parser.error('Install the CUDA runtime with tools/install_runtime.py before inference')
    if not torch.cuda.is_available(): parser.error('A CUDA GPU is required by this full pipeline')
    if not args.low_memory and torch.cuda.mem_get_info()[0] <= 22*2**30:
        parser.error('Less than 22GiB free CUDA memory; use --low-memory with the same weights and CPU offload')
    work.mkdir(parents=True)
    (work/'PREFLIGHT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    index,output=work/'input.jsonl',work/'predictions.jsonl'
    index.write_text(json.dumps(dict(video_id='user_video',video=video.name,targetRatioWH=list(map(int,args.ratio.split(':')))))+'\n',encoding='utf-8')
    command=[sys.executable,str(root/'code/run_inference.py'),'--package',str(root),'--index',str(index),'--video-root',str(video.parent),'--work-dir',str(work),'--output',str(output)]
    if args.models: command+=['--models',str(Path(args.models).resolve())]
    command+=['--temporal-backend',args.temporal_backend]
    if args.legacy_kts_dir: command+=['--legacy-kts-dir',str(args.legacy_kts_dir)]
    if args.low_memory: command+=['--low-memory']
    if args.public: command+=['--public']
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(command,check=True,env=env)
    if not args.no_render:
        subprocess.run([sys.executable,str(root/'tools/render_video.py'),'--video',str(video),'--predictions',str(output),'--output',str(work/'highlight.mp4')],check=True,env=env)
    print('Completed:',work/'highlight.mp4' if not args.no_render else output)


if __name__=='__main__': main()
