"""Install the reference CUDA runtime into a new Python 3.12 virtual environment."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv',type=Path,default=ROOT/'.venv-inference')
    p.add_argument('--plan',action='store_true',help='Show commands without creating or installing anything')
    a=p.parse_args()
    target=a.venv.resolve()
    executable=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    commands=[
        [str(executable),'-m','pip','install','torch==2.9.0','torchvision==0.24.0','--index-url','https://download.pytorch.org/whl/cu128'],
        [str(executable),'-m','pip','install','-r',str(ROOT/'environment/requirements.txt')],
        [str(executable),'-m','pip','check'],
        [str(executable),'-c',"import torch; assert torch.cuda.is_available(), 'CUDA is not available; check the NVIDIA driver'; print(torch.__version__, torch.cuda.get_device_name(0))"],
    ]
    if a.plan:
        import json
        print(json.dumps(dict(python='3.12',create_venv=str(target),commands=commands,weights_downloaded=False),indent=2));return
    if sys.version_info[:2]!=(3,12): p.error('Run this installer with Python 3.12')
    if target.exists(): p.error('Choose a new venv path; existing environments are preserved')
    if 'FINAL_20261024' in target.parts: p.error('Competition archive is read-only')
    venv.EnvBuilder(with_pip=True).create(target)
    for command in commands: subprocess.run(command,check=True)
    print('Runtime installed. Acquire authorized assets, run tools/preflight.py --inference --environment, then infer.py.')


if __name__=='__main__': main()
