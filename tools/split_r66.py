"""Split a trusted local R66 bundle without training or granting publication rights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from r66_checkpoint import split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = split(args.checkpoint, args.output)
    print(json.dumps(dict(success=True, parts=list(result['parts']), redistribution=result['redistribution'])))
