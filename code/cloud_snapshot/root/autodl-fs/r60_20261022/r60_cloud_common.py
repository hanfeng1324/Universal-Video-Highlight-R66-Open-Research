"""Persistent cloud-local state helpers. No network or desktop dependencies."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text('utf8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        tmp = p.with_name(p.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with tmp.open('w', encoding='utf8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(p)
            return
        except FileNotFoundError:
            if attempt==4:raise
            time.sleep(.05*(attempt+1))


def progress(stage, **kw):
    save(ROOT / (stage.upper()+'_PROGRESS.json'), dict(stage=stage, heartbeat=time.time(), **kw))
    if '_shard_' not in stage and stage not in ('student','spatial'):
        save(ROOT / 'PROGRESS.json', dict(stage=stage, heartbeat=time.time(), **kw))
    print(json.dumps(dict(stage=stage, **kw)), flush=True)
