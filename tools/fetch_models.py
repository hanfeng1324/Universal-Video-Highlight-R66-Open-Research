"""Fetch identity-verified upstream models at frozen revisions for research.

No R66 substitute weights and no restricted model mirrors. Existing files are
preserved. --dry-run prints the exact transfer plan without network activity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from model_assets import component, configuration, resolve
"""Bounded, read-only HTTP ZIP access. Never fall back to a whole archive transfer."""
import io
import re
import urllib.parse
import urllib.request
import zipfile
import urllib.error
import http.client
import socket
import time
from html.parser import HTMLParser


def open_upstream(request,timeout=45):
    """Retry connection establishment only; never restart a partially read body."""
    for attempt in range(3):
        try: return urllib.request.urlopen(request,timeout=timeout)
        except (urllib.error.URLError,TimeoutError,socket.timeout,ConnectionError) as exc:
            if isinstance(exc,urllib.error.HTTPError) and exc.code not in {408,429,500,502,503,504}: raise
            if attempt==2: raise
            print('RETRY_CONNECTION','attempt',attempt+2,flush=True)
            time.sleep(attempt+1)


class _Form(HTMLParser):
    def __init__(self):
        super().__init__(); self.fields = {}; self.action = None; self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.active = True; self.action = attrs.get('action')
        if self.active and tag == 'input' and attrs.get('type') == 'hidden':
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form': self.active = False


def google_drive_url(file_id):
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    with open_upstream(url, timeout=45) as response:
        body = response.read(32769)
    if len(body) > 32768: raise RuntimeError('Expected a small Drive confirmation page')
    form = _Form(); form.feed(body.decode('utf-8'))
    if form.action != 'https://drive.usercontent.google.com/download' or form.fields.get('id') != file_id:
        raise RuntimeError('Drive confirmation changed; refusing an unverified download route')
    return form.action + '?' + urllib.parse.urlencode(form.fields)


class RangeReader(io.RawIOBase):
    def __init__(self, url, budget=400 * 2**20):
        self.url = url; self.position = 0; self.transferred = 0; self.budget = budget
        _, total = self._range(0, 1)
        self.size = total

    def _range(self, start, size):
        if size > 16 * 2**20:
            raise RuntimeError('ZIP range transfer budget exceeded')
        req = urllib.request.Request(self.url, headers={
            'Range': f'bytes={start}-{start+size-1}', 'Accept-Encoding': 'identity'})
        for attempt in range(3):
            if self.transferred + size > self.budget: raise RuntimeError('ZIP range transfer budget exceeded')
            # Charge the complete requested range even on a partial failed read.
            self.transferred += size
            try:
                with urllib.request.urlopen(req, timeout=45) as response:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, start+size-1):
                        raise RuntimeError('Server did not honor Range; whole archive transfer refused')
                    body = response.read(size + 1)
                if len(body) != size: raise http.client.IncompleteRead(body,size-len(body))
                return body, int(match.group(3))
            except (urllib.error.URLError,TimeoutError,socket.timeout,http.client.IncompleteRead,ConnectionError) as exc:
                if isinstance(exc,urllib.error.HTTPError) and exc.code not in {408,429,500,502,503,504}: raise
                if attempt==2: raise
                print('RETRY_RANGE',start,size,'attempt',attempt+2,flush=True)
                time.sleep(attempt+1)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= position <= self.size: raise ValueError('Invalid remote ZIP seek')
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0: size = self.size - self.position
        size = min(size, self.size - self.position)
        if not size: return b''
        body, total = self._range(self.position, size)
        if total != self.size: raise RuntimeError('Remote ZIP size changed')
        self.position += size
        return body


def copy_member(url, member, destination, expected_bytes, expected_archive_bytes=None):
    with RangeReader(url) as reader:
        if expected_archive_bytes and reader.size != expected_archive_bytes:
            raise RuntimeError('Upstream archive size changed')
        with zipfile.ZipFile(reader) as archive:
            matches = [x for x in archive.infolist() if x.filename == member]
            if len(matches) != 1 or matches[0].file_size != expected_bytes:
                raise RuntimeError('Unexpected ZIP member identity')
            with archive.open(matches[0]) as source:
                total = 0
                while True:
                    block = source.read(4 * 2**20)
                    if not block: break
                    total += len(block)
                    if total > expected_bytes: raise RuntimeError('Member size exceeded')
                    destination.write(block)
            if total != expected_bytes: raise RuntimeError('Incomplete ZIP member')
        return reader.transferred


def copy_direct_remainder(url,destination,expected_bytes,start):
    """Resume only an exact-size original file; final full SHA is still mandatory."""
    with RangeReader(url,budget=expected_bytes+max(64*2**20,expected_bytes//10)) as reader:
        if reader.size!=expected_bytes: raise RuntimeError('Direct upstream file size changed')
        reader.seek(start)
        while reader.tell()<reader.size:
            destination.write(reader.read(min(4*2**20,reader.size-reader.tell())))
        return reader.transferred


def copy_direct(url,destination,expected_bytes):
    size=0
    with open_upstream(url,timeout=60) as source:
        try:
            while block:=source.read(8*2**20):
                if size+len(block)>expected_bytes:
                    raise RuntimeError('Upstream size exceeded; partial preserved')
                destination.write(block)
                size+=len(block)
        except (TimeoutError,ConnectionError,http.client.IncompleteRead) as exc:
            if not 0<size<expected_bytes: raise
            print('DIRECT_BODY_INTERRUPTED',type(exc).__name__,size,flush=True)
    if size==expected_bytes: return None
    if not size: raise RuntimeError('Empty upstream response; partial preserved')
    print('RESUME_DIRECT_RANGE',size,flush=True)
    return copy_direct_remainder(url,destination,expected_bytes,size)


SOURCES={
    'qwen':dict(repo='Qwen/Qwen2.5-VL-3B-Instruct',revision='66285546d2b821cf421d4f5eb2576359d3770cd3',license='Qwen Research License; research/evaluation noncommercial only',status='UPSTREAM_RESEARCH_ONLY'),
    'dinov2':dict(repo='facebook/dinov2-base',revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415',license='Apache-2.0; keep license/notices',status='UPSTREAM_ALLOWED'),
    'sam2':dict(url='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',license='Apache-2.0; keep license/notices',status='UPSTREAM_ALLOWED_EXPECT_FROZEN_HASH'),
    'clip':dict(repo='openai/clip-vit-base-patch32',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',license='Official research acquisition; upstream MIT notices',status='EXACT_OFFICIAL_RESEARCH_FETCH'),
    'clip_tokenizer':dict(repo='openai/clip-vit-base-patch32',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',license='Upstream CLIP MIT notices',status='FROZEN_HASH_REQUIRED'),
    'googlenet':dict(url='https://download.pytorch.org/models/googlenet-1378be20.pth',license='Official research acquisition; no mirror grant asserted',status='EXACT_OFFICIAL_RESEARCH_FETCH'),
    'unisal':dict(prefix='https://raw.githubusercontent.com/rdroste/unisal/7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9/training_runs/pretrained_unisal/',license='Apache-2.0',status='EXACT_OFFICIAL_FETCH'),
    'gaic':dict(url='https://raw.githubusercontent.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch/f6f925dadaf3c975f84c422ad575e0c92dc87320/pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth',license='Author-linked original research distribution; retain GAIC notices',status='EXACT_AUTHOR_LINKED_RESEARCH_FETCH'),
    'cg_detr':dict(drive_id='1jxs_bvwttXTF9Lk3aKLohkqfYOonLyrO',member='results/cg_detr/qvhighlight/clip/best.ckpt',archive_bytes=44692159878,license='Lighthouse original research acquisition; no mirror grant asserted',status='EXACT_MEMBER_SHA256_VERIFIED'),
    'uvcom':dict(drive_id='1jxs_bvwttXTF9Lk3aKLohkqfYOonLyrO',member='results/uvcom/qvhighlight/clip/best.ckpt',archive_bytes=44692159878,license='Lighthouse original research acquisition; no mirror grant asserted',status='EXACT_MEMBER_SHA256_VERIFIED'),
    'dsnet':dict(archive_url='https://www.dropbox.com/s/2hjngmb0f97nxj0/pretrain_af_basic.zip?dl=1',member='pretrain_af_basic/checkpoint/tvsum.yml.0.pt',license='DSNet original research acquisition; no mirror grant asserted',status='EXACT_MEMBER_SHA256_VERIFIED; RANGE_REQUIRED'),
}
BLOCKED=['fcdb','dino_projection','r66']


def plan(models, config):
    rows=json.loads((ROOT/'config/weight_collection.json').read_text(encoding='utf-8'))['rows']
    result=[]
    for row in rows:
        key=component(row['path'])
        if key not in models or Path(row['path']).name in {'model_manifest.json','source_metadata.json','PRETRAINED_SOURCE.json'}: continue
        source=SOURCES[key]
        name=Path(row['path']).name
        url=source.get('url') or source.get('archive_url')
        if source.get('drive_id'): url='https://drive.google.com/file/d/'+source['drive_id']+'/view'
        if source.get('prefix'): url=source['prefix']+urllib.parse.quote(name)
        if not url: url='https://huggingface.co/'+source['repo']+'/resolve/'+source['revision']+'/'+urllib.parse.quote(name)
        result.append(dict(component=key,url=url,target=str(resolve(ROOT,row['path'],config)),sha256=row['sha256'],bytes=row['bytes'],**({k:source[k] for k in ('member','drive_id','archive_bytes') if k in source})))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true',help='Print supported sources and genuine remaining blockers')
    p.add_argument('--model',action='append',choices=sorted(SOURCES),help='Repeat for each allowed model')
    p.add_argument('--models',help='Same model-path JSON accepted by infer.py')
    p.add_argument('--file',action='append',help='Only named manifest files (repeat); useful for small configuration/license transfers')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--report',type=Path,help='Write an incremental SHA-verified acquisition receipt')
    p.add_argument('--resume-partials',action='store_true',help='Preserve failed direct-file partials and resume with strict HTTP Range plus full final SHA; ZIP partials are never resumed')
    p.add_argument('--failed-dir',type=Path,help='Optional separate directory for preserved failed direct-file partials')
    p.add_argument('--accept-qwen-research-license',action='store_true')
    a=p.parse_args()
    if a.list:
        print(json.dumps(dict(allowed=SOURCES,not_hosted=BLOCKED,remaining='Project research parameters are bundled separately from these upstream assets. Run inference preflight after acquisition.'),indent=2));return
    if not a.model: p.error('Specify --list or at least one --model')
    if 'qwen' in a.model and not a.accept_qwen_research_license and not a.dry_run:
        p.error('Read the fixed Qwen LICENSE and supply --accept-qwen-research-license for permitted research/evaluation use')
    rows=plan(set(a.model),configuration(a.models))
    if a.file:
        unknown=set(a.file)-{Path(r['target']).name for r in rows}
        if unknown: p.error('Requested files are not in the frozen model manifest: '+str(sorted(unknown)))
        rows=[r for r in rows if Path(r['target']).name in a.file]
    if a.dry_run:
        print(json.dumps(dict(downloads=rows,total_bytes=sum(r['bytes'] for r in rows),project_parameters_not_fetched=BLOCKED),indent=2));return
    receipt=dict(complete=False,rows=[],source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Original upstream acquisition and SHA verification')
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    def record(row,status,transferred=None):
        receipt['rows'].append(dict(row,status=status,sha256_verified=True,verified_mtime_ns=Path(row['target']).stat().st_mtime_ns,range_transfer_bytes=transferred))
        if a.report:
            a.report.parent.mkdir(parents=True,exist_ok=True)
            a.report.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for row in rows:
        target=Path(row['target'])
        if 'final_20261024' in {part.casefold() for part in target.resolve().parts}: p.error('Competition archive is read-only')
        if target.exists():
            if target.stat().st_size!=row['bytes']: raise RuntimeError('Existing file has unexpected size; preserved: '+str(target))
            h=hashlib.sha256()
            with target.open('rb') as existing:
                for block in iter(lambda:existing.read(8*2**20),b''): h.update(block)
            if h.hexdigest()!=row['sha256']: raise RuntimeError('Existing file hash mismatch; preserved: '+str(target))
            print('EXISTS_VERIFIED_UNCHANGED',target);record(row,'EXISTS_VERIFIED_UNCHANGED');continue
        target.parent.mkdir(parents=True,exist_ok=True)
        pending=target.with_name(target.name+'.download-part')
        h=hashlib.sha256();transferred=None
        backup=None
        if pending.exists():
            if not a.resume_partials or row.get('member'): raise RuntimeError('Partial preserved; explicit direct-file --resume-partials required: '+str(pending))
            if pending.stat().st_size>row['bytes']: raise RuntimeError('Oversized partial preserved; cannot resume')
            failed_dir=a.failed_dir.resolve() if a.failed_dir else pending.parent
            if 'final_20261024' in {part.casefold() for part in failed_dir.parts}: p.error('Competition archive is read-only')
            failed_dir.mkdir(parents=True,exist_ok=True)
            backup=failed_dir/(pending.name+'.failed-'+str(time.time_ns()))
            pending.rename(backup)
            row['resumed_from_bytes']=backup.stat().st_size
            row['failed_partial_preserved_as']=str(backup)
        # Exclusive partial file: interrupted/failed evidence is never overwritten.
        with pending.open('xb') as dst:
            if backup:
                with backup.open('rb') as prefix:
                    for block in iter(lambda:prefix.read(8*2**20),b''):dst.write(block)
                transferred=copy_direct_remainder(row['url'],dst,row['bytes'],backup.stat().st_size)
                print('RESUMED_RANGE_BYTES',transferred)
            elif row.get('member'):
                url=google_drive_url(row['drive_id']) if row.get('drive_id') else row['url']
                transferred=copy_member(url,row['member'],dst,row['bytes'],row.get('archive_bytes'))
                print('RANGE_BYTES',transferred)
            else:
                transferred=copy_direct(row['url'],dst,row['bytes'])
        with pending.open('rb') as downloaded:
            size=0
            for block in iter(lambda:downloaded.read(8*2**20),b''):h.update(block);size+=len(block)
        if h.hexdigest()!=row['sha256'] or size!=row['bytes']:
            raise RuntimeError('Upstream file differs from frozen R66 identity; partial preserved: '+str(pending))
        # Hard-link only after verification, refusing replacement even if a target
        # was concurrently created. Preserve partial on failure.
        os.link(pending,target)
        pending.unlink()
        record(row,'DOWNLOADED_VERIFIED',transferred)
        print('VERIFIED',target)
    receipt['complete']=True
    if a.report: a.report.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('Requested upstream subset SHA-verified. Run tools/preflight.py --inference --environment next.')


if __name__=='__main__': main()
