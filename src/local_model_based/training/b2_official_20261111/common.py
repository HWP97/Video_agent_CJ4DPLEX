"""공식 train 재실행의 읽기 경계와 재개 가능한 파일 저장."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/RG2/my_works/b2/results/2026-11-05/002_veatic_forecast')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8388608),b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)

def freeze_contract(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('resume_contract_mismatch:'+str(path))
    else:save(path,value)
    return digest(path)

def guard(stage):
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('network_forbidden')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            writing=bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and not path.is_relative_to(ROOT) and path!=Path('/dev/null'):
                raise PermissionError('write_outside_owned_run:'+str(path))
            if '/sol/' in str(path) or '/api_based/' in str(path):
                raise PermissionError('API_asset_forbidden')
            if stage in ('extract','infer') and ('labels_' in path.name or 'rating_averaged' in path.parts):
                raise PermissionError('GT_forbidden_during_inference')
            if stage=='train' and path.name=='labels_eval.json':
                raise PermissionError('evaluation_GT_forbidden_during_training')
            if stage=='train' and ('rating_averaged' in path.parts or
                    (path.name.startswith('labels_') and path not in (ROOT/'labels_train.json',ROOT/'labels_dev.json'))):
                raise PermissionError('non_official_training_labels_forbidden')
    sys.addaudithook(audit)
