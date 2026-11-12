"""관측과 supervision 경계를 공유하는 adapter의 파일 계약."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
from features import Batch, load

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_role(fold,role,targets=False):
    if role not in ('train','dev','eval') or (targets and role=='eval'):
        raise PermissionError('invalid_role_or_eval_supervision')
    plan=json.loads((ROOT/'runplan.json').read_text());entry=plan['folds'][fold-1]
    group=entry[{'train':'train','dev':'development','eval':'evaluation'}[role]]
    batches=[];metadata=[];labels=[]
    for row in plan['groups'][group]:
        name=Path(row['file']).stem;receipt=json.loads((ROOT/'data/receipts'/f'{name}.json').read_text())
        assert receipt['signature']['runplan']==sha(ROOT/'runplan.json')
        inp=ROOT/'data/inputs'/f'{name}_inputs.npz';idx=ROOT/'data/indices'/f'{name}_index.json'
        for path in (inp,idx):assert sha(path)==receipt['outputs'][str(path.relative_to(ROOT))]
        batches.append(load(inp));metadata.extend(json.loads(idx.read_text()))
        if targets:
            path=ROOT/'data/targets'/f'{name}_labels.npz';assert sha(path)==receipt['outputs'][str(path.relative_to(ROOT))]
            labels.append(np.load(path)['xy'])
    batch=Batch(**{k:np.concatenate([getattr(b,k) for b in batches]) for k in Batch.__dataclass_fields__})
    assert len(batch.ego)==len(metadata)
    return batch,metadata,np.concatenate(labels) if targets else None


def anchor(fold,role):
    if role not in ('train','dev','eval'):raise ValueError(role)
    path=ROOT/'attention'/f'fold{fold}'/f'direct_unshared4_{role}.npy'
    receipt=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==receipt['sha256']
    return np.load(path)


def install_guard(stage,fold):
    plan=json.loads((ROOT/'runplan.json').read_text());entry=plan['folds'][fold-1]
    allowed={Path(r['file']).stem+'_labels.npz' for key in ('train','development') for r in plan['groups'][entry[key]]}
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('network_forbidden')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if 'evaluation_only' in path.parts:raise PermissionError('evaluation_cache_forbidden_to_adapter')
            if '/api_based/' in str(path) or '/sol/' in str(path):raise PermissionError('API_asset_forbidden')
            if path.name.endswith('_labels.npz') and (stage!='train' or path.name not in allowed):
                raise PermissionError('supervision_forbidden')
    sys.addaudithook(audit)
