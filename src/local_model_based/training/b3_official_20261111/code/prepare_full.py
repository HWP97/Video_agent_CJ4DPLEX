"""실사 파일별 변환과 원자적 receipt. 최대4 CPU worker, 파일당 index 재사용."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import time
from prepare_pilot import ROOT, convert
from fast_features import tokenize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(item):
    name=Path(item['file']).stem
    signature={'source':item['sha256'],'runplan':sha(ROOT/'runplan.json'),
               'code':{n:sha(ROOT/'code'/n) for n in ['prepare_pilot.py','fast_features.py','features_reference.py','trajectory_baselines.py']}}
    receipt=ROOT/'data/receipts'/f'{name}.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text());assert saved['signature']==signature
        for path,digest in saved['outputs'].items():assert sha(ROOT/path)==digest
        return saved
    result=convert(name,item['scenes'],tokenize,'data/staging/'+name)
    assert result['samples']==item['scenes']
    outputs={}
    for suffix,folder in [('_inputs.npz','inputs'),('_labels.npz','targets'),('_index.json','indices')]:
        source=ROOT/'data/staging'/name/(name+suffix);target=ROOT/'data'/folder/(name+suffix)
        source.replace(target);outputs[str(target.relative_to(ROOT))]=sha(target)
    saved={'signature':signature,'outputs':outputs,'measurement':result}
    temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(saved,indent=2));temporary.replace(receipt)
    return saved


if __name__=='__main__':
    assert int(os.environ.get('OPENBLAS_NUM_THREADS','1'))==1
    for folder in ['inputs','targets','indices','receipts','staging']:(ROOT/'data'/folder).mkdir(parents=True,exist_ok=True)
    rows=[r for r in json.loads((ROOT/'provenance/full_release_audit.json').read_text())['files'] if r['file'].startswith('train/real_data/')]
    rows.sort(key=lambda r:r['scenes'])
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(task,rows):
            print(json.dumps({'completed':r['measurement']['dataset'],'seconds':time.monotonic()-start}),flush=True)
    receipts=[json.loads(p.read_text()) for p in sorted((ROOT/'data/receipts').glob('*.json'))]
    assert sum(r['measurement']['samples'] for r in receipts)==240890
    (ROOT/'data/preparation_complete.json').write_text(json.dumps({'scenes':240890,'files':len(receipts),'seconds':time.monotonic()-start,'workers':4,'API':0},indent=2))
