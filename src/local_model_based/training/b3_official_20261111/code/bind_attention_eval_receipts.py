"""기존 eval 영수증을 보존하고 명시적 역할과 원본 SHA를 별도 문서로 묶는다."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adapter_io import ROOT,sha

plan=json.loads((ROOT/'runplan.json').read_text())
for fold in (1,2,3):
    folder=ROOT/'attention'/f'fold{fold}';original=folder/'eval_inference_receipt.json';value=json.loads(original.read_text())
    assert value['fold']==fold and value['runplan']==sha(ROOT/'runplan.json') and value['source']==sha(ROOT/'code/export_attention.py')
    index=[r for item in plan['groups'][plan['folds'][fold-1]['evaluation']] for r in json.loads((ROOT/'data/indices'/(Path(item['file']).stem+'_index.json')).read_text())]
    assert value['order_sha256']==hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest() and value['n']==len(index)
    for name,item in value['outputs'].items():
        path=folder/item['path'];assert '_eval_d' in path.name and sha(path)==item['sha256'];array=np.load(path,mmap_mode='r');assert array.shape==(len(index),12,2) and np.isfinite(array).all()
    bound={**value,'role':'eval','original_receipt_sha256':sha(original),'role_binding_source_sha256':sha(__file__),'amendment':'role field was implicit in the original eval-only exporter; original receipt and every prediction byte preserved'}
    out=folder/'eval_inference_receipt_role_bound.json'
    if out.exists():assert json.loads(out.read_text())==bound
    else:out.write_text(json.dumps(bound,indent=2))
    print(json.dumps({'fold':fold,'role':'eval','n':len(index),'original_sha256':sha(original),'bound_sha256':sha(out)}),flush=True)
