# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: 기존 b1 model/env/bin/python -B infer.py
"""선택 동결 뒤 정답 파일 접근 없이 네 split 전체를 예측한다."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
from prepare import OUT, SOURCE
from followup_common import sha,save,guard
from head import features


def main() -> None:
    guard(OUT,True)
    def forbid_labels(event: str,args: tuple) -> None:
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            assert not Path(os.fsdecode(args[0])).name.startswith('labels_'), 'inference_GT_forbidden'
    sys.addaudithook(forbid_labels)
    blocked = False
    try:
        with (SOURCE/'labels_eval.json').open('r'):
            pass
    except (AssertionError,PermissionError) as error:
        assert str(error)=='inference_GT_forbidden' or error.args[0]=='outer_target_forbidden'
        blocked = True
    assert blocked
    torch.set_num_threads(2)
    lock=json.loads((OUT/'selection_lock.json').read_text())
    receipt=json.loads((OUT/'training_receipt.json').read_text())
    assert all(sha(OUT/p)==h for p,h in receipt['outputs'].items())
    ids=np.array(json.loads((OUT/'split.json').read_text())['train_full'])
    x,hold=features(ids,OUT/'full')
    affine=np.column_stack((hold,np.ones(len(hold))))
    values={}
    for method in lock['models'][:3]:
        design=affine if method=='B2-D01' else x
        with np.load(OUT/'full'/f'{method}.npz') as data:
            current=hold+design@data['coefficient']
        values[method]=current
        np.savez(OUT/f'{method}_prediction.npz',current=current,prediction=np.repeat(current[:,None],4,1))
    if lock['D04_parent'] is not None:
        with np.load(OUT/'full/B2-D04.npz') as data:
            delta=(x@data['coefficient']).reshape(-1,4,2)
        current=values[lock['D04_parent']]
        np.savez(OUT/'B2-D04_prediction.npz',current=current,prediction=current[:,None]+delta)
    rows=json.loads((SOURCE/'manifest.json').read_text())
    save(OUT/'inference_receipt.json',{'selection_lock_sha256':sha(OUT/'selection_lock.json'),'training_receipt_sha256':sha(OUT/'training_receipt.json'),
        'sample_ids':[r['sample_id'] for r in rows],'samples':len(rows),'missing':0,'failed':0,'GT_read':False,'negative_GT_open_blocked':blocked,'source_sha256':sha(Path(__file__)),
        'outputs':{p.name:sha(p) for p in OUT.glob('*_prediction.npz')}})
    print(json.dumps({'samples':len(rows),'models':lock['models'],'GT_read':False}),flush=True)


if __name__=='__main__':
    main()
