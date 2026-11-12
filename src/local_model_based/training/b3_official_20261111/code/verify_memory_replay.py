"""엄격 guard 아래 동결 memory의 전체 eval 예측을 다시 계산해 비교한다."""
import json
import numpy as np
from adapter_io import ROOT,load_role,anchor,sha
from adapter_memory import predict_memory

records=[]
for fold in (1,2,3):
    folder=ROOT/'adapters/memory'/f'fold{fold}';receipt=json.loads((folder/'eval_receipt.json').read_text());modelpath=folder/'memory.npz';assert sha(modelpath)==receipt['checkpoint_sha256']
    with np.load(modelpath) as saved:model={key:saved[key] for key in saved.files}
    batch,index,_=load_role(fold,'eval');values=predict_memory(batch,anchor(fold,'eval'),np.array([1/r['fps'] for r in index]),model)
    assert set(values)==set(receipt['outputs']);difference=0.
    for name,actual in values.items():
        path=folder/'eval'/(name+'.npy');assert sha(path)==receipt['outputs'][name];expected=np.load(path,mmap_mode='r');assert np.isfinite(actual).all() and actual.shape==expected.shape
        difference=max(difference,float(np.abs(actual-expected).max()));assert np.array_equal(actual,expected),(fold,name,difference)
    records.append({'fold':fold,'scenes':len(index),'methods':len(values),'maxdiff':difference});print(json.dumps(records[-1]),flush=True)
    del values,batch,model
(ROOT/'pilot/memory_strict_guard_replay.json').write_text(json.dumps({'passed':True,'records':records,'retrained':False,'GT_input':False},indent=2))
