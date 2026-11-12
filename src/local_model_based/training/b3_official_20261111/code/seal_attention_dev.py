"""동결 체크포인트로 dev 전수를 재현하여 최초 출력 SHA 누락을 보완한다."""
import hashlib
import json
import time
import numpy as np
import torch
from adapter_io import ROOT,load_role,sha
from model import Config,Recurrent,predict

torch.set_num_threads(1)
for fold in (1,2,3):
    batch,index,_=load_role(fold,'dev');directory=ROOT/'attention'/f'fold{fold}';outputs={};models={};started=time.monotonic()
    for path in sorted(directory.glob('*.json')):
        receipt=json.loads(path.read_text())
        if 'signature' not in receipt or 'checkpoint_sha256' not in receipt:continue
        assert receipt['signature']['runplan']==sha(ROOT/'runplan.json')
        for name,digest in receipt['signature']['source'].items():assert sha(ROOT/'code'/name)==digest
        for name,digest in receipt['signature']['data'].items():assert sha(ROOT/name)==digest
        checkpoint=path.with_suffix('.pt');assert sha(checkpoint)==receipt['checkpoint_sha256'];saved=torch.load(checkpoint,weights_only=False);net=Recurrent(Config(**saved['config'])).cuda();net.load_state_dict(saved['state_dict'])
        for item in receipt['diagnostics']:
            depth=item['depth'];prediction=directory/f'{path.stem}_dev_d{depth}.npy';old=np.load(prediction,mmap_mode='r');fresh=predict(net,batch,depth)
            assert fresh.shape==(len(index),12,2) and np.isfinite(fresh).all() and np.array_equal(fresh,old),(fold,path.stem,depth,float(np.abs(fresh-old).max()))
            outputs[f'{path.stem}_d{depth}']={'path':prediction.name,'sha256':sha(prediction),'fresh_values_sha256':hashlib.sha256(fresh.tobytes()).hexdigest(),'replay_maxdiff':0.}
        models[path.stem]={'checkpoint':sha(checkpoint),'receipt':sha(path)};del net
        print(json.dumps({'fold':fold,'config':path.stem,'n':len(index),'verified_outputs':len(outputs),'seconds':time.monotonic()-started}),flush=True)
    assert len(models)==12 and len(outputs)==30
    result={'fold':fold,'role':'dev','n':len(index),'order_sha256':hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'outputs':outputs,'models':models,'source':sha(__file__),'guard_source':sha(ROOT/'code/run_guarded.py'),'runplan':sha(ROOT/'runplan.json'),'GT_read':False,'seconds':time.monotonic()-started,'weights_retrained':False,'reason':'original producer receipt omitted prediction SHA; full checkpoint replay proves every saved value before sealing'}
    (directory/'dev_inference_receipt.json').write_text(json.dumps(result,indent=2))
