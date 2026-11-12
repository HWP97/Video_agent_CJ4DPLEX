"""dev에서 정한 반복 모델의 이웃 제거 진단만 관측 입력으로 실행한다."""
import hashlib
import json
import sys
import time
import numpy as np
import torch
from adapter_io import ROOT,load_role,sha,install_guard
from model import Config,Recurrent,predict

fold=int(sys.argv[1]);install_guard('infer',fold);torch.set_num_threads(1)
selection=ROOT/'selections'/f'fold{fold}.json';chosen=json.loads(selection.read_text())['aliases']['loop/idea_diagnostic'];name,depth=chosen.split('/',1)[1].rsplit('_d',1)
path=ROOT/'attention'/f'fold{fold}'/(name+'.pt');r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['checkpoint_sha256']
saved=torch.load(path,weights_only=False);net=Recurrent(Config(**saved['config'])).cuda();net.load_state_dict(saved['state_dict']);directory=ROOT/'selections'/f'fold{fold}';directory.mkdir(exist_ok=True)
for role in ('dev','eval'):
    batch,index,_=load_role(fold,role);started=time.monotonic();values=predict(net,batch,int(depth),no_neighbors=True)
    assert values.shape==(len(index),12,2) and np.isfinite(values).all();out=directory/(role+'_no_neighbors.npy');np.save(out,values)
    result={'n':len(index),'role':role,'order_sha256':hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'sha256':sha(out),'checkpoint':sha(path),'selection':sha(selection),'source':sha(__file__),'seconds':time.monotonic()-started,'GT_read':False}
    (directory/(role+'_diagnostic.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
