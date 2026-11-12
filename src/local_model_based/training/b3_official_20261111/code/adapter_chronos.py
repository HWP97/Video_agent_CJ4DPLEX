"""기존 Chronos2 가중치만 사용하여 각 실사 그룹을 한 번 예측한다."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from adapter_io import ROOT,load_role,sha,install_guard

ASSET=Path('/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/assets/chronos2')
sys.path.insert(0,str(ASSET))
from adapter import load_pipeline,forecast


if __name__=='__main__':
    install_guard('infer',1);torch.set_num_threads(1);pipeline=load_pipeline('cuda:0')
    plan=json.loads((ROOT/'runplan.json').read_text())
    for fold in (2,1,3):
        group=plan['folds'][fold-1]['evaluation'];batch,index,_=load_role(fold,'eval');context=batch.ego[:,:,:2]
        directory=ROOT/'adapters/chronos/groups'/group;directory.mkdir(parents=True,exist_ok=True)
        signature={'source':sha(__file__),'adapter':sha(ASSET/'adapter.py'),'manifest':sha(ASSET/'manifest.json'),'runplan':sha(ROOT/'runplan.json'),'n':len(index)}
        for joint in (True,False):
            name='chronos_joint' if joint else 'chronos_independent';receipt=directory/(name+'.json');out=directory/(name+'.npy')
            if receipt.exists():
                saved=json.loads(receipt.read_text());assert saved['signature']==signature and sha(out)==saved['sha256'];continue
            started=time.monotonic();values=np.empty((len(index),12,2),np.float64)
            for start in range(0,len(index),4096):
                end=min(start+4096,len(index));offset=forecast(pipeline,context[start:end],12,joint,64)
                values[start:end]=np.matmul(offset,batch.rotation[start:end].transpose(0,2,1))*batch.scale[start:end,None,None]+batch.origin[start:end,None]
                print(json.dumps({'group':group,'method':name,'processed':end,'total':len(index),'seconds':time.monotonic()-started}),flush=True)
            assert np.isfinite(values).all();np.save(out,values)
            receipt.write_text(json.dumps({'signature':signature,'sha256':sha(out),'seconds':time.monotonic()-started,'GPU_physical':os.environ['CUDA_VISIBLE_DEVICES'],'peak_vram':torch.cuda.max_memory_allocated(),'GT_read':False,'API':0,'condition':'pretrained zero-shot; 9 observed discrete steps, physical normalization per-scene fps; no fit'},indent=2))
