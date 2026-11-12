"""동결 attention의 공통 anchor를 supervision 없이 한 번 저장한다."""
import json
import sys
import time
import numpy as np
import torch
from adapter_io import ROOT,load_role,sha
from model import Config,Recurrent,predict


if __name__=='__main__':
    fold=int(sys.argv[1]);roles=sys.argv[2:] or ['train','dev','eval'];torch.set_num_threads(1)
    path=ROOT/'attention'/f'fold{fold}'/'direct_unshared4.pt'
    model_receipt=json.loads(path.with_suffix('.json').read_text());assert sha(path)==model_receipt['checkpoint_sha256']
    saved=torch.load(path,weights_only=False);net=Recurrent(Config(**saved['config'])).cuda();net.load_state_dict(saved['state_dict'])
    def guard(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('network_forbidden')
        if event=='open' and isinstance(args[0],str) and args[0].endswith('_labels.npz'):raise PermissionError('GT_forbidden')
    sys.addaudithook(guard)
    for role in roles:
        batch,index,_=load_role(fold,role);started=time.monotonic();values=predict(net,batch,4)
        out=path.parent/f'direct_unshared4_{role}.npy';np.save(out,values)
        receipt={'fold':fold,'role':role,'scenes':len(index),'sha256':sha(out),'checkpoint_sha256':sha(path),'runplan_sha256':sha(ROOT/'runplan.json'),'seconds':time.monotonic()-started,'GT_read':False,'sample_ids':[r['sample_id'] for r in index]}
        out.with_suffix('.json').write_text(json.dumps(receipt));print(json.dumps({k:v for k,v in receipt.items() if k!='sample_ids'}),flush=True)
