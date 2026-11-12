"""이미 dev로 동결한 attention 체크포인트의 모든 지정 깊이를 GT 없이 추론한다."""
import hashlib
import json
import sys
import time
import numpy as np
import torch
from adapter_io import ROOT, load_role, sha, install_guard
from model import Config, Recurrent, predict


def main() -> None:
    fold=int(sys.argv[1]);install_guard('infer',fold);torch.set_num_threads(1)
    batch,index,_=load_role(fold,'eval');directory=ROOT/'attention'/f'fold{fold}';outputs={};models={};started=time.monotonic()
    for receipt in sorted(directory.glob('*.json')):
        data=json.loads(receipt.read_text())
        if 'checkpoint_sha256' not in data or 'signature' not in data:continue
        checkpoint=receipt.with_suffix('.pt');assert sha(checkpoint)==data['checkpoint_sha256']
        saved=torch.load(checkpoint,weights_only=False);net=Recurrent(Config(**saved['config'])).cuda();net.load_state_dict(saved['state_dict'])
        for diagnostic in data['diagnostics']:
            depth=diagnostic['depth'];name=f'{receipt.stem}_d{depth}';out=directory/f'{receipt.stem}_eval_d{depth}.npy'
            values=predict(net,batch,depth);assert values.shape==(len(index),12,2) and np.isfinite(values).all();np.save(out,values);outputs[name]={'path':out.name,'sha256':sha(out)}
        models[receipt.stem]={'checkpoint':sha(checkpoint),'receipt':sha(receipt)}
        print(json.dumps({'fold':fold,'config':receipt.stem,'outputs':len(outputs),'elapsed':time.monotonic()-started}),flush=True)
        del net
    assert len(models)==12 and len(outputs)==30,(len(models),len(outputs))
    result={'fold':fold,'n':len(index),'order_sha256':hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'outputs':outputs,'models':models,'source':sha(__file__),'runplan':sha(ROOT/'runplan.json'),'GT_read':False,'seconds':time.monotonic()-started}
    (directory/'eval_inference_receipt.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
