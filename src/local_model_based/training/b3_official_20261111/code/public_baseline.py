"""all-train refit 없이 세 동결 attention 모델의 public 예측을 평균한다."""
import json
import numpy as np
import torch
from adapter_io import ROOT,sha,install_guard
from features import load
from model import Config,Recurrent,predict
from public_submission import write_submission

install_guard('infer',1);torch.set_num_threads(1);directory=ROOT/'public';receipt=json.loads((directory/'preparation_receipt.json').read_text())
for name,digest in receipt['outputs'].items():assert sha(directory/name)==digest
batch=load(directory/'inputs.npz');out=directory/'predictions';out.mkdir(exist_ok=True);folds=[];models={}
for fold in (1,2,3):
    path=ROOT/'attention'/f'fold{fold}'/'direct_unshared4.pt';r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['checkpoint_sha256']
    saved=torch.load(path,weights_only=False);net=Recurrent(Config(**saved['config'])).cuda();net.load_state_dict(saved['state_dict']);values=predict(net,batch,4)
    np.save(out/f'baseline_fold{fold}.npy',values);folds.append(values);models[str(fold)]=sha(path);del net
values={'baseline_ensemble':np.mean(folds,axis=0),'hold':np.repeat(batch.origin[:,None],12,axis=1),'constant_velocity':batch.cv,'kalman':batch.kalman};submissions=[]
for name,prediction in values.items():
    assert prediction.shape==(3146,12,2) and np.isfinite(prediction).all();np.save(out/(name+'.npy'),prediction);submissions.append(write_submission(name,prediction))
result={'samples':3146,'models':models,'source':sha(__file__),'input':sha(directory/'inputs.npz'),'outputs':{p.name:sha(p) for p in out.glob('*.npy')},'submissions':submissions,'new_fit':False,'ensemble_rule':'arithmetic mean of three frozen fold models, fixed before hidden GT scoring','hidden_GT_score':None,'API':0}
(directory/'baseline_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
