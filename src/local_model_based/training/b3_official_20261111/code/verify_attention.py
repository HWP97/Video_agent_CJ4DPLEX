"""36개 완료 fit의 원본서명·체크포인트·전수예측과 소수 재추론을 검증한다."""
import hashlib
import json
import numpy as np
import torch
from adapter_io import ROOT,sha,load_role,install_guard
from model import Config,Recurrent,predict

install_guard('infer',1);torch.set_num_threads(1);records=[];total_arrays=0
for fold in (1,2,3):
    directory=ROOT/'attention'/f'fold{fold}';batch,index,_=load_role(fold,'dev');ids=np.unique(np.linspace(0,len(index)-1,16,dtype=int));part=batch.take(ids);models=[]
    for path in sorted(directory.glob('*.json')):
        receipt=json.loads(path.read_text())
        if 'signature' not in receipt or 'checkpoint_sha256' not in receipt:continue
        models.append(path.stem);assert receipt['signature']['runplan']==sha(ROOT/'runplan.json') and not receipt['eval_GT_opened'] and receipt['API']==0
        for name,digest in receipt['signature']['source'].items():assert sha(ROOT/'code'/name)==digest
        for name,digest in receipt['signature']['data'].items():assert sha(ROOT/name)==digest
        checkpoint=path.with_suffix('.pt');assert sha(checkpoint)==receipt['checkpoint_sha256'];saved=torch.load(checkpoint,weights_only=False);net=Recurrent(Config(**saved['config'])).cuda();net.load_state_dict(saved['state_dict'])
        differences=[]
        for item in receipt['diagnostics']:
            depth=item['depth'];values=np.load(directory/f'{path.stem}_dev_d{depth}.npy',mmap_mode='r');assert values.shape==(len(index),12,2) and np.isfinite(values).all();fresh=predict(net,part,depth);difference=float(np.abs(fresh-values[ids]).max());assert difference<1e-4,(fold,path.stem,depth,difference);differences.append(difference);total_arrays+=1
        records.append({'fold':fold,'config':path.stem,'train_n':receipt['train_n'],'dev_n':receipt['dev_n'],'epochs':len(receipt['curve']),'best_epoch':receipt['best_epoch'],'fit_seconds':receipt['seconds'],'checkpoint_sha256':sha(checkpoint),'dev_replay16_maxdiff':max(differences)});del net
    assert len(models)==12
    receipt=json.loads((directory/'eval_inference_receipt.json').read_text());assert len(receipt['outputs'])==30
    for item in receipt['outputs'].values():
        path=directory/item['path'];assert sha(path)==item['sha256'];values=np.load(path,mmap_mode='r');assert values.shape==(receipt['n'],12,2) and np.isfinite(values).all();total_arrays+=1
    baseline=np.load(directory/'direct_unshared4_eval.npy',mmap_mode='r');exported=np.load(directory/'direct_unshared4_eval_d4.npy',mmap_mode='r');assert np.array_equal(baseline,exported)
result={'passed':True,'fits':len(records),'full_arrays_verified':total_arrays,'sample_replay_scenes_per_config':16,'max_replay_difference':max(r['dev_replay16_maxdiff'] for r in records),'baseline_anchor_vs_full_export_bitwise':True,'records':records,'API':0}
(ROOT/'attention/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
