"""공유 공개 Chronos를 오프라인으로 재사용하고 동일입력 비교군을 공식 train으로 학습한다."""
import sys
import os
import time
import numpy as np
import torch
from common import ROOT,read,save,digest,guard
from learning import metric,labels,npz

ASSET=__import__('pathlib').Path('/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/assets/chronos2')

def summaries(x):
    t=np.arange(x.shape[1])/10;t-=t.mean()
    return np.concatenate([x[:,-1],x.mean(1),x.std(1),(x*t[None,:,None]).sum(1)/(t@t)],1)

def forecast():
    sys.path.insert(0,str(ASSET));from adapter import load_pipeline,forecast as predict
    assert digest(ASSET/'model/model.safetensors')=='ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'
    a=np.load(ROOT/'dense_inputs.npz');pipeline=load_pipeline();times=[]
    directory=ROOT/'predictions';directory.mkdir(exist_ok=True)
    for c in read(ROOT/'adapter_runplan.json')['chronos_configs']:
        name=c['method'];path=directory/(name+'.npz');receipt=directory/(name+'.json')
        if path.exists() and receipt.exists():
            result=read(receipt);assert digest(path)==result['sha256'];times.append(result);continue
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        raw=predict(pipeline,a['short' if c['length_s']==1 else 'long'],30,c['joint'])[:,[4,9,19,29]]
        torch.cuda.synchronize();npz(path,raw=raw,prediction=np.clip(raw,-1,1))
        result={'method':name,'samples':len(raw),'seconds':time.perf_counter()-started,'peak_VRAM_bytes':torch.cuda.max_memory_allocated(),'sha256':digest(path),'GT_access':False,
                'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'precision':'float32 with TF32 disabled'}
        save(receipt,result);times.append(result);print(result,flush=True)
    save(ROOT/'chronos_inference.json',times)

def controls():
    rows=read(ROOT/'manifest.json');splits=np.array([r['split'] for r in rows]);tr=splits=='train';dev=splits=='dev'
    a=np.load(ROOT/'dense_inputs.npz');y=labels('train',rows);dy=labels('dev',rows);results=[]
    for key in ['short','long']:
        started=time.perf_counter()
        x=a[key];features=summaries(x)
        mean=features[tr].mean(0);scale=np.maximum(features[tr].std(0),1e-6);z=(features[tr]-mean)/scale;center=y.reshape(-1,8).mean(0)
        coef=np.linalg.solve(z.T@z/len(z)+np.eye(z.shape[1]),z.T@(y.reshape(-1,8)-center)/len(z))
        npz(ROOT/'trained_models'/f'summary_ridge_{key}.npz',mean=mean,scale=scale,center=center,coef=coef)
        ridge=(((features-mean)/scale)@coef+center).reshape(-1,4,2)
        # Preserve the original dense baseline's centering, including its historical time origin.
        t=np.arange(x.shape[1])/10;t-=t.mean()
        linear=x.mean(1)[:,None,:]+features[:,-2:][:,None,:]*(np.array([.5,1,2,3])-t[0])[None,:,None]
        for name,pred in [(f'summary_ridge_{key}',ridge),(f'linear_{key}',linear)]:
            npz(ROOT/'predictions'/(name+'.npz'),raw=pred,prediction=np.clip(pred,-1,1))
            results.append({'method':name,'dev':metric(dy,pred[dev]),'fit_samples':int(tr.sum()) if 'ridge' in name else 0,'shared_key_fit_and_predict_seconds':time.perf_counter()-started})
    for c in read(ROOT/'adapter_runplan.json')['chronos_configs']:
        p=np.load(ROOT/'predictions'/(c['method']+'.npz'))['prediction']
        results.append({'method':c['method'],'dev':metric(dy,p[dev]),'fit_samples':0})
    primary=read(ROOT/'primary_frozen.json');all_results=primary['methods']+results
    baseline=next(r['dev'] for r in all_results if r['method']=='hold')
    eligible=[r for r in all_results if r['dev']['MAE']<baseline['MAE'] and r['dev']['CCC_mean8']>=baseline['CCC_mean8']]
    selected=min(eligible,key=lambda r:r['dev']['MAE'])['method'] if eligible else 'hold'
    save(ROOT/'all_frozen.json',{'selected':selected,'primary_selected':primary['selected'],'dev_results':all_results,'eval_GT_opened':False,
          'selection':'dev MAE decrease AND CCC nondecrease against hold; lowest eligible MAE',
          'hashes':{str(p.relative_to(ROOT)):digest(p) for p in list((ROOT/'trained_models').glob('*'))+list((ROOT/'predictions').glob('*.npz'))}})

if __name__=='__main__':
    stage=sys.argv[1];torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    guard('infer' if stage=='forecast' else 'train')
    {'forecast':forecast,'controls':controls}[stage]()
