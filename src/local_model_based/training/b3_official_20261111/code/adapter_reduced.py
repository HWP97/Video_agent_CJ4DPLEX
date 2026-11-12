"""기존 축소 Re/SocialMOIF 설정과 동결 attention anchor 전이 실행기."""
from __future__ import annotations
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch

from adapter_io import ROOT, anchor, install_guard, load_role, sha
from adapter_memory import role_metadata
from observed_io import load_observed_role

SEED=20261109
FOLDER=ROOT/'adapters/reduced_papers'


def model_module():
    spec=importlib.util.spec_from_file_location('reduced_physical_models',FOLDER/'models.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def save(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temporary.replace(path)


def role_signature(fold: int,role: str,metadata: list[dict]) -> dict[str,int|str|dict[str,str]]:
    plan=json.loads((ROOT/'runplan.json').read_text())
    group=plan['folds'][fold-1][{'train':'train','dev':'development','eval':'evaluation'}[role]]
    paths=[ROOT/'data/receipts'/(Path(row['file']).stem+'.json') for row in plan['groups'][group]]
    paths.extend(ROOT/'auxiliary_receipts'/(Path(row['file']).stem+'.json') for row in plan['groups'][group])
    paths.extend([ROOT/'attention'/f'fold{fold}'/f'direct_unshared4_{role}.json'])
    return dict(role_metadata(metadata),role=role,receipts={str(path):sha(path) for path in paths})


def sources() -> dict[str,str]:
    paths=[Path(__file__),FOLDER/'models.py',ROOT/'code/adapter_io.py',ROOT/'code/observed_io.py',ROOT/'code/features.py',ROOT/'code/adapter_memory.py',ROOT/'runplan.json',
           Path('/mnt/RG2/paper_experiments_20261109/b3/models.py'),Path('/mnt/RG2/paper_experiments_20261109/b3/experiment.py')]
    return {str(path):sha(path) for path in paths}


def infer(net,arrays: tuple[np.ndarray,...],baseline: np.ndarray,device: str) -> np.ndarray:
    net.eval()
    values=[]
    with torch.inference_mode():
        for start in range(0,len(baseline),128):
            inputs=[torch.as_tensor(value[start:start+128],device=device,dtype=torch.float32) for value in arrays]
            values.append(net(*inputs).cpu().numpy())
    result=baseline+np.concatenate(values)
    assert result.shape==(len(baseline),12,2) and np.isfinite(result).all()
    return result


def fit(config,kind: str,fold: int,probe: bool,train,dev,folder: Path,device: str) -> str:
    module=model_module()
    name=('attention_' if kind=='attention' else '')+config.family+'_'+config.variant
    checkpoint=folder/(name+'.pt');receipt_path=folder/(name+'.receipt.json')
    tx,tbase,ty,tmeta=train;dx,dbase,dy,dmeta=dev
    signature={'sources':sources(),'config':asdict(config),'kind':kind,'seed':SEED,'probe':probe,'train':role_signature(fold,'train',tmeta),'dev':role_signature(fold,'dev',dmeta)}
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        assert receipt['signature']==signature and sha(checkpoint)==receipt['checkpoint_sha256']
        assert sha(folder/'fit_dev'/(name+'.npy'))==receipt['prediction_sha256']
        return name
    assert not checkpoint.exists(),f'Unreceipted checkpoint: {checkpoint}'
    torch.manual_seed(SEED)
    if device=='cuda':
        torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats()
    net=module.Core(config).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    tensor_inputs=tuple(torch.as_tensor(x,device=device,dtype=torch.float32) for x in tx)
    base=torch.as_tensor(tbase,device=device,dtype=torch.float32)
    truth=torch.as_tensor(ty,device=device,dtype=torch.float32)
    best=float('inf');stale=0;curve=[];started=time.monotonic()
    for epoch in range(1,2 if probe else 31):
        net.train();losses=[]
        for ids in torch.randperm(len(ty),device=device).split(128):
            prediction=base[ids]+net(*(x[ids] for x in tensor_inputs))
            loss=torch.linalg.vector_norm(prediction-truth[ids],dim=-1).mean()
            if config.family=='moif' and config.direction:
                pv=torch.diff(prediction,dim=1,prepend=torch.zeros_like(prediction[:,:1]))
                tv=torch.diff(truth[ids],dim=1,prepend=torch.zeros_like(prediction[:,:1]))
                valid=(pv.norm(dim=-1)>1e-4)&(tv.norm(dim=-1)>1e-4)
                cosine=(pv*tv).sum(-1)/(pv.norm(dim=-1)*tv.norm(dim=-1)).clamp_min(1e-6)
                angle=torch.acos(cosine.clamp(-.9999,.9999))
                loss=loss+config.direction*(angle*valid).sum()/valid.sum().clamp_min(1)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        prediction=infer(net,dx,dbase,device)
        error=np.linalg.norm(prediction-dy,axis=-1)
        score=float(error.mean())
        curve.append({'epoch':epoch,'loss':float(np.mean(losses)),'ADE':score,'FDE':float(error[:,-1].mean())})
        if score<best:
            best=score;stale=0
            torch.save({'state_dict':net.state_dict(),'config':asdict(config),'kind':kind,'signature':signature},checkpoint)
        else:stale+=1
        print(json.dumps({'method':name,'epoch':epoch,'dev_ADE':score,'seconds':time.monotonic()-started}),flush=True)
        if stale>=6:break
    saved=torch.load(checkpoint,map_location=device,weights_only=False);net.load_state_dict(saved['state_dict'])
    prediction=infer(net,dx,dbase,device)
    path=folder/'fit_dev'/(name+'.npy');np.save(path,prediction)
    receipt={'signature':signature,'checkpoint_sha256':sha(checkpoint),'prediction_sha256':sha(path),'curve':curve,'best_dev_ADE':best,
             'seconds':time.monotonic()-started,'peak_vram_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
             'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'API_calls':0,'device':device,'probe_only':probe}
    save(receipt_path,receipt)
    return name


def main() -> None:
    action,fold_text=sys.argv[1:3];fold=int(fold_text)
    assert action in ('train','probe','predict-dev','predict-eval') and fold in (1,2,3)
    probe=action=='probe';device='cpu' if probe else 'cuda'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    install_guard('train' if action in ('train','probe') else 'predict',fold)
    folder=(FOLDER/'pilot' if probe else FOLDER)/f'fold{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    module=model_module()
    if action in ('train','probe'):
        packed={}
        for role in ('train','dev'):
            batch,metadata,truth=load_role(fold,role,targets=True)
            observed=load_observed_role(fold,role)
            local=anchor(fold,role)
            if probe:
                ids=np.arange(min(64,len(metadata)));batch=batch.take(ids);truth=truth[ids];local=local[ids];observed=tuple(x[ids] for x in observed);metadata=[metadata[i] for i in ids]
            arrays=(*observed,np.asarray([1/r['fps'] for r in metadata],dtype=np.float32))
            packed[role]=(arrays,{'cv':batch.cv-batch.origin[:,None],'attention':local-batch.origin[:,None]},truth-batch.origin[:,None],metadata,batch.origin)
        (folder/'fit_dev').mkdir(exist_ok=True)
        names=[]
        for kind in ('cv','attention'):
            training=(packed['train'][0],packed['train'][1][kind],packed['train'][2],packed['train'][3])
            development=(packed['dev'][0],packed['dev'][1][kind],packed['dev'][2],packed['dev'][3])
            for config in module.CONFIGS:names.append(fit(config,kind,fold,probe,training,development,folder,device))
        selection={}
        for kind in ('cv','attention'):
            prefix='attention_' if kind=='attention' else ''
            for family in ('re','moif'):
                candidates=[prefix+c.family+'_'+c.variant for c in module.CONFIGS if c.family==family]
                selection[prefix+family+'_selected']=min(candidates,key=lambda n:json.loads((folder/(n+'.receipt.json')).read_text())['best_dev_ADE'])
            selection[prefix+'self_control']=prefix+'re_no_resonance'
        save(folder/'selection.json',{'aliases':selection,'rule':'원 grouped-dev ADE 최소, config 순서 tie-break','sources':sources(),'probe_only':probe})
        save(folder/'training_receipt.json',{'fits':len(names),'sources':sources(),'probe_only':probe,'API_calls':0})
    else:
        role=action.removeprefix('predict-');selection=json.loads((folder/'selection.json').read_text());assert selection['sources']==sources() and not selection['probe_only']
        batch,metadata,truth=load_role(fold,role,targets=False);assert truth is None
        completed=folder/(role+'_receipt.json')
        if completed.exists():
            receipt=json.loads(completed.read_text())
            assert receipt['role']==role_signature(fold,role,metadata)
            assert all(sha(folder/role/(name+'.npy'))==value for name,value in receipt['outputs'].items())
            for checkpoint in folder.glob('*.pt'):
                assert sha(checkpoint)==json.loads(checkpoint.with_suffix('.receipt.json').read_text())['checkpoint_sha256']
            print(json.dumps({'status':'verified_existing','role':role,'outputs':len(receipt['outputs'])}))
            return
        arrays=(*load_observed_role(fold,role),np.asarray([1/r['fps'] for r in metadata],dtype=np.float32));local=anchor(fold,role)
        destination=folder/role;destination.mkdir(exist_ok=True);outputs={};started=time.monotonic()
        for kind in ('cv','attention'):
            for config in module.CONFIGS:
                name=('attention_' if kind=='attention' else '')+config.family+'_'+config.variant
                receipt=json.loads((folder/(name+'.receipt.json')).read_text());checkpoint=folder/(name+'.pt');assert sha(checkpoint)==receipt['checkpoint_sha256']
                saved=torch.load(checkpoint,map_location=device,weights_only=False);assert saved['signature']==receipt['signature']
                net=module.Core(config).to(device);net.load_state_dict(saved['state_dict'])
                prediction=infer(net,arrays,local if kind=='attention' else batch.cv,device)
                path=destination/(name+'.npy');np.save(path,prediction);outputs[name]=sha(path)
        for alias,target in selection['aliases'].items():
            path=destination/(alias+'.npy');np.save(path,np.load(destination/(target+'.npy')));outputs[alias]=sha(path)
        save(folder/(role+'_receipt.json'),{'role':role_signature(fold,role,metadata),'outputs':outputs,'seconds':time.monotonic()-started,
             'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'API_calls':0,'future_targets_loaded':False})


if __name__=='__main__':main()
