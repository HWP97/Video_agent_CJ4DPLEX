"""기존 CV/KF 및 현재 attention 기준의 관측 전용 잔차 입력."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True,slots=True)
class ResidualInputs:
    own: np.ndarray
    social: np.ndarray
    old_ego: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    neighbor_future: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True,slots=True)
class Trial:
    name: str
    baseline: str
    social: bool = False
    collision: float = 0.
    old: bool = False


TRIALS = (
    Trial('cv_random_self', 'cv', old=True), Trial('cv_zero_self', 'cv'),
    Trial('kalman_self', 'kalman'), Trial('kalman_social', 'kalman', social=True),
    Trial('kalman_social_collision', 'kalman', social=True, collision=.1),
    Trial('attention_self', 'attention'), Trial('attention_social', 'attention', social=True),
    Trial('attention_social_collision', 'attention', social=True, collision=.1),
)


def observed_features(batch, baseline: np.ndarray, dt: np.ndarray) -> ResidualInputs:
    positions = batch.ego[:, :, :2]
    path = np.matmul(baseline - batch.origin[:, None], batch.rotation) / batch.scale[:, None, None]
    own = np.concatenate((positions.reshape(len(positions), -1), batch.ego[:, 1:, 2:4].reshape(len(positions), -1), path.reshape(len(positions), -1)), axis=1)
    relative = batch.neighbors[:, :, :2]
    distances = np.linalg.norm(relative, axis=-1) * batch.scale[:, None]
    weights = np.exp(-distances / 2) * batch.mask
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    summary = np.zeros((len(positions), 12))
    summary[:, :9] = np.sum(batch.neighbors * weights[:, :, None], axis=1)
    valid = batch.mask.any(1)
    summary[:, 9] = batch.mask.sum(1) / 8
    summary[valid, 10] = distances[valid, 0] / batch.scale[valid]
    projected = np.where(batch.mask.astype(bool), batch.neighbors[:, :, 6], np.inf)
    summary[valid, 11] = projected[valid].min(1)
    old_ego = np.matmul(positions, batch.rotation.transpose(0, 2, 1)) * batch.scale[:, None, None]
    velocity = (old_ego[:, -1] - old_ego[:, -3]) / (2 * dt[:, None])
    horizon = 12 * dt
    relative_velocity = np.matmul(batch.neighbors[:, :, 2:4], batch.rotation.transpose(0, 2, 1)) * batch.scale[:, None, None] / horizon[:, None, None]
    neighbor_velocity = relative_velocity + velocity[:, None]
    current = batch.origin[:, None] + np.matmul(relative, batch.rotation.transpose(0, 2, 1)) * batch.scale[:, None, None]
    future = current[:, :, None] + neighbor_velocity[:, :, None] * (dt[:, None] * np.arange(1, 13)[None])[:, None, :, None]
    future *= batch.mask[:, :, None, None]
    return ResidualInputs(own, np.concatenate((own, summary), axis=1), old_ego, batch.rotation, batch.scale, future, batch.mask)


class Residual(nn.Module):
    def __init__(self, center: torch.Tensor, spread: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer('center', center)
        self.register_buffer('spread', spread)
        self.hidden = nn.Sequential(nn.Linear(len(center), 48), nn.Tanh())
        self.output = nn.Linear(48, 24)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden((features - self.center) / self.spread)).reshape(-1, 12, 2)


def predict_residual(net,inputs: ResidualInputs,baseline: np.ndarray,trial: Trial,device: str) -> np.ndarray:
    net.eval();values=[]
    with torch.inference_mode():
        for first in range(0,len(baseline),256):
            last=first+256
            if trial.old:
                ego=torch.as_tensor(inputs.old_ego[first:last],device=device,dtype=torch.float32)
                shift=net(ego,torch.zeros(len(ego),8,9,2,device=device),torch.zeros(len(ego),8,device=device)).cpu().numpy().astype(np.float64)
            else:
                x=inputs.social if trial.social else inputs.own
                output=net(torch.as_tensor(x[first:last],device=device,dtype=torch.float32)).cpu().numpy().astype(np.float64)
                shift=np.matmul(output,inputs.rotation[first:last].transpose(0,2,1))*inputs.scale[first:last,None,None]
            values.append(baseline[first:last]+shift)
    result=np.concatenate(values)
    assert result.shape==(len(baseline),12,2) and np.isfinite(result).all()
    return result


def sources() -> dict[str,str]:
    from adapter_io import ROOT,sha
    files=[Path(__file__),ROOT/'code/adapter_reduced.py',ROOT/'code/adapter_memory.py',ROOT/'code/adapter_io.py',ROOT/'code/features.py',ROOT/'runplan.json',ROOT/'adapters/reduced_papers/models.py']
    files.extend(Path('/mnt/RG2/baseline_residual_20261110/b3')/name for name in ('features.py','residual_model.py','train_local.py','select_local.py'))
    return {str(path):sha(path) for path in files}


def create_model(trial: Trial,center: torch.Tensor,spread: torch.Tensor,device: str):
    from adapter_reduced import model_module
    if trial.old:
        module=model_module();net=module.Core(module.Config('re','no_resonance'))
    else:net=Residual(center,spread)
    return net.to(device)


def fit(trial: Trial,fold: int,train,dev,folder: Path,device: str,probe: bool) -> None:
    from adapter_io import sha
    from adapter_reduced import role_signature,save
    tb,tm,ty,ta=train;db,dm,dy,da=dev
    baseline=ta if trial.baseline=='attention' else getattr(tb,trial.baseline)
    devbase=da if trial.baseline=='attention' else getattr(db,trial.baseline)
    inputs=observed_features(tb,baseline,np.asarray([1/r['fps'] for r in tm]))
    development=observed_features(db,devbase,np.asarray([1/r['fps'] for r in dm]))
    signature={'sources':sources(),'trial':asdict(trial),'train':role_signature(fold,'train',tm),'dev':role_signature(fold,'dev',dm),'seed':20261110,'probe_only':probe}
    checkpoint=folder/(trial.name+'.pt');receipt_path=folder/(trial.name+'.receipt.json')
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());assert receipt['signature']==signature and sha(checkpoint)==receipt['checkpoint_sha256']
        assert sha(folder/'fit_dev'/(trial.name+'.npy'))==receipt['prediction_sha256']
        return
    assert not checkpoint.exists(),f'Unreceipted checkpoint: {checkpoint}'
    torch.manual_seed(20261110)
    if device=='cuda':torch.cuda.manual_seed_all(20261110);torch.cuda.reset_peak_memory_stats()
    tx=torch.as_tensor(inputs.social if trial.social else inputs.own,device=device,dtype=torch.float32)
    center=tx.mean(0);spread=tx.std(0,unbiased=False).clamp_min(1e-4)
    net=create_model(trial,center,spread,device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01)
    base=torch.as_tensor(baseline,device=device,dtype=torch.float32);truth=torch.as_tensor(ty,device=device,dtype=torch.float32)
    rotation=torch.as_tensor(inputs.rotation,device=device,dtype=torch.float32);scale=torch.as_tensor(inputs.scale,device=device,dtype=torch.float32)
    old_ego=torch.as_tensor(inputs.old_ego,device=device,dtype=torch.float32)
    neighbor=torch.as_tensor(inputs.neighbor_future,device=device,dtype=torch.float32) if trial.collision else None
    mask=torch.as_tensor(inputs.mask,device=device,dtype=torch.float32)
    initial=predict_residual(net,development,devbase,trial,device)
    if not trial.old:assert np.array_equal(initial,devbase)
    best=float('inf');stale=0;curve=[];started=time.monotonic()
    for epoch in range(1,2 if probe else 41):
        net.train();losses=[]
        for ids in torch.randperm(len(ty),device=device).split(256):
            if trial.old:
                shift=net(old_ego[ids],torch.zeros(len(ids),8,9,2,device=device),torch.zeros(len(ids),8,device=device))
            else:
                output=net(tx[ids]);shift=torch.bmm(output,rotation[ids].transpose(1,2))*scale[ids,None,None]
            prediction=base[ids]+shift
            loss=torch.linalg.vector_norm(prediction-truth[ids],dim=-1).mean()+.01*shift.square().mean()
            if trial.collision:
                distance=torch.linalg.vector_norm(prediction[:,None]-neighbor[ids],dim=-1)
                weights=mask[ids,:,None]
                penalty=((.4-distance).clamp_min(0).square()*weights).sum()/(weights.sum()*12).clamp_min(1)
                loss=loss+trial.collision*penalty
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        values=predict_residual(net,development,devbase,trial,device)
        errors=np.linalg.norm(values-dy,axis=-1);score=float(errors.mean())
        curve.append({'epoch':epoch,'loss':float(np.mean(losses)),'ADE':score,'FDE':float(errors[:,-1].mean())})
        if score<best-1e-12:
            best=score;stale=0
            torch.save({'state_dict':net.state_dict(),'trial':asdict(trial),'signature':signature,'center':center,'spread':spread},checkpoint)
        else:stale+=1
        print(json.dumps({'method':trial.name,'epoch':epoch,'ADE':score,'seconds':time.monotonic()-started}),flush=True)
        if stale>=8:break
    saved=torch.load(checkpoint,map_location=device,weights_only=False);net.load_state_dict(saved['state_dict'])
    path=folder/'fit_dev'/(trial.name+'.npy');np.save(path,predict_residual(net,development,devbase,trial,device))
    save(receipt_path,{'signature':signature,'checkpoint_sha256':sha(checkpoint),'prediction_sha256':sha(path),'best_dev_ADE':best,'curve':curve,
                      'initial_baseline_max_difference':float(np.max(np.abs(initial-devbase))),'seconds':time.monotonic()-started,
                      'peak_vram_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'API_calls':0,'probe_only':probe})


def main() -> None:
    from adapter_io import ROOT,anchor,load_role,install_guard,sha
    from adapter_reduced import role_signature,save
    action,fold_text=sys.argv[1:3];fold=int(fold_text)
    assert action in ('train','probe','predict-dev','predict-eval') and fold in (1,2,3)
    probe=action=='probe';device='cpu' if probe else 'cuda'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    install_guard('train' if action in ('train','probe') else 'predict',fold)
    root=ROOT/'adapters/residual';folder=(root/'pilot' if probe else root)/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
    if action in ('train','probe'):
        packed={}
        for role in ('train','dev'):
            batch,metadata,labels=load_role(fold,role,targets=True);local=anchor(fold,role)
            if probe:
                ids=np.arange(min(64,len(metadata)));batch=batch.take(ids);labels=labels[ids];local=local[ids];metadata=[metadata[i] for i in ids]
            packed[role]=(batch,metadata,labels,local)
        (folder/'fit_dev').mkdir(exist_ok=True)
        for trial in TRIALS:fit(trial,fold,packed['train'],packed['dev'],folder,device,probe)
        save(folder/'training_receipt.json',{'fits':len(TRIALS),'sources':sources(),'probe_only':probe,'API_calls':0,'selection_pending_controller':True})
    else:
        trained=json.loads((folder/'training_receipt.json').read_text());assert trained['sources']==sources() and not trained['probe_only']
        role=action.removeprefix('predict-');batch,metadata,labels=load_role(fold,role,targets=False);assert labels is None
        completed=folder/(role+'_receipt.json')
        if completed.exists():
            receipt=json.loads(completed.read_text());assert receipt['role']==role_signature(fold,role,metadata)
            assert all(sha(folder/role/(name+'.npy'))==value for name,value in receipt['outputs'].items())
            for checkpoint in folder.glob('*.pt'):assert sha(checkpoint)==json.loads(checkpoint.with_suffix('.receipt.json').read_text())['checkpoint_sha256']
            print(json.dumps({'status':'verified_existing','role':role}));return
        local=anchor(fold,role);dt=np.asarray([1/r['fps'] for r in metadata]);destination=folder/role;destination.mkdir(exist_ok=True)
        values={'kalman_zero':batch.kalman,'attention_zero':local,'constant_velocity':batch.cv};started=time.monotonic()
        for trial in TRIALS:
            checkpoint=folder/(trial.name+'.pt');receipt=json.loads(checkpoint.with_suffix('.receipt.json').read_text());assert sha(checkpoint)==receipt['checkpoint_sha256']
            saved=torch.load(checkpoint,map_location=device,weights_only=False);assert saved['signature']==receipt['signature']
            net=create_model(trial,saved['center'],saved['spread'],device);net.load_state_dict(saved['state_dict'])
            base=local if trial.baseline=='attention' else getattr(batch,trial.baseline)
            values[trial.name]=predict_residual(net,observed_features(batch,base,dt),base,trial,device)
        outputs={}
        for name,prediction in values.items():
            path=destination/(name+'.npy');np.save(path,prediction);outputs[name]=sha(path)
        save(completed,{'role':role_signature(fold,role,metadata),'outputs':outputs,'seconds':time.monotonic()-started,'future_targets_loaded':False,
                        'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'API_calls':0})


if __name__=='__main__':main()
