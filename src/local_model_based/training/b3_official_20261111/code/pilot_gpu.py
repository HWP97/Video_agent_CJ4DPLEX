"""GPU1 제한 pilot. 기존 결과 재현과 3epoch 실측이며 전수 결과가 아니다."""
from pathlib import Path
import dataclasses
import hashlib
import json
import os
import resource
import sys
import time
import numpy as np
import torch
from data import Observation, Track
from features import load, tokenize
from model import Config, Recurrent, predict, tensors
from trajectory_baselines import forecast

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/RG2/HWP_CJ4DPLEX_AI/src/local_model_based/data/b3')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    def guard(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('network_forbidden')
    sys.addaudithook(guard)
    torch.set_num_threads(2);torch.manual_seed(20261110);torch.cuda.manual_seed_all(20261110)
    cfg=Config('direct_unshared4',False,4,False)
    old=load(OLD/'validation.npz');index=json.loads((OLD/'index.json').read_text())
    reproduced=np.empty((len(index),12,2))
    for group in ('hotel','students','zara'):
        ids=[i for i,r in enumerate(index) if r['group']==group]
        net=Recurrent(cfg).cuda();saved=torch.load(OLD/f'models/{group}_direct_unshared4.pt',weights_only=False)
        net.load_state_dict(saved['state_dict']);reproduced[ids]=predict(net,old.take(ids),4)
    expected=np.load('/mnt/RG2/HWP_CJ4DPLEX_AI/results/local_model_based/b3/original_predictions.npz')['direct_unshared4_d4']
    difference=float(np.max(np.abs(expected-reproduced)));assert difference<=1e-9,difference
    np.save(ROOT/'pilot/old_baseline_reproduced.npy',reproduced)
    token_diff=0.
    for i,line in enumerate((OLD/'observations.jsonl').open()):
        if i==16:break
        r=json.loads(line);tracks=tuple(Track(t['pedestrian_id'],np.array([p[1] for p in t['points']]),np.array([[p[2],p[3]] for p in t['points']])) for t in r['tracks'])
        primary=next(t for t in tracks if t.pedestrian==r['primary_id'])
        obs=Observation(r['sample_id'],r['dataset'],r['primary_id'],(),primary.xy,tracks)
        value=tokenize(obs)
        token_diff=max(token_diff,max(float(np.max(np.abs(getattr(value,k)-getattr(old,k)[i]))) for k in value.__dataclass_fields__))
    assert token_diff<1e-10,token_diff
    fps_checks=[]
    for fps in (2.,2.5):
        times=np.arange(9)/fps;xy=np.column_stack((2*times,-times));history=[dict(time_s=float(t),x=float(v[0]),y=float(v[1])) for t,v in zip(times,xy)]
        future=(8+np.arange(1,13))/fps;obs=Observation(str(fps),'synthetic_boundary_check',1,(),xy,(Track(1,times,xy),));tokens=tokenize(obs)
        cv=np.column_stack((2*future,-future));assert np.allclose(tokens.cv,cv,atol=1e-12)
        kf=np.array([[v['x'],v['y']] for v in forecast(history,future,'kalman')]);assert np.allclose(tokens.kalman,kf,atol=1e-12)
        changed=Observation('future','check',1,(),xy,(Track(1,np.r_[times[:-1],times[-1]+1],xy),))
        try:tokenize(changed)
        except Exception:pass
        else:raise AssertionError('future timing accepted')
        fps_checks.append({'fps':fps,'cutoff':float(times[-1]),'last_future':float(future[-1]),'cv_maxdiff':float(np.max(np.abs(tokens.cv-cv))),'KF_exact_same_physical_times':True})
    train=load(ROOT/'pilot/cff_06_inputs.npz');dev=load(ROOT/'pilot/biwi_hotel_inputs.npz')
    labels=np.load(ROOT/'pilot/cff_06_labels.npz')['xy']
    torch.manual_seed(20261110);net=Recurrent(cfg).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01)
    x=tensors(train,'cuda');truth=torch.as_tensor(labels,device='cuda',dtype=torch.float32)
    rotation=torch.as_tensor(train.rotation.transpose(0,2,1),device='cuda',dtype=torch.float32)
    scale=torch.as_tensor(train.scale[:,None,None],device='cuda',dtype=torch.float32)
    anchor=torch.as_tensor(np.repeat(train.origin[:,None],12,axis=1),device='cuda',dtype=torch.float32)
    assert np.array_equal(predict(net,dev,4),np.repeat(dev.origin[:,None],12,axis=1))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic();epochs=[]
    for epoch in range(3):
        t=time.monotonic();net.train();losses=[]
        for ids in torch.randperm(len(train.ego),device='cuda').split(256):
            offset=net(*(v[ids] for v in x),4);delta=torch.bmm(offset,rotation[ids])*scale[ids]
            loss=torch.linalg.vector_norm(anchor[ids]+delta-truth[ids],dim=-1).mean()+.01*delta.square().mean();assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        torch.cuda.synchronize();epochs.append({'epoch':epoch+1,'seconds':time.monotonic()-t,'train_loss':float(np.mean(losses))})
    train_seconds=time.monotonic()-started
    torch.save({'state_dict':net.state_dict(),'config':dataclasses.asdict(cfg),'pilot_only':True},ROOT/'pilot/pilot_model.pt')
    torch.cuda.synchronize();t=time.monotonic();pred=predict(net,dev,4);torch.cuda.synchronize();inference_seconds=time.monotonic()-t
    assert np.isfinite(pred).all();np.save(ROOT/'pilot/dev_predictions.npy',pred)
    report={'status':'pilot_only_not_final','old_full1943_maxdiff':difference,'old_tokens16_maxdiff':token_diff,'physical_fps_checks':fps_checks,'train_n':len(train.ego),'dev_forward_n':len(dev.ego),'epochs':epochs,'train_seconds':train_seconds,'dev_forward_seconds':inference_seconds,'peak_vram_allocated':torch.cuda.max_memory_allocated(),'peak_vram_reserved':torch.cuda.max_memory_reserved(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_physical':1,'device':torch.cuda.get_device_name(),'cpu_threads':torch.get_num_threads(),'plan_sha256':hashlib.sha256((ROOT/'runplan.json').read_bytes()).hexdigest(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'API_calls':0,'evaluation_labels_read':False,'dev_labels_read':False}
    (ROOT/'pilot/gpu_receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
