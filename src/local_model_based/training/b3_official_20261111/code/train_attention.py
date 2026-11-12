"""동결 outerfold의 train/dev만 읽는 attention 학습. eval GT 접근 금지."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
import numpy as np
import torch
from features import Batch, load
from model import CONFIGS, Recurrent, predict, tensors

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_data(plan,name):
    files=[Path(r['file']).stem for r in plan['groups'][name]]
    batches=[load(ROOT/'data/inputs'/f'{n}_inputs.npz') for n in files]
    batch=Batch(**{k:np.concatenate([getattr(b,k) for b in batches]) for k in Batch.__dataclass_fields__})
    labels=np.concatenate([np.load(ROOT/'data/targets'/f'{n}_labels.npz')['xy'] for n in files])
    return batch,labels


def main():
    fold_id=int(sys.argv[1]);only=sys.argv[2] if len(sys.argv)>2 else 'all'
    plan=json.loads((ROOT/'runplan.json').read_text());fold=plan['folds'][fold_id-1]
    allowed={Path(r['file']).stem+'_labels.npz' for g in (fold['train'],fold['development']) for r in plan['groups'][g]}
    def guard(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('network_forbidden')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]))
            if path.name.endswith('_labels.npz') and path.name not in allowed:raise PermissionError('outer_eval_labels_forbidden')
    sys.addaudithook(guard);torch.set_num_threads(2)
    train,y=group_data(plan,fold['train']);dev,dy=group_data(plan,fold['development'])
    directory=ROOT/'attention'/f'fold{fold_id}';directory.mkdir(parents=True,exist_ok=True)
    source={name:sha(Path(__file__).parent/name) for name in ('train_attention.py','model.py','features.py','common.py')}
    input_hashes={str(p.relative_to(ROOT)):sha(p) for group in (fold['train'],fold['development']) for row in plan['groups'][group] for p in [ROOT/'data/receipts'/(Path(row['file']).stem+'.json')]}
    for cfg in CONFIGS:
        if only!='all' and cfg.name!=only:continue
        receipt=directory/(cfg.name+'.json');checkpoint=directory/(cfg.name+'.pt')
        signature={'runplan':sha(ROOT/'runplan.json'),'config':asdict(cfg),'source':source,'data':input_hashes}
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['signature']==signature and sha(checkpoint)==old['checkpoint_sha256'];continue
        torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);rng=np.random.default_rng(plan['seed'])
        net=Recurrent(cfg).cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01)
        x=tensors(train,'cuda');truth=torch.tensor(y,dtype=torch.float32,device='cuda')
        rotation=torch.tensor(train.rotation.transpose(0,2,1),dtype=torch.float32,device='cuda');scale=torch.tensor(train.scale[:,None,None],dtype=torch.float32,device='cuda')
        base=train.kalman if cfg.anchored else np.repeat(train.origin[:,None],12,axis=1)
        anchor=torch.tensor(base,dtype=torch.float32,device='cuda')
        torch.cuda.reset_peak_memory_stats();started=time.monotonic();best=float('inf');stale=0;curve=[]
        for epoch in range(1,41):
            t=time.monotonic();net.train();losses=[]
            for ids in torch.randperm(len(y),device='cuda').split(256):
                depth=int(np.clip(rng.poisson(4),1,8)) if cfg.dynamic else cfg.depth
                offset=net(*(v[ids] for v in x),depth);delta=torch.bmm(offset,rotation[ids])*scale[ids]
                loss=torch.linalg.vector_norm(anchor[ids]+delta-truth[ids],dim=-1).mean()+.01*delta.square().mean();assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
            torch.cuda.synchronize();train_sec=time.monotonic()-t;t=time.monotonic();pred=predict(net,dev,cfg.depth);error=np.linalg.norm(pred-dy,axis=-1)
            row={'epoch':epoch,'train_loss':float(np.mean(losses)),'dev_ADE':float(error.mean()),'dev_FDE':float(error[:,-1].mean()),'train_seconds':train_sec,'dev_seconds':time.monotonic()-t};curve.append(row)
            if row['dev_ADE']<best-1e-12:
                best=row['dev_ADE'];stale=0;best_epoch=epoch
                temporary=checkpoint.with_suffix('.tmp');torch.save({'state_dict':net.state_dict(),'config':asdict(cfg),'signature':signature},temporary);temporary.replace(checkpoint)
            else:stale+=1
            print(json.dumps({'fold':fold_id,'config':cfg.name,**row}),flush=True)
            (directory/(cfg.name+'_working.json')).write_text(json.dumps({'signature':signature,'curve':curve,'status':'training'}))
            if stale>=8:break
        net.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict'])
        depths=(1,2,4,8) if cfg.shared and cfg.depth>1 else (cfg.depth,)
        diagnostics=[]
        for depth in depths:
            pred=predict(net,dev,depth);np.save(directory/f'{cfg.name}_dev_d{depth}.npy',pred);err=np.linalg.norm(pred-dy,axis=-1)
            diagnostics.append({'depth':depth,'ADE':float(err.mean()),'FDE':float(err[:,-1].mean())})
        report={'signature':signature,'fold':fold_id,'config':asdict(cfg),'train_n':len(y),'dev_n':len(dy),'best_epoch':best_epoch,'curve':curve,'diagnostics':diagnostics,'seconds':time.monotonic()-started,'peak_vram':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'checkpoint_sha256':sha(checkpoint),'GPU_physical':os.environ['CUDA_VISIBLE_DEVICES'],'eval_GT_opened':False,'API':0}
        receipt.write_text(json.dumps(report,indent=2));print(json.dumps({'completed':cfg.name,'fold':fold_id,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
