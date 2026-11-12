# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
# 실행: 기존 AG Python -B code/train_sweep.py development|final
"""세부 local config를 기존 train 그룹으로 학습한다. 네트워크 입력은 없다."""
import hashlib
import json
import socket
import sys
import time
from pathlib import Path
import torch
from sweep_data import HERE,SOURCE,pack
from sweep_models import Head,Denoiser
from local_extensions import ranking_loss
from log_objective import family_rank,teacher_kl
from provenance import freeze,digest,training_signature


def deny(*args,**kwargs):
    raise PermissionError('Training is local-only')


def main() -> None:
    socket.socket.connect=deny;torch.set_num_threads(1)
    stage=sys.argv[1];assert stage in ('development','final')
    if stage=='final':
        assert (HERE/'protocol/selection.json').exists()
    plan=json.loads((HERE/'runplan.json').read_text())
    signature=freeze(HERE/f'protocol/{stage}_training_freeze.json',training_signature(HERE,SOURCE,stage))
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    for mode in ('sgcls','predcls'):
        data,rows=pack(mode,stage)
        support=data['target'].sum(0)
        weights=support.clamp_min(1).rsqrt();weights/=weights.mean()
        memory={'context':data['context'],'profile':data['profile'],'groups':sorted({r['video'] for r in rows})}
        memory_path=HERE/f'assets/{stage}_{mode}_memory.pt'
        memory_receipt=HERE/f'protocol/{stage}_{mode}_memory.json'
        if memory_path.exists():
            assert json.loads(memory_receipt.read_text())=={'sha256':digest(memory_path),'signature':signature}
        else:
            torch.save(memory,memory_path)
            freeze(memory_receipt,{'sha256':digest(memory_path),'signature':signature})
        for name in plan['head_configs']+plan['masked_patterns']:
            target=HERE/f'assets/{stage}_{mode}_{name}.pt'
            checkpoint_receipt=HERE/f'protocol/{stage}_{mode}_{name}_checkpoint.json'
            if target.exists():
                assert json.loads(checkpoint_receipt.read_text())=={'sha256':digest(target),'signature':signature}
                continue
            torch.manual_seed(plan['seed'])
            masked=name in plan['masked_patterns']
            model=Denoiser() if masked else Head(name)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            start=time.monotonic();losses=[]
            count=len(data['context'] if masked else data['target'])
            epochs=plan['masked_epochs'] if masked else plan['heads_epochs']
            for epoch in range(epochs):
                total=0.
                for indices in torch.randperm(count).split(64 if masked else 256):
                    if masked:
                        y=data['profile'][indices]
                        all_mask=name=='all_mask' or (name=='mixed50' and bool(torch.rand(())<.5))
                        probability=torch.ones(len(indices),1,1) if all_mask else .001+.999*torch.rand(len(indices),1,1)
                        mask=torch.rand_like(y)<probability;visible=(~mask).float()
                        logits=model(data['context'][indices],torch.where(mask,0.,y),visible)
                        element=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                        loss=(element*mask/probability).mean()
                    else:
                        history,delta,base,y=(data[k][indices] for k in ('history','delta','base','target'))
                        logits=model(history,delta,base)
                        element=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                        loss=(element*weights).mean() if name.startswith(('macro','rank','log_rank')) else element.mean()
                        if 'top_w' in name:
                            confidence=base.max(1).values
                            teacher=torch.nn.functional.binary_cross_entropy_with_logits(logits,base,reduction='none').mean(1)
                            loss=loss+(0.1 if name.endswith('w01') else 1.)*(teacher*confidence).mean()
                        if name=='prior_linear':
                            loss=loss+.1*torch.nn.functional.binary_cross_entropy_with_logits(logits,data['target'].mean(0).expand_as(y))
                        if name=='log_rank_residual':
                            positive=y/support.clamp_min(1)
                            loss=(family_rank(logits,y)*positive).sum()*count/(26*len(indices))+.5*teacher_kl(logits,base)
                        elif 'rank' in name:
                            loss=loss+.1*ranking_loss(logits,y)
                        if 'mtp' in name:
                            middle=model(history,data['middle_delta'][indices],base)
                            loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(middle,data['middle'][indices])
                    assert torch.isfinite(loss)
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                    total+=float(loss.detach())*len(indices)
                losses.append(total/count)
            receipt={'mode':mode,'method':name,'stage':stage,'groups':len(rows),'examples':count,'epochs':epochs,
                     'seconds':time.monotonic()-start,'losses':losses,'device':'cpu','peak_vram_bytes':0,'source_sha256':sources,
                     'runplan_sha256':hashlib.sha256((HERE/'runplan.json').read_bytes()).hexdigest()}
            temporary=target.with_suffix('.tmp')
            torch.save({'state':model.state_dict(),'method':name,'support':support,'receipt':receipt},temporary);temporary.replace(target)
            freeze(checkpoint_receipt,{'sha256':digest(target),'signature':signature})
            (HERE/f'results/{stage}_{mode}_{name}_training.json').write_text(json.dumps(receipt,indent=2))
            print(json.dumps({k:v for k,v in receipt.items() if k not in ('losses','source_sha256')}),flush=True)


if __name__=='__main__':
    main()
