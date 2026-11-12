# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
# 실행: train_sweep.py가 호출
"""원 log-score 실험의 family rank + 0.5 local-teacher KL 목적함수."""
import torch


def family_rank(logscores: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    result=torch.zeros_like(logscores)
    for start,end in ((0,3),(3,9),(9,26)):
        score=logscores[:,start:end];negative=1-target[:,start:end]
        loss=torch.nn.functional.softplus(score[:,None,:]-score[:,:,None]+.1)
        result[:,start:end]=(loss*negative[:,None,:]).sum(-1)/negative.sum(-1,keepdim=True).clamp_min(1)
    return result


def teacher_kl(logscores: torch.Tensor,base: torch.Tensor) -> torch.Tensor:
    values=[]
    for start,end in ((0,3),(3,9),(9,26)):
        teacher=base[:,start:end]/base[:,start:end].sum(-1,keepdim=True).clamp_min(1e-6)
        values.append((teacher*(teacher.clamp_min(1e-6).log()-logscores[:,start:end].log_softmax(-1))).sum(-1).mean())
    return torch.stack(values).mean()


def restored_scores(logscores: torch.Tensor,base: torch.Tensor,raw_base: torch.Tensor | None = None) -> torch.Tensor:
    return (base if raw_base is None else raw_base)*(logscores-base.log()).exp()
