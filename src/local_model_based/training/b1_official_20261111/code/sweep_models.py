# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
# 실행: 학습·추론 모듈이 호출
"""전조건 고정 local head 및 target-conditioned masked 복원 구조."""
import torch
from torch import nn


class Head(nn.Module):
    def __init__(self,name: str):
        super().__init__();self.name=name
        self.direct='direct' in name or name=='family_head'
        linear='linear' in name or name.startswith(('macro','rank','log_rank'))
        self.net=nn.Linear(311,26) if linear else nn.Sequential(nn.Linear(311,64),nn.Tanh(),nn.Linear(64,26))
        if name=='family_head':
            self.families=nn.ModuleList([nn.Linear(311,n) for n in (3,6,17)])

    def forward(self,history,delta,base):
        h=history
        if 'bag' in self.name:
            h=history.mean(1,keepdim=True).expand_as(history)
        if 'category' in self.name:
            h=history.clone();h[:,:,26:63]*=2
        x=torch.cat((h.flatten(1),delta[:,None]/30,base),-1)
        value=torch.cat([head(x) for head in self.families],-1) if self.name=='family_head' else self.net(x)
        if self.direct:
            return value
        correction=.75*value.tanh()
        return base.log()+correction if self.name=='log_rank_residual' else torch.logit(base)+correction


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(279,64),nn.Tanh(),nn.Linear(64,3))

    def forward(self,context,revealed,visible):
        n=len(context)
        features=torch.cat((context[:,None,:].expand(-1,26,-1),torch.eye(26)[None].expand(n,-1,-1)), -1)
        condition=torch.cat((revealed.flatten(1),visible.flatten(1)),-1)
        return self.net(torch.cat((features,condition[:,None,:].expand(-1,26,-1)),-1))


def denoise(context,model,steps: int):
    revealed=torch.zeros(len(context),26,3);visible=torch.zeros_like(revealed)
    for step in range(steps):
        output=model(context,revealed,visible).sigmoid()
        if step+1<steps:
            confidence=(output-.5).abs().flatten(1);confidence[visible.flatten(1).bool()]=-1
            indices=confidence.topk(78//steps,dim=1).indices
            newly=torch.zeros_like(visible);newly.flatten(1).scatter_(1,indices,1)
            revealed=torch.where(newly.bool(),output,revealed);visible=torch.maximum(visible,newly)
    return torch.where(visible.bool(),revealed,output)
