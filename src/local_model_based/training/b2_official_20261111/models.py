# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: imported by run.py in existing action_genome environment.
"""Exemplar memory, conventional low-rank regression, and future-summary learning."""
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

class Trace(TypedDict):
    epoch: int
    main_loss: float
    summary_loss: float

class SummaryMLP(nn.Module):
    """The auxiliary head receives only the observation representation."""
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(inputs,hidden),nn.Tanh())
        self.main=nn.Linear(hidden,8)
        self.aux=nn.Linear(hidden,4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        hidden=self.encoder(x)
        return self.main(hidden),self.aux(hidden)

def summaries(y: NDArray) -> NDArray:
    """Handcrafted global future mean and least-squares time slope per VA axis."""
    h=np.array([.5,1.,2.,3.]); t=h-h.mean()
    return np.concatenate((y.mean(1),(y*t[None,:,None]).sum(1)/(t@t)),1)

def train_summary(x: NDArray, target: NDArray, future: NDArray, weight: float, hidden: int=16, decay: float=.05) -> tuple[SummaryMLP,list[Trace]]:
    """Fix initialization, optimizer, epoch count, and architecture across loss ablations."""
    torch.manual_seed(42)
    model=SummaryMLP(x.shape[1],hidden)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=decay)
    xx=torch.tensor(x,dtype=torch.float32); yy=torch.tensor(target.reshape(-1,8),dtype=torch.float32)
    summary=summaries(future); summary=(summary-summary.mean(0))/np.maximum(summary.std(0),1e-4)
    ss=torch.tensor(summary,dtype=torch.float32); trace=[]
    for epoch in range(200):
        opt.zero_grad(); main,aux=model(xx)
        lm=(main-yy).square().mean(); ls=(aux-ss).square().mean()
        (lm+weight*ls).backward(); opt.step()
        if epoch%20==0 or epoch==199:
            trace.append({'epoch':epoch+1,'main_loss':float(lm.detach()),'summary_loss':float(ls.detach())})
    return model,trace

def neural_predict(model: SummaryMLP, x: NDArray) -> NDArray:
    """No future values or auxiliary outputs are needed at prediction time."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x,dtype=torch.float32))[0].numpy().reshape(-1,4,2)

def neighbors(query: NDArray, keys: NDArray, qgroups: NDArray, kgroups: NDArray, k: int, weighted: bool) -> tuple[NDArray,NDArray]:
    """Block original-group matches before choosing nearest training examples."""
    distance=np.maximum((query**2).sum(1)[:,None]+(keys**2).sum(1)[None,:]-2*query@keys.T,0)
    distance[qgroups[:,None]==kgroups[None,:]]=np.inf
    index=np.argsort(distance,axis=1,kind='stable')[:,:k]
    selected=np.take_along_axis(distance,index,axis=1)
    assert np.isfinite(selected).all()
    weights=np.ones_like(selected)
    if weighted:
        bandwidth=np.maximum(np.median(selected,axis=1,keepdims=True),1e-8)
        weights=np.exp(-(selected-selected[:,:1])/bandwidth)
    weights/=weights.sum(1,keepdims=True)
    return index,weights

def retrieve(values: NDArray, index: NDArray, weights: NDArray) -> NDArray:
    """Retrieve train continuations with normalized nonnegative weights."""
    return (values[index]*weights[:,:,None,None]).sum(1)

def linear_fit(x: NDArray, y: NDArray, rank: int) -> tuple[NDArray,NDArray,NDArray]:
    """Conventional target PCA truncation; rank8 is ordinary ridge, not FSP."""
    mean=x.mean(0); center=y.reshape(-1,8).mean(0); z=x-mean
    target=y.reshape(-1,8)-center
    _,_,v=np.linalg.svd(target,full_matrices=False); projection=v[:rank].T@v[:rank]
    coef=np.linalg.solve(z.T@z/len(z)+np.eye(z.shape[1]),z.T@target@projection/len(z))
    return mean,center,coef
