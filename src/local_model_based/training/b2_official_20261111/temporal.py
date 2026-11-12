# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: imported by experiment.py; CPU-only deterministic temporal learner.
"""Small temporal representation with same-video order contrast."""
from typing import Literal
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F

Arm = Literal['direct', 'residual', 'ridge']

class Temporal(nn.Module):
    """Fixed parameter count; local-only arm never consumes Sol numbers."""
    def __init__(self, arm: Arm, hidden: int=32) -> None:
        super().__init__()
        self.arm=arm
        self.gru=nn.GRU(32,hidden,batch_first=True)
        self.head=nn.Linear(hidden+16,8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Last recurrent state integrates ordered observations."""
        return self.gru(x)[1][-1]

    def forward(self, x: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        """Predict eight coordinates or their Sol residuals."""
        if self.arm=='direct':
            numeric=torch.cat((torch.zeros_like(numeric[:,:10]),numeric[:,10:]),1)
        return self.head(torch.cat((self.encode(x),numeric),1)).reshape(-1,4,2)

def views(x: torch.Tensor, generator: torch.Generator, hard: bool=False) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Keep the endpoint and frame multiset fixed in both negative sequences."""
    length=x.shape[1]
    drop=torch.randperm(length-1,generator=generator)[:2]
    keep=[i for i in range(length) if i not in drop.tolist()]
    negatives=[]
    for _ in range(2):
        indices=torch.randperm(length-1,generator=generator).tolist()
        if hard:
            indices=list(range(length-1)); left=int(torch.randint(length-2,(1,),generator=generator))
            indices[left],indices[left+1]=indices[left+1],indices[left]
        if indices==list(range(length-1)):
            indices=indices[1:]+indices[:1]
        negatives.append(x[:,indices+[length-1]])
    return x[:,keep],negatives[0],negatives[1]

def contrast(model: Temporal, x: torch.Tensor, transformed: tuple[torch.Tensor,torch.Tensor,torch.Tensor]) -> torch.Tensor:
    """InfoNCE uses only same-video views, never target labels."""
    anchor=F.normalize(model.encode(x),dim=-1)
    candidates=torch.stack([F.normalize(model.encode(view),dim=-1) for view in transformed],1)
    logits=(anchor[:,None,:]*candidates).sum(-1)/.1
    return F.cross_entropy(logits,torch.zeros(len(x),dtype=torch.long,device=logits.device))

def fit_projection(x: NDArray, numeric: NDArray) -> tuple[NDArray,NDArray,NDArray,NDArray,NDArray]:
    """Fit every normalization and PCA component on training observations only."""
    flat=x.reshape(-1,x.shape[-1]).astype(np.float64)
    mean=flat.mean(0); scale=np.maximum(flat.std(0),1e-4)
    z=(flat-mean)/scale
    _,_,vectors=np.linalg.svd(z,full_matrices=False)
    return mean,scale,vectors[:32].T,numeric.mean(0),np.maximum(numeric.std(0),1e-4)

def project(x: NDArray, numeric: NDArray, values: tuple) -> tuple[torch.Tensor,torch.Tensor]:
    """Apply frozen preprocessing without fitting on inference data."""
    mean,scale,basis,nmean,nscale=values
    transformed=((x-mean)/scale)@basis/np.sqrt(32.)
    return torch.tensor(transformed,dtype=torch.float32),torch.tensor((numeric-nmean)/nscale,dtype=torch.float32)
