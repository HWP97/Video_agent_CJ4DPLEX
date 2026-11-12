#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: imported by train.py/infer.py; no labels in forward
"""Shared recurrent attention over observation and future-query tokens."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Final
import numpy as np
import torch
from torch import nn
from common import BoundaryError
from features import Array,Batch,decode

@dataclass(frozen=True,slots=True)
class Config:
    name: str
    anchored: bool
    depth: int
    shared: bool=True
    dynamic: bool=False
    width: int=48

CONFIGS: Final=tuple(Config(f'{anchor}_{name}',anchored,depth,shared,dynamic,width)
    for anchor,anchored in (('direct',False),('kf',True))
    for name,depth,shared,dynamic,width in (
        ('single',1,True,False,48),('shared2',2,True,False,48),('shared4',4,True,False,48),
        ('dynamic',4,True,True,48),('unshared4',4,False,False,48),('unshared4_small',4,False,False,24)))

class Block(nn.Module):
    def __init__(self,width: int) -> None:
        super().__init__();self.width=width
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        with torch.no_grad():
            self.proj.weight.mul_(.1);self.proj.bias.zero_();self.ff[-1].weight.mul_(.1);self.ff[-1].bias.zero_()
    def forward(self,x: torch.Tensor,valid: torch.Tensor) -> torch.Tensor:
        b,n,_=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,4,self.width//4).permute(2,0,3,1,4).unbind(0)
        weights=(q@k.transpose(-1,-2)/(self.width//4)**.5).masked_fill(~valid[:,None,None,:],-1e4).softmax(-1)
        x=x+self.proj((weights@v).transpose(1,2).reshape(b,n,self.width))
        return x+self.ff(self.norm2(x))

class Recurrent(nn.Module):
    def __init__(self,cfg: Config) -> None:
        super().__init__();self.cfg=cfg
        self.ego_encoder=nn.Linear(5,cfg.width);self.neighbor_encoder=nn.Linear(9,cfg.width);self.query_encoder=nn.Linear(3,cfg.width)
        self.blocks=nn.ModuleList([Block(cfg.width) for _ in range(1 if cfg.shared else cfg.depth)])
        self.norm=nn.LayerNorm(cfg.width);self.head=nn.Linear(cfg.width,2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,ego: torch.Tensor,neighbors: torch.Tensor,mask: torch.Tensor,path: torch.Tensor,depth: int) -> torch.Tensor:
        if depth<1 or (not self.cfg.shared and depth!=self.cfg.depth):raise BoundaryError('invalid_depth',str(depth))
        b=len(ego);times=torch.arange(1,13,device=ego.device,dtype=ego.dtype)[None,:,None].expand(b,-1,-1)/12
        coordinates=path if self.cfg.anchored else torch.zeros_like(path)
        query=torch.cat((times,coordinates),dim=-1)
        x=torch.cat((self.ego_encoder(ego),self.neighbor_encoder(neighbors),self.query_encoder(query)),dim=1)
        valid=torch.cat((torch.ones((b,9),device=ego.device,dtype=torch.bool),mask.bool(),torch.ones((b,12),device=ego.device,dtype=torch.bool)),dim=1)
        for i in range(depth):x=self.blocks[0 if self.cfg.shared else i](x,valid)
        return self.head(self.norm(x[:,-12:]))

def tensors(batch: Batch,device: str) -> tuple[torch.Tensor,...]:
    path=np.matmul(batch.kalman-batch.origin[:,None],batch.rotation)/batch.scale[:,None,None]
    return tuple(torch.as_tensor(a,dtype=torch.float32,device=device) for a in (batch.ego,batch.neighbors,batch.mask,path))

def predict(net: Recurrent,batch: Batch,depth: int,device: str='cuda',no_neighbors: bool=False) -> Array:
    net.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(batch.ego),256):
            sub=batch.take(list(range(start,min(start+256,len(batch.ego)))))
            e,n,m,p=tensors(sub,device)
            if no_neighbors:m=torch.zeros_like(m)
            parts.append(net(e,n,m,p,depth).cpu().numpy().astype(np.float64))
    return decode(np.concatenate(parts),batch,net.cfg.anchored)
