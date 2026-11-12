"""Paper-inspired reduced deterministic cores; not complete paper models."""
from dataclasses import dataclass
from typing import Final
import numpy as np
import torch
from torch import nn

HIDDEN: Final = 48

@dataclass(frozen=True, slots=True)
class Config:
    family: str
    variant: str
    partitions: int = 8
    spectral: bool = True
    resonance: bool = True
    higher: bool = True
    eta_init: float = .1
    direction: float = .05

CONFIGS: Final = (
 Config('re','default'), Config('re','no_resonance',resonance=False),
 Config('re','identity_transform',spectral=False), Config('re','partitions4',partitions=4),
 Config('moif','default'), Config('moif','first_only',higher=False),
 Config('moif','no_direction',direction=0.),Config('moif','eta_zero',eta_init=0.),
)

def haar(x: torch.Tensor) -> torch.Tensor:
    """Official one-level Haar ordering: low XY then high XY."""
    return torch.cat(((x[..., ::2,:]+x[...,1::2,:])/2**.5,
                      (x[..., ::2,:]-x[...,1::2,:])/2**.5),dim=-1)

class Core(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config=config
        self.self_encoder=nn.Sequential(nn.Linear(18,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.self_decoder=nn.Linear(HIDDEN,24)
        if config.family == 're':
            channels=4 if config.spectral else 2
            steps=4 if config.spectral else 8
            self.spectral_encoder=nn.Sequential(nn.Linear(channels,HIDDEN),nn.ReLU())
            self.product=nn.Sequential(nn.Linear(steps*HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,24),nn.ReLU())
            self.position=nn.Sequential(nn.Linear(2,24),nn.ReLU())
            self.decoder=nn.Sequential(nn.Linear(config.partitions*48+HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,24))
        else:
            self.u=nn.Sequential(nn.Linear(36,HIDDEN),nn.ReLU())
            self.s=nn.Sequential(nn.Linear(45,HIDDEN),nn.ReLU())
            self.qs=nn.Linear(HIDDEN,HIDDEN);self.ks=nn.Linear(HIDDEN,HIDDEN);self.vs=nn.Linear(HIDDEN,HIDDEN)
            self.qu=nn.Linear(HIDDEN,HIDDEN);self.ku=nn.Linear(HIDDEN,HIDDEN)
            self.eta=nn.Parameter(torch.full((6,),config.eta_init))
            self.norm1=nn.LayerNorm(HIDDEN);self.norm2=nn.LayerNorm(HIDDEN)
            self.ff=nn.Sequential(nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN))
            self.decoder=nn.Sequential(nn.Linear(HIDDEN*2,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,24))

    def forward(self, ego: torch.Tensor, neighbors: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor | None = None) -> torch.Tensor:
        if dt is None:
            dt=torch.full((len(ego),),.4,device=ego.device,dtype=ego.dtype)
        assert dt.shape==(len(ego),) and torch.isfinite(dt).all() and (dt>0).all()
        own=self.self_encoder(ego.flatten(1))
        base_self=self.self_decoder(own).reshape(-1,12,2)
        if self.config.family == 're':
            if not self.config.resonance:
                return base_self
            centered=neighbors-neighbors[:,:,-1:]
            ep=ego[:,-8:]-ego[:,-1:]
            npart=centered[:,:,-8:]
            if self.config.spectral:
                ep,npart=haar(ep),haar(npart)
            ef=self.spectral_encoder(ep)
            nf=self.spectral_encoder(npart)
            product=self.product((ef[:,None]*nf).flatten(-2))
            pos=neighbors[:,:,-1]
            distance=pos.norm(dim=-1)
            angle=torch.atan2(pos[...,0],pos[...,1])%(2*torch.pi)
            bucket=(angle/(2*torch.pi/self.config.partitions)).long()
            parts=[]
            for p in range(self.config.partitions):
                weights=mask*(bucket==p)
                den=weights.sum(1,keepdim=True).clamp_min(1)
                spec=(product*weights[...,None]).sum(1)/den
                geom=torch.stack(((distance*weights).sum(1),(angle*weights).sum(1)),dim=-1)/den
                parts.append(torch.cat((spec,self.position(geom)),dim=-1))
            social=torch.cat(parts,dim=-1)
            return base_self+self.decoder(torch.cat((own,social),dim=-1)).reshape(-1,12,2)
        velocity=torch.diff(neighbors,dim=2,prepend=neighbors[:,:,:1])/dt[:,None,None,None]
        ev=torch.diff(ego,dim=1,prepend=ego[:,:1])/dt[:,None,None]
        rel=neighbors-ego[:,None]
        rv=velocity-ev[:,None]
        distance=rel.norm(dim=-1)
        cosine=(velocity*ev[:,None]).sum(-1)/(velocity.norm(dim=-1)*ev.norm(dim=-1)[:,None]).clamp_min(1e-6)
        angle=torch.acos(cosine.clamp(-.99999,.99999))
        lam=((rel*rv).sum(-1)/rv.square().sum(-1).clamp_min(1e-6)).abs()
        remaining=torch.arange(8,-1,-1,device=ego.device)[None,None,:]*dt[:,None,None]
        projected=(rel+torch.minimum(lam,remaining)[...,None]*rv).norm(dim=-1)
        s=self.s(torch.cat((ego[:,None].expand(-1,neighbors.shape[1],-1,-1),distance[...,None],angle[...,None],projected[...,None]),dim=-1).flatten(-2))
        u=self.u(torch.cat((neighbors,velocity),dim=-1).flatten(-2))
        valid=mask.bool();safe=valid.clone();safe[:,0]=True
        ws=(self.qs(s)@self.ks(s).transpose(1,2)/HIDDEN).masked_fill(~safe[:,None],-1e4).softmax(-1)
        fused=ws
        if self.config.higher:
            qu=self.qu(u).reshape(len(u),-1,6,8).transpose(1,2)
            ku=self.ku(u).reshape(len(u),-1,6,8).transpose(1,2)
            wu=(qu@ku.transpose(-1,-2)/8).masked_fill(~safe[:,None,None],-1e4).softmax(-1)
            fused=fused+(wu*self.eta[None,:,None,None]).sum(1)
        mixed=self.norm1(s+fused@self.vs(s))
        mixed=self.norm2(mixed+self.ff(mixed))
        social=(mixed*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return base_self+self.decoder(torch.cat((own,social),dim=-1)).reshape(-1,12,2)
