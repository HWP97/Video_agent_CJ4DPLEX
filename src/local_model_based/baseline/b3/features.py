#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: imported by prepare.py and test_contracts.py
"""Observed-only tokenization with physical timestamps and explicit masks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from common import BoundaryError
from data import Observation
from trajectory_baselines import forecast
Array=NDArray[np.float64]

@dataclass(frozen=True,slots=True)
class Tokens:
    ego: Array
    neighbors: Array
    mask: Array
    rotation: Array
    scale: float
    origin: Array
    kalman: Array
    cv: Array

def velocity(times: Array,xy: Array) -> Array:
    return np.linalg.lstsq(np.column_stack((times-times[-1],np.ones(len(times)))),xy,rcond=None)[0][0]

def tokenize(obs: Observation) -> Tokens:
    if obs.xy.shape!=(9,2) or not np.isfinite(obs.xy).all():raise BoundaryError('invalid_primary',obs.sample_id)
    for track in obs.tracks:
        if len(track.times)!=len(track.xy) or not np.isfinite(track.times).all() or not np.isfinite(track.xy).all() or np.any(np.diff(track.times)<=0):
            raise BoundaryError('invalid_track',obs.sample_id)
        if len(track.times) and (track.times.min()<0 or track.times.max()>3.2+1e-8):
            raise BoundaryError('future_observation',obs.sample_id)
    primary=next(t for t in obs.tracks if t.pedestrian==obs.primary)
    if len(primary.times)!=9 or abs(primary.times[-1]-3.2)>1e-8:raise BoundaryError('invalid_cutoff',obs.sample_id)
    v=velocity(primary.times[-3:],obs.xy[-3:]);speed=float(np.linalg.norm(v))
    direction=v/speed if speed>1e-8 else np.array([1.,0.])
    rotation=np.array([[direction[0],-direction[1]],[direction[1],direction[0]]]);scale=max(1.,4.8*speed)
    pos=(obs.xy-obs.xy[-1])@rotation/scale
    vel=np.zeros((9,2));vel[1:]=np.diff(obs.xy,axis=0)/np.diff(primary.times)[:,None]@rotation*4.8/scale
    ego=np.column_stack((pos,vel,(primary.times-3.2)/4.8))
    rows=[]
    for t in obs.tracks:
        if t.pedestrian==obs.primary or len(t.times)<2 or t.times[-1]<2.8-1e-8:continue
        nv=velocity(t.times[-3:],t.xy[-3:]);current=t.xy[-1]+(3.2-t.times[-1])*nv
        rel=current-obs.xy[-1];rv=nv-v;distance=float(np.linalg.norm(rel))
        closing=-float(rel@rv)/max(distance,1e-8);tca=float(np.clip(-rel@rv/max(float(rv@rv),1e-8),0,4.8))
        values=np.r_[rel@rotation/scale,rv@rotation*4.8/scale,closing*4.8/scale,tca/4.8,
                     np.linalg.norm(rel+tca*rv)/scale,(3.2-t.times[-1])/.4,len(t.times)/9]
        rows.append((distance,t.pedestrian,values))
    neighbors=np.zeros((8,9));mask=np.zeros(8)
    for i,(_,_,values) in enumerate(sorted(rows,key=lambda r:(r[0],r[1]))[:8]):neighbors[i]=values;mask[i]=1
    history=[{'time_s':float(t),'x':float(p[0]),'y':float(p[1])} for t,p in zip(primary.times,primary.xy)]
    bases=[np.array([[r['x'],r['y']] for r in forecast(history,3.2+np.arange(1,13)*.4,method)]) for method in ('kalman','constant_velocity')]
    out=Tokens(ego,neighbors,mask,rotation,scale,obs.xy[-1].copy(),*bases)
    if not all(np.isfinite(np.asarray(getattr(out,k))).all() for k in Tokens.__dataclass_fields__):raise BoundaryError('nonfinite_tokens',obs.sample_id)
    return out

@dataclass(frozen=True,slots=True)
class Batch:
    ego: Array
    neighbors: Array
    mask: Array
    rotation: Array
    scale: Array
    origin: Array
    kalman: Array
    cv: Array
    def take(self,ids: list[int] | NDArray[np.int64]) -> Batch:
        return Batch(**{k:getattr(self,k)[ids] for k in self.__dataclass_fields__})

def load(path: Path) -> Batch:
    with np.load(path) as values:return Batch(**{k:values[k] for k in Batch.__dataclass_fields__})

def decode(offset: Array, batch: Batch, anchored: bool) -> Array:
    anchor=batch.kalman if anchored else np.repeat(batch.origin[:,None],12,axis=1)
    if not np.any(offset):return anchor.copy()
    result=anchor+np.matmul(offset,batch.rotation.transpose(0,2,1))*batch.scale[:,None,None]
    if not np.isfinite(result).all():raise BoundaryError('nonfinite_output')
    return result
