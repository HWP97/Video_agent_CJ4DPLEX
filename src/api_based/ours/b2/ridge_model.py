# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "typing-extensions"]
# ///
# How to run: imported by run.py using existing action_genome model/env/bin/python -B
"""Prediction-only ridge residuals with an exact zero-correction baseline."""
from dataclasses import dataclass
from typing import Literal, TypedDict
from typing_extensions import assert_never
import numpy as np
from numpy.typing import NDArray

Family = Literal['current_bias', 'sol_residual', 'observation_residual']
Array = NDArray[np.float64]

class Model(TypedDict):
    family: Family
    penalty: float
    mean: list
    scale: list
    coef: list
    intercept: list

@dataclass(frozen=True, slots=True)
class Inputs:
    baseline: Array
    current: Array
    local: Array
    horizons: Array

def features(inputs: Inputs, family: Family) -> Array:
    """Accept no GT fields; construct observed/predicted covariates only."""
    b,c,l=inputs.baseline,inputs.current,inputs.local
    match family:
        case 'current_bias':
            return np.concatenate((c,l,l[:,:2]-c),axis=-1)
        case 'sol_residual' | 'observation_residual':
            n,h,_=b.shape
            time=np.broadcast_to(inputs.horizons[None,:,None],(n,h,1))
            current=np.broadcast_to(c[:,None,:],b.shape)
            parts=[time,time**2,current,b,b-current]
            if family=='observation_residual':
                local=np.broadcast_to(l[:,None,:],(n,h,6))
                parts.extend([local,local[:,:,:2]-current,time*local[:,:,2:4]])
            return np.concatenate(parts,axis=-1).reshape(n*h,-1)
        case unreachable:
            assert_never(unreachable)

def fit(x: Array, target: Array, family: Family, penalty: float) -> Model:
    """Ridge with train-only standardization and unpenalized target mean."""
    mean=x.mean(0); scale=x.std(0); scale=np.where(scale>1e-12,scale,1.)
    z=(x-mean)/scale; intercept=target.mean(0)
    coef=np.linalg.solve(z.T@z/len(z)+penalty*np.eye(z.shape[1]),z.T@(target-intercept)/len(z))
    return {'family':family,'penalty':penalty,'mean':mean.tolist(),'scale':scale.tolist(),
            'coef':coef.tolist(),'intercept':intercept.tolist()}

def predict(inputs: Inputs, model: Model | None) -> tuple[Array,Array]:
    """Return raw and service-clipped forecasts; None is exact identity."""
    if model is None:
        return inputs.baseline.copy(),inputs.baseline.copy()
    x=features(inputs,model['family'])
    correction=(x-np.array(model['mean']))/np.array(model['scale'])@np.array(model['coef'])+np.array(model['intercept'])
    match model['family']:
        case 'current_bias':
            correction=correction[:,None,:]
        case 'sol_residual' | 'observation_residual':
            correction=correction.reshape(inputs.baseline.shape)
        case unreachable:
            assert_never(unreachable)
    raw=inputs.baseline+correction
    return raw,np.clip(raw,-1,1)
