"""최근접8을 먼저 고르고 기존 회귀를 그대로 사용하여 수치값을 보존한다."""
from dataclasses import replace
import numpy as np
from common import BoundaryError
from features_reference import tokenize as reference, velocity


def tokenize(obs):
    primary=next(t for t in obs.tracks if t.pedestrian==obs.primary)
    dt=float(primary.times[-1]/8);cutoff=8*dt
    candidates=[]
    for t in obs.tracks:
        if len(t.times)!=len(t.xy) or not np.isfinite(t.times).all() or not np.isfinite(t.xy).all() or np.any(np.diff(t.times)<=0):
            raise BoundaryError('invalid_track',obs.sample_id)
        if len(t.times) and (t.times.min()<0 or t.times.max()>cutoff+1e-8):
            raise BoundaryError('future_observation',obs.sample_id)
        if t.pedestrian==obs.primary or len(t.times)<2 or t.times[-1]<cutoff-dt-1e-8:
            continue
        # 현재 시각에 도달한 이웃은 속도를 계산해도 외삽 간격이 정확히0이다.
        current=t.xy[-1] if t.times[-1]==cutoff else t.xy[-1]+(cutoff-t.times[-1])*velocity(t.times[-3:],t.xy[-3:])
        candidates.append((float(np.linalg.norm(current-obs.xy[-1])),t.pedestrian,t))
    selected=[t for _,_,t in sorted(candidates,key=lambda row:(row[0],row[1]))[:8]]
    return reference(replace(obs,tracks=(primary,*selected)))
