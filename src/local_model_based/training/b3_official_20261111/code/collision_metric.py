"""공식 endpoint/midpoint 충돌 검사를 여러 예측에 공유한다. 평가 전용."""
import numpy as np
from scipy.spatial import cKDTree


def collision_segments(predictions,first,last,start,end,visible):
    """이미 ID순서로 정리된 공통미래 구간을 공식3점 표본으로 비교한다."""
    predictions=np.asarray(predictions,dtype=np.float64)
    all_hits=np.zeros(len(predictions),bool);observed_hits=all_hits.copy()
    keys=np.r_[first*12+first,last*12+last,first*12+last]
    points=np.concatenate((start,end,start+(end-start)*.5))
    observed=np.tile(visible,3)
    for key in np.unique(keys):
        a,b=divmod(int(key),12);mask=keys==key
        query=predictions[:,a] if a==b else predictions[:,a]+(predictions[:,b]-predictions[:,a])*.5
        bounded=mask&np.all((points>=query.min(0)-.2)&(points<=query.max(0)+.2),axis=1)
        for condition,hits in ((bounded,all_hits),(bounded&observed,observed_hits)):
            ids=np.flatnonzero(~hits)
            if condition.any() and len(ids):hits[ids]=cKDTree(points[condition]).query(query[ids],k=1,workers=1)[0]<=.2
    return all_hits,observed_hits


def collision_many(predictions,future_frames,neighbor_paths,primary_id,observed_ids):
    predictions=np.asarray(predictions,dtype=np.float64)
    assert predictions.ndim==3 and predictions.shape[1:]==(12,2) and np.isfinite(predictions).all()
    mapping={int(frame):i for i,frame in enumerate(future_frames)}
    points_all={};points_observed={}
    for pedestrian,path in neighbor_paths.items():
        if pedestrian==primary_id:continue
        common=sorted((mapping[int(frame)],np.asarray(xy,dtype=np.float64)) for frame,xy in path if int(frame) in mapping)
        if len(common)<2:continue
        for (first,a),(last,b) in zip(common,common[1:]):
            for key,point in [((first,first),a),((last,last),b),((first,last),a+(b-a)*.5)]:
                points_all.setdefault(key,[]).append(point)
                if pedestrian in observed_ids:points_observed.setdefault(key,[]).append(point)
    outcomes=[]
    for groups in (points_all,points_observed):
        hit=np.zeros(len(predictions),bool)
        for (first,last),points in groups.items():
            ids=np.flatnonzero(~hit)
            if not len(ids):break
            query=predictions[ids,first] if first==last else predictions[ids,first]+(predictions[ids,last]-predictions[ids,first])*.5
            distance=cKDTree(np.asarray(points)).query(query,k=1,workers=1)[0]
            hit[ids]=distance<=.2
        outcomes.append(hit)
    return tuple(outcomes)
