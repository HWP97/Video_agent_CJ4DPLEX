"""이웃 범위·자기제외·시간교집합과 공식 함수 동등성을 검사한다."""
from collections import namedtuple
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
from collision_metric import collision_many,collision_segments

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/mnt/RG2/my_works/experiments/benchmarks/2026-11-06/001_frozen_full/sources/trajnetplusplustools/code/trajnetplusplustools/metrics.py')
spec=importlib.util.spec_from_file_location('official_metric',SOURCE);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
Track=namedtuple('Track','frame pedestrian x y')


def check(name,predictions,paths,visible,expected=None):
    frames=np.arange(12);actual=collision_many(predictions,frames,paths,1,set(visible));reference=[]
    for observed_only in (False,True):
        results=[]
        for positions in predictions:
            primary=[Track(int(f),1,*xy) for f,xy in zip(frames,positions)]
            results.append(any(official.collision(primary,[Track(int(f),ped,*xy) for f,xy in path]) for ped,path in paths.items() if ped!=1 and (not observed_only or ped in visible)))
        reference.append(np.array(results))
    assert all(np.array_equal(a,b) for a,b in zip(actual,reference)),name
    starts=[];ends=[];first=[];last=[];seen=[]
    for ped,path in paths.items():
        if ped==1:continue
        common=sorted((f,np.asarray(xy)) for f,xy in path if f in set(frames))
        for (a,x),(b,y) in zip(common,common[1:]):
            first.append(a);last.append(b);starts.append(x);ends.append(y);seen.append(ped in visible)
    batched=collision_segments(predictions,np.asarray(first,dtype=int),np.asarray(last,dtype=int),np.asarray(starts).reshape(-1,2),np.asarray(ends).reshape(-1,2),np.asarray(seen,dtype=bool))
    assert all(np.array_equal(a,b) for a,b in zip(batched,reference)),name
    if expected is not None:assert [a.tolist() for a in actual]==expected,name
    return {'scenario':name,'all_GT':actual[0].tolist(),'observed_GT':actual[1].tolist(),'official_equal':True}


if __name__=='__main__':
    line=np.column_stack((np.arange(12),np.zeros(12)));cross=np.column_stack((np.arange(12)[::-1],np.zeros(12)))
    path=lambda xy:list(zip(range(12),xy))
    outcomes=[check('self_path_excluded',line[None],{1:path(line)},[1],[[False],[False]]),
      check('crossing_midpoint',line[None],{2:path(cross)},[2],[[True],[True]]),
      check('future_only_neighbor',line[None],{2:path(cross)},[],[[True],[False]]),
      check('no_neighbors',line[None],{},[],[[False],[False]]),
      check('no_common_future',line[None],{2:[(-2,[0,0]),(-1,[0,0])]},[2],[[False],[False]]),
      check('one_common_frame_no_segment',line[None],{2:[(0,[0,0])]},[2],[[False],[False]]),
      check('noncrossing',line[None],{2:path(line+[0,5])},[2],[[False],[False]]),
      check('radius_boundary',line[None],{2:path(line+[0,.2])},[2],[[True],[True]])]
    rng=np.random.default_rng(42)
    for i in range(64):
        predictions=rng.normal(size=(4,12,2));paths={p:[(int(f),rng.normal(size=2)) for f in sorted(rng.choice(12,size=int(rng.integers(0,13)),replace=False))] for p in range(1,6)}
        outcomes.append(check('random_partial_'+str(i),predictions,paths,[2,4]))
    report={'official_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'cases':outcomes,'passed':True}
    (ROOT/'pilot/collision_equivalence.json').write_text(json.dumps(report,indent=2));print(json.dumps({'cases':len(outcomes),'passed':True}),flush=True)
