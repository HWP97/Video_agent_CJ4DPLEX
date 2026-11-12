"""실제 2.5fps 장면에서 기존 paper 함수와 관측 cache의 입력 동등성을 확인한다."""
import ast
import json
from pathlib import Path
import numpy as np
from adapter_io import ROOT, sha
from data import Observation, Track
from observed_io import load_observed_role

source=Path('/mnt/RG2/paper_experiments_20261109/b3/experiment.py')
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='observed_arrays')
namespace={'np':np,'Observation':Observation};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
records=[]
for fold,stem in [(1,'biwi_hotel'),(3,'crowds_students001')]:
    plan=json.loads((ROOT/'runplan.json').read_text());group=plan['folds'][fold-1]['development'];files=plan['groups'][group]
    offset=0
    for row in files:
        if Path(row['file']).stem==stem:break
        offset+=row['scenes']
    indices=json.loads((ROOT/'data/indices'/f'{stem}_index.json').read_text())[:32]
    raw=np.load(ROOT/'evaluation_only'/f'{stem}.npz');tracks=raw['tracks'];scenes={int(r[0]):int(r[1]) for r in raw['scenes']};observations=[]
    for row in indices:
        primary=scenes[int(row['sample_id'].rsplit('_',1)[1])];frames=row['observed_frames'];selected=tracks[np.isin(tracks[:,0],frames)];parts=[]
        for pid in np.unique(selected[:,1]):
            tr=selected[selected[:,1]==pid];times=np.array([frames.index(int(f))*.4 for f in tr[:,0]])
            parts.append(Track(int(pid),times,tr[:,2:]))
        own=next(t.xy for t in parts if t.pedestrian==primary)
        observations.append(Observation(row['sample_id'],row['dataset'],primary,tuple(frames),own,tuple(parts)))
    for count,steps,include in [(16,9,False),(49,8,True)]:
        original=namespace['observed_arrays'](observations,count,steps,include);current=load_observed_role(fold,'dev',steps,count,include)
        differences=[float(np.abs(a-b[offset:offset+len(a)]).max()) for a,b in zip(original,current)]
        assert max(differences)<=1e-6,(stem,steps,differences)
        records.append({'file':stem,'count':count,'steps':steps,'include_ego':include,'cases':len(observations),'maxdiff':differences})
result={'passed':True,'source_sha256':sha(source),'adapter_sha256':sha(ROOT/'code/observed_io.py'),'records':records,'tolerance':1e-6,'reason':'float32 interpolation roundoff; ordering/masks require exact equality'}
(ROOT/'pilot/observed_array_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
