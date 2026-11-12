"""실제1000장면에서 최적화 충돌값을 공식 Python 함수와 직접 대조한다."""
from collections import defaultdict,namedtuple
from pathlib import Path
import importlib.util
import json
import sys
import time
import numpy as np
from adapter_io import ROOT,sha
from collision_metric import collision_segments

SOURCE=Path('/mnt/RG2/my_works/experiments/benchmarks/2026-11-06/001_frozen_full/sources/trajnetplusplustools/code/trajnetplusplustools/metrics.py')
spec=importlib.util.spec_from_file_location('official',SOURCE);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
Track=namedtuple('Track','frame pedestrian x y')


if __name__=='__main__':
    started=time.monotonic();plan=json.loads((ROOT/'runplan.json').read_text());records=[];official_seconds=0.;fast_seconds=0.
    cff=len(sys.argv)>1 and sys.argv[1]=='cff'
    cases=[('cff_06',1000,2)] if cff else [('crowds_students001',600,3),('biwi_hotel',200,1),('wildtrack',200,1)]
    for stem,count,fold in cases:
        group=plan['folds'][fold-1]['development'];offset=0
        for row in plan['groups'][group]:
            if Path(row['file']).stem==stem:break
            offset+=row['scenes']
        predictions=np.load(ROOT/'attention'/f'fold{fold}'/'direct_unshared4_dev.npy',mmap_mode='r')[offset:offset+count]
        index=json.loads((ROOT/'data/indices'/f'{stem}_index.json').read_text())[:count]
        raw=np.load(ROOT/'evaluation_only'/f'{stem}.npz');tracks=raw['tracks'];scenes={int(r[0]):int(r[1]) for r in raw['scenes']}
        frames=defaultdict(list)
        for f,p,x,y in tracks:frames[int(f)].append((int(p),float(x),float(y)))
        for prediction,row in zip(predictions,index):
            primary=scenes[int(row['sample_id'].rsplit('_',1)[1])];visible={p for f in row['observed_frames'] for p,_,_ in frames[f]};paths=defaultdict(list)
            for step,f in enumerate(row['future_frames']):
                for p,x,y in frames[f]:paths[p].append(Track(f,p,x,y))
            first=[];last=[];start=[];end=[];seen=[];mapping={f:i for i,f in enumerate(row['future_frames'])}
            for p,path in paths.items():
                if p==primary:continue
                for a,b in zip(path,path[1:]):
                    first.append(mapping[a.frame]);last.append(mapping[b.frame]);start.append([a.x,a.y]);end.append([b.x,b.y]);seen.append(p in visible)
            tick=time.monotonic();fast=collision_segments(prediction[None],np.array(first,dtype=int),np.array(last,dtype=int),np.array(start).reshape(-1,2),np.array(end).reshape(-1,2),np.array(seen,bool));fast_seconds+=time.monotonic()-tick
            pred=[Track(f,primary,*xy) for f,xy in zip(row['future_frames'],prediction)];tick=time.monotonic()
            expected=[any(official.collision(pred,path) for p,path in paths.items() if p!=primary and (not observed or p in visible)) for observed in [False,True]];official_seconds+=time.monotonic()-tick
            actual=[bool(v[0]) for v in fast];assert actual==expected,(row['sample_id'],actual,expected)
            records.append({'sample_id':row['sample_id'],'fps':row['fps'],'all_GT':actual[0],'observed_GT':actual[1],'official_equal':True})
        print(json.dumps({'dataset':stem,'checked':len(records),'seconds':time.monotonic()-started}),flush=True)
    assert len(records)==1000
    name='cff_collision_equivalence.json' if cff else 'real_collision_equivalence.json'
    (ROOT/'pilot'/name).write_text(json.dumps({'cases':1000,'passed':True,'official_seconds':official_seconds,'optimized_seconds':fast_seconds,'wall_seconds':time.monotonic()-started,'official_sha256':sha(SOURCE),'optimized_sha256':sha(ROOT/'code/collision_metric.py'),'records':records},indent=2))
