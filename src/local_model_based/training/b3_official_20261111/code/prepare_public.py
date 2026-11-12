"""정답이 없는 공식 public realtest의 9관측만 별도 cache로 변환한다."""
import bisect
import collections
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from adapter_io import ROOT,sha
from data import Observation,Track
from features import Tokens
from fast_features import tokenize

directory=ROOT/'public';directory.mkdir(exist_ok=True);started=time.monotonic();tokens=[];index=[];egos=[];neighbor_rows=[];mask_rows=[];sources={}
archive=ROOT/'provenance/official_v4_test.zip'
with zipfile.ZipFile(archive) as z:
    files=sorted(n for n in z.namelist() if n.startswith('test/real_data/') and n.endswith('.ndjson'))
    for name in files:
        raw=z.read(name);sources[name]=__import__('hashlib').sha256(raw).hexdigest();scenes=[];by_frame=collections.defaultdict(list);primary=collections.defaultdict(list)
        for line in raw.splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if 'scene' in row:scenes.append(row['scene'])
            else:
                tr=row['track'];by_frame[tr['f']].append(tr);primary[tr['p']].append(tr)
        for values in primary.values():values.sort(key=lambda r:r['f'])
        frames={p:[r['f'] for r in values] for p,values in primary.items()}
        for scene in sorted(scenes,key=lambda s:s['id']):
            rows=primary[scene['p']];fs=frames[scene['p']];observed=rows[bisect.bisect_left(fs,scene['s']):bisect.bisect_right(fs,scene['e'])];assert len(observed)==9
            dt=1/scene['fps'];timing={r['f']:i*dt for i,r in enumerate(observed)};tracks=collections.defaultdict(list)
            for frame in timing:
                for tr in by_frame[frame]:tracks[tr['p']].append(tr)
            parts=tuple(Track(p,np.array([timing[r['f']] for r in rs]),np.array([[r['x'],r['y']] for r in rs])) for p,rs in sorted(tracks.items()))
            xy=np.array([[r['x'],r['y']] for r in observed]);sid=Path(name).stem+'_'+str(scene['id']);obs=Observation(sid,Path(name).stem,scene['p'],tuple(timing),xy,parts)
            assert all(t.times.max()<=8*dt for t in parts);tokens.append(tokenize(obs));egos.append((xy-xy[-1]).astype(np.float32))
            candidates=[]
            for tr in parts:
                if tr.pedestrian==scene['p'] or len(tr.times)<2 or tr.times[-1]<7*dt-1e-8:continue
                values=np.column_stack([np.interp(np.arange(9)*dt,tr.times,tr.xy[:,axis]) for axis in (0,1)])-xy[-1]
                candidates.append((float(np.linalg.norm(values[-1])),tr.pedestrian,values))
            neighbors=np.zeros((49,9,2),np.float32);mask=np.zeros(49,np.float32)
            for j,(_,_,values) in enumerate(sorted(candidates,key=lambda r:(r[0],r[1]))[:49]):neighbors[j]=values;mask[j]=1
            neighbor_rows.append(neighbors);mask_rows.append(mask)
            index.append({'sample_id':sid,'dataset':Path(name).stem,'scene_id':scene['id'],'primary_id':scene['p'],'fps':scene['fps'],'observed_frames':list(timing),'future_steps':list(range(1,13)),'future_GT_available':False})
assert len(index)==3146 and len({r['sample_id'] for r in index})==3146
np.savez(directory/'inputs.npz',**{key:np.array([getattr(t,key) for t in tokens]) for key in Tokens.__dataclass_fields__})
np.savez(directory/'observed_sequences.npz',ego=np.array(egos),neighbors=np.array(neighbor_rows),mask=np.array(mask_rows))
(directory/'index.json').write_text(json.dumps(index))
result={'samples':len(index),'files':sources,'archive_sha256':sha(archive),'source_sha256':sha(__file__),'runplan':sha(ROOT/'runplan.json'),'outputs':{name:sha(directory/name) for name in ('inputs.npz','observed_sequences.npz','index.json')},'future_GT_read':False,'future_coordinates_present':False,'seconds':time.monotonic()-started}
(directory/'preparation_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
