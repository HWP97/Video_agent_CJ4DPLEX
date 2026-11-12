"""동결된 pilot train/dev 파일의 관측과 정답을 분리한다."""
from pathlib import Path
import bisect
import collections
import hashlib
import json
import resource
import time
import zipfile
import numpy as np
from data import Observation, Track
from features import Tokens, tokenize

ROOT = Path(__file__).resolve().parents[1]


def convert(name, limit, token_function=tokenize, destination='pilot', compare=False):
    start = time.monotonic()
    scenes = []
    by_frame = collections.defaultdict(list)
    primary = collections.defaultdict(list)
    with zipfile.ZipFile(ROOT/'provenance/official_v4_train.zip') as z:
        for line in z.open('train/real_data/'+name+'.ndjson'):
            if not line.strip():
                continue
            row = json.loads(line)
            if 'scene' in row:
                scenes.append(row['scene'])
            else:
                t = row['track']
                by_frame[t['f']].append(t)
                primary[t['p']].append(t)
    for values in primary.values():
        values.sort(key=lambda t: t['f'])
    frames = {p: [t['f'] for t in values] for p, values in primary.items()}
    parse_seconds = time.monotonic()-start
    values = [];truth = [];index = [];maxdiff=0.;reference_seconds=0.;optimized_seconds=0.
    for scene in sorted(scenes, key=lambda s: s['id'])[:limit]:
        ts=primary[scene['p']];fs=frames[scene['p']]
        full=ts[bisect.bisect_left(fs,scene['s']):bisect.bisect_right(fs,scene['e'])]
        assert len(full)==21
        dt=1/scene['fps'];observed=full[:9];timing={t['f']:i*dt for i,t in enumerate(observed)}
        neighbors=collections.defaultdict(list)
        for frame in timing:
            for t in by_frame[frame]:
                neighbors[t['p']].append(t)
        tracks=tuple(Track(p,np.array([timing[t['f']] for t in records]),np.array([[t['x'],t['y']] for t in records])) for p,records in sorted(neighbors.items()))
        xy=np.array([[t['x'],t['y']] for t in full]);assert np.isfinite(xy).all()
        sid=name+'_'+str(scene['id']);obs=Observation(sid,name,scene['p'],tuple(timing),xy[:9],tracks)
        assert all(t.times.max()<=8*dt for t in tracks)
        tick=time.monotonic();value=token_function(obs);optimized_seconds+=time.monotonic()-tick
        if compare:
            tick=time.monotonic();expected=tokenize(obs);reference_seconds+=time.monotonic()-tick
            maxdiff=max(maxdiff,max(float(np.max(np.abs(getattr(value,k)-getattr(expected,k)))) for k in Tokens.__dataclass_fields__))
            assert maxdiff<1e-9,(sid,maxdiff)
        values.append(value);truth.append(xy[9:]);index.append({'sample_id':sid,'dataset':name,'fps':scene['fps'],'observed_frames':list(timing),'future_frames':[t['f'] for t in full[9:]]})
    folder=ROOT/destination;folder.mkdir(exist_ok=True,parents=True);out={k:np.array([getattr(v,k) for v in values]) for k in Tokens.__dataclass_fields__}
    np.savez(folder/(name+'_inputs.npz'),**out)
    np.savez(folder/(name+'_labels.npz'),xy=np.array(truth))
    (folder/(name+'_index.json')).write_text(json.dumps(index))
    receipt={'dataset':name,'samples':len(values),'total_file_scenes':len(scenes),'parse_seconds':parse_seconds,'tokenize_seconds':time.monotonic()-start-parse_seconds,'optimized_token_seconds':optimized_seconds,'reference_token_seconds':reference_seconds,'maxdiff':maxdiff if compare else None,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    plan=json.loads((ROOT/'runplan.json').read_text())
    receipts=[convert(name,count) for name,count in plan['pilot']['files'].items()]
    (ROOT/'pilot/preparation_receipt.json').write_text(json.dumps(receipts,indent=2))
