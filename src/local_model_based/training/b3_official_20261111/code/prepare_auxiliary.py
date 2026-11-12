"""파일별 원자료1회 파싱으로 Re 관측 cache와 격리된 평가 cache를 만든다."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import collections
import hashlib
import json
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def task(row):
    name=Path(row['file']).stem;receipt=ROOT/'auxiliary_receipts'/f'{name}.json'
    signature={'ndjson':row['sha256'],'runplan':sha(ROOT/'runplan.json'),'source':sha(Path(__file__))}
    if receipt.exists():
        result=json.loads(receipt.read_text());assert result['signature']==signature
        for path,h in result['outputs'].items():assert sha(ROOT/path)==h
        return result
    started=time.monotonic();scenes={};records=[];by_frame=collections.defaultdict(list)
    with zipfile.ZipFile(ROOT/'provenance/official_v4_train.zip') as z:
        for line in z.open(row['file']):
            if not line.strip():continue
            r=json.loads(line)
            if 'scene' in r:scenes[r['scene']['id']]=r['scene']
            else:
                t=r['track'];by_frame[t['f']].append(len(records));records.append((t['f'],t['p'],t['x'],t['y']))
    tracks=np.array(records,dtype=np.float64);del records
    index=json.loads((ROOT/'data/indices'/f'{name}_index.json').read_text());n=len(index)
    ego=np.zeros((n,9,2),np.float32);neighbors=np.zeros((n,49,9,2),np.float32);mask=np.zeros((n,49),np.float32);ids=np.full((n,49),-1,np.int64);primaries=np.zeros(n,np.int64)
    for i,item in enumerate(index):
        sid=int(item['sample_id'].rsplit('_',1)[1]);scene=scenes[sid];primary=scene['p'];primaries[i]=primary
        groups=collections.defaultdict(list);timing={f:j/scene['fps'] for j,f in enumerate(item['observed_frames'])}
        for f in timing:
            for j in by_frame[f]:groups[int(tracks[j,1])].append(j)
        own=tracks[groups[primary],2:];assert own.shape==(9,2);origin=own[-1];ego[i]=own-origin
        cutoff=8/scene['fps'];dt=1/scene['fps'];grid=np.arange(9)*dt;candidates=[]
        for pedestrian,indices in groups.items():
            if pedestrian==primary or len(indices)<2:continue
            rows=tracks[indices];times=np.array([timing[int(f)] for f in rows[:,0]])
            assert times.max()<=cutoff+1e-8
            if times[-1]<cutoff-dt-1e-8:continue
            candidates.append((float(np.linalg.norm(rows[-1,2:]-origin)),pedestrian,indices,times))
        for j,(_,pedestrian,indices,times) in enumerate(sorted(candidates,key=lambda r:(r[0],r[1]))[:49]):
            rows=tracks[indices];neighbors[i,j]=np.column_stack([np.interp(grid,times,rows[:,axis]) for axis in (2,3)])-origin
            mask[i,j]=1;ids[i,j]=pedestrian
    obs=ROOT/'observed_sequences'/f'{name}.npz';np.savez(obs,ego=ego,neighbors=neighbors,mask=mask,neighbor_ids=ids,primary_ids=primaries)
    raw=ROOT/'evaluation_only'/f'{name}.npz';np.savez(raw,tracks=tracks,scenes=np.array([[s['id'],s['p'],s['s'],s['e'],s['fps']] for s in scenes.values()]))
    result={'signature':signature,'samples':n,'seconds':time.monotonic()-started,'outputs':{str(path.relative_to(ROOT)):sha(path) for path in (obs,raw)},'observation_policy':'only input index observed_frames; all neighbor ranking from observed positions; no future coordinates','evaluation_policy':'all raw tracks isolated from inference'}
    temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(receipt)
    print(json.dumps({'dataset':name,'n':n,'seconds':result['seconds']}),flush=True);return result


if __name__=='__main__':
    for folder in ['observed_sequences','evaluation_only','auxiliary_receipts']:(ROOT/folder).mkdir(exist_ok=True)
    rows=[r for r in json.loads((ROOT/'provenance/full_release_audit.json').read_text())['files'] if r['file'].startswith('train/real_data/')]
    rows.sort(key=lambda r:r['scenes'])
    with ProcessPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(task,rows))
    (ROOT/'auxiliary_complete.json').write_text(json.dumps({'files':len(receipts),'samples':sum(r['samples'] for r in receipts),'API':0},indent=2))
