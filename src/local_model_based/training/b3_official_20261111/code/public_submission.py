"""검증된 등간격 public frame에만 공식 writer로 단일 예측을 직렬화한다."""
import json
import importlib
from pathlib import Path
import sys
from types import ModuleType
import zipfile
import numpy as np
from adapter_io import ROOT,sha

TOOLS=Path('/mnt/RG2/my_works/experiments/benchmarks/2026-11-06/001_frozen_full/sources/trajnetplusplustools/code')
package=ModuleType('pinned_submission_tools');package.__path__=[str(TOOLS/'trajnetplusplustools')];sys.modules[package.__name__]=package
data=importlib.import_module(package.__name__+'.data');writers=importlib.import_module(package.__name__+'.writers')


def write_submission(name: str, values: np.ndarray) -> dict:
    index=json.loads((ROOT/'public/index.json').read_text());assert values.shape==(3146,12,2) and np.isfinite(values).all()
    scenes={}
    with zipfile.ZipFile(ROOT/'provenance/official_v4_test.zip') as z:
        for filename in z.namelist():
            if not filename.startswith('test/real_data/') or not filename.endswith('.ndjson'):continue
            for line in z.read(filename).splitlines():
                row=json.loads(line)
                if 'scene' in row:scenes[(Path(filename).stem,row['scene']['id'])]=row['scene']
    directory=ROOT/'public/submissions'/name;directory.mkdir(parents=True,exist_ok=True);counts={};max_rounding=0.
    for dataset in sorted({row['dataset'] for row in index}):
        ids=[i for i,row in enumerate(index) if row['dataset']==dataset];out=directory/(dataset+'.ndjson')
        with out.open('w') as stream:
            for i in ids:
                row=index[i];scene=scenes[dataset,row['scene_id']];frames=row['observed_frames'];step=frames[1]-frames[0]
                assert np.all(np.diff(frames)==step) and step>0
                assert scene['s']==frames[0] and scene['e']==frames[-1]+12*step and scene['p']==row['primary_id']
                scene_row=data.SceneRow(scene['id'],scene['p'],scene['s'],scene['e'],scene['fps'],scene['tag']);stream.write(writers.trajnet(scene_row)+'\n')
                for j,(x,y) in enumerate(values[i]):
                    track=data.TrackRow(frames[-1]+(j+1)*step,row['primary_id'],float(x),float(y),0,row['scene_id']);stream.write(writers.trajnet(track)+'\n')
        parsed=[json.loads(line) for line in out.read_text().splitlines()];assert len(parsed)==13*len(ids)
        for offset,i in enumerate(ids):
            row=index[i];block=parsed[offset*13:(offset+1)*13];assert block[0]['scene']['id']==row['scene_id']
            tracks=[r['track'] for r in block[1:]];assert all(r['p']==row['primary_id'] and r['scene_id']==row['scene_id'] and r['prediction_number']==0 for r in tracks)
            step=row['observed_frames'][1]-row['observed_frames'][0];assert [r['f'] for r in tracks]==[row['observed_frames'][-1]+(j+1)*step for j in range(12)]
            rounded=np.array([[r['x'],r['y']] for r in tracks]);assert np.array_equal(rounded,np.array([[round(float(x),2),round(float(y),2)] for x,y in values[i]]))
            max_rounding=max(max_rounding,float(np.abs(rounded-values[i]).max()))
        counts[dataset]={'scenes':len(ids),'prediction_rows':12*len(ids),'sha256':sha(out)}
    result={'method':name,'requested':3146,'provided':sum(r['scenes'] for r in counts.values()),'missing':0,'files':counts,'writer_sha256':sha(TOOLS/'trajnetplusplustools/writers.py'),'official_rounding_decimal_places':2,'max_rounding_difference':max_rounding,'frame_schedule':'all public observed differences regular and predicted endpoint equals scene.e; nonuniform frame input rejected','future_GT_score':None,'uploaded':False}
    (directory/'receipt.json').write_text(json.dumps(result,indent=2));return result
