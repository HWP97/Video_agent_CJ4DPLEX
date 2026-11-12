"""19파일 전체 입력의 SHA·관측길이·시간·회전·mask·그룹 경계를 검사한다."""
import json
from pathlib import Path
import time
import numpy as np
from adapter_io import ROOT,sha

plan=json.loads((ROOT/'runplan.json').read_text());started=time.monotonic();seen=set();groups={};records=[];maximum=0.
for group,files in plan['groups'].items():
    group_ids=set()
    for row in files:
        name=Path(row['file']).stem;receipt=json.loads((ROOT/'data/receipts'/f'{name}.json').read_text());assert receipt['signature']['runplan']==sha(ROOT/'runplan.json')
        for filename,digest in receipt['outputs'].items():assert sha(ROOT/filename)==digest
        index=json.loads((ROOT/'data/indices'/f'{name}_index.json').read_text());n=len(index);assert n==row['scenes']
        ids={r['sample_id'] for r in index};assert len(ids)==n and not seen&ids;seen|=ids;group_ids|=ids
        assert all(len(r['observed_frames'])==9 and len(r['future_frames'])==12 and not set(r['observed_frames'])&set(r['future_frames']) and r['fps'] in (2,2.5) for r in index)
        with np.load(ROOT/'data/inputs'/f'{name}_inputs.npz') as values:
            assert set(values.files)=={'ego','neighbors','mask','rotation','scale','origin','kalman','cv'}
            assert all(np.isfinite(values[k]).all() for k in values.files)
            ego=values['ego'];mask=values['mask'];rotation=values['rotation'];scale=values['scale']
            assert ego.shape==(n,9,5) and values['neighbors'].shape==(n,8,9) and mask.shape==(n,8)
            assert np.all(np.isin(mask,[0,1])) and np.all(values['neighbors'][mask==0]==0) and np.all(scale>=1)
            assert np.max(np.abs(ego[:,:,4]-(np.arange(9)-8)/12))<1e-12 and np.all(ego[:,-1,:2]==0)
            assert np.max(np.abs(np.matmul(rotation,rotation.transpose(0,2,1))-np.eye(2)))<1e-12
            relative=np.matmul(ego[:,:,:2],rotation.transpose(0,2,1))*scale[:,None,None]
        aux=json.loads((ROOT/'auxiliary_receipts'/f'{name}.json').read_text());path=ROOT/'observed_sequences'/f'{name}.npz';assert sha(path)==aux['outputs'][str(path.relative_to(ROOT))]
        with np.load(path) as values:
            assert values['neighbors'].shape==(n,49,9,2) and np.isfinite(values['neighbors']).all() and np.all(np.isin(values['mask'],[0,1]))
            difference=float(np.abs(relative-values['ego']).max());assert difference<1e-4;maximum=max(maximum,difference)
        records.append({'file':name,'group':group,'scenes':n,'ego_reconstruction_float32_maxdiff':difference});print(json.dumps(records[-1]),flush=True)
    groups[group]=group_ids
assert len(seen)==240890 and len(records)==19
for fold in plan['folds']:
    train,dev,evaluation=(groups[fold[k]] for k in ('train','development','evaluation'))
    assert not train&dev and not train&evaluation and not dev&evaluation and len(train|dev|evaluation)==240890
result={'passed':True,'files':19,'scenes':240890,'group_n':{k:len(v) for k,v in groups.items()},'sample_id_intersection_between_roles':0,'physical_recording_identity':'not independently established; this is metadata-group separation','max_float32_observed_reconstruction_error':maximum,'seconds':time.monotonic()-started,'records':records}
(ROOT/'pilot/all_input_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
