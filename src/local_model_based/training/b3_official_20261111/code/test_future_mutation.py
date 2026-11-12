"""실제 ZIP 미래좌표를 바꿔 production parser의 관측 불변성을 검사한다."""
import json
from pathlib import Path
import zipfile
import numpy as np
import prepare_pilot
from fast_features import tokenize

ROOT=Path(__file__).resolve().parents[1]


def main():
    original_zip=zipfile.ZipFile
    outcomes=[]
    for name in ('biwi_hotel','wildtrack'):
        member='train/real_data/'+name+'.ndjson'
        with original_zip(ROOT/'provenance/official_v4_train.zip') as z:
            rows=[json.loads(line) for line in z.read(member).splitlines() if line.strip()]
        scene=min((r['scene'] for r in rows if 'scene' in r),key=lambda s:s['id'])
        primary=sorted((r['track'] for r in rows if 'track' in r and r['track']['p']==scene['p'] and scene['s']<=r['track']['f']<=scene['e']),key=lambda t:t['f'])
        cutoff=primary[8]['f'];changed=0
        for r in rows:
            if 'track' in r and r['track']['f']>cutoff:
                r['track']['x']=1e9+r['track']['p'];r['track']['y']=-1e9-r['track']['f'];changed+=1
        destination=ROOT/'pilot'/f'mutant_{name}.zip'
        with original_zip(destination,'w') as z:z.writestr(member,'\n'.join(json.dumps(r) for r in rows))
        prepare_pilot.convert(name,1,tokenize,'pilot/mutation_reference/'+name)
        def redirected_zip(path,*args,**kwargs):
            if isinstance(path,(str,Path)) and Path(path)==ROOT/'provenance/official_v4_train.zip':
                return original_zip(destination,*args,**kwargs)
            return original_zip(path,*args,**kwargs)
        prepare_pilot.zipfile.ZipFile=redirected_zip
        try:prepare_pilot.convert(name,1,tokenize,'pilot/mutation_changed/'+name)
        finally:prepare_pilot.zipfile.ZipFile=original_zip
        before=ROOT/'pilot/mutation_reference'/name;after=ROOT/'pilot/mutation_changed'/name
        with np.load(before/(name+'_inputs.npz')) as a,np.load(after/(name+'_inputs.npz')) as b:
            checks={k:bool(np.array_equal(a[k],b[k])) for k in a.files};assert all(checks.values())
        a=np.load(before/(name+'_labels.npz'))['xy'];b=np.load(after/(name+'_labels.npz'))['xy'];assert not np.array_equal(a,b)
        outcomes.append({'dataset':name,'primary_id':scene['p'],'cutoff_frame':cutoff,'changed_future_primary_and_neighbor_records':changed,'all_input_fields_bitwise_equal':checks,'training_targets_changed':True})
    (ROOT/'pilot/future_mutation_receipt.json').write_text(json.dumps(outcomes,indent=2));print(json.dumps(outcomes),flush=True)


if __name__=='__main__':main()
