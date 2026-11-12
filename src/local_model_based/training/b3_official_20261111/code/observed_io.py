"""Re/social용 실제 관측좌표9시점 cache만 읽는다. 미래 GT는 포함하지 않는다."""
from pathlib import Path
import json
import numpy as np
from adapter_io import ROOT,sha


def load_observed_role(fold,role,steps=9,count=16,include_ego=False):
    assert role in ('train','dev','eval') and steps in (8,9) and 1<=count<=49
    plan=json.loads((ROOT/'runplan.json').read_text());group=plan['folds'][fold-1][{'train':'train','dev':'development','eval':'evaluation'}[role]]
    egos=[];neighbors=[];masks=[]
    for row in plan['groups'][group]:
        name=Path(row['file']).stem;path=ROOT/'observed_sequences'/f'{name}.npz';receipt=json.loads((ROOT/'auxiliary_receipts'/f'{name}.json').read_text())
        assert sha(path)==receipt['outputs'][str(path.relative_to(ROOT))]
        with np.load(path) as values:
            ego=values['ego'][:,-steps:];n=values['neighbors'][:,:count,-steps:];mask=values['mask'][:,:count]
            if include_ego:
                ids=values['neighbor_ids'][:,:count];primary=values['primary_ids'];zero=np.all(n[:,:,-1]==0,axis=-1)&mask.astype(bool)
                position=np.sum(zero&(ids<primary[:,None]),axis=1)
                output=np.concatenate((ego[:,None],n[:,:count-1]),axis=1);valid=np.concatenate((np.ones((len(ego),1),np.float32),mask[:,:count-1]),axis=1)
                for i in np.flatnonzero(position):
                    p=int(position[i]);output[i]=np.concatenate((n[i,:p],ego[i,None],n[i,p:count-1]));valid[i]=np.r_[mask[i,:p],1,mask[i,p:count-1]]
                n,mask=output,valid
            egos.append(ego);neighbors.append(n);masks.append(mask)
    return tuple(np.concatenate(values) for values in (egos,neighbors,masks))
