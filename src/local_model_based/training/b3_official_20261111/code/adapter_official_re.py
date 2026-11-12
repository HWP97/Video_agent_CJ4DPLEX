"""기존 공식 Re 체크포인트와 사전 지정 3변형을 관측 입력으로만 재실행한다."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from adapter_io import ROOT, load_role, sha, install_guard
from observed_io import load_observed_role
from adapter_selector import weights

SOURCE=Path('/mnt/RG2/paper_experiments_20261109/b3')
def protect_sources(event, args) -> None:
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        path=Path(os.fsdecode(args[0])).resolve()
        if writing and not path.is_relative_to(ROOT) and not path.is_relative_to(Path('/tmp')) and not path.is_relative_to(Path('/dev')):raise PermissionError('source_write_forbidden')
sys.addaudithook(protect_sources)
runtime=ROOT/'adapters/official_re/runtime';(runtime/'dataset_configs').mkdir(parents=True,exist_ok=True)
os.chdir(runtime)
sys.path.insert(0,str(SOURCE/'assets/python_deps'))
sys.path.insert(0,str(SOURCE/'sources/Re-main'))
import qpid
from qpid.base import BaseManager
from qpid.dataset import Annotation, AnnotationManager
from resonance.model import ResonanceModel


class CoordinateModel(ResonanceModel):
    @property
    def picker(self) -> Annotation:
        return Annotation('coordinate')


def checkpoint_for(dataset: str) -> str:
    if 'hotel' in dataset:return 'rehotel'
    if 'students' in dataset:return 'reuniv13'
    if 'zara01' in dataset or 'zara1' in dataset:return 'rezara1'
    return 'rezara2'


def main() -> None:
    install_guard('infer',1);torch.set_num_threads(1)
    plan=json.loads((ROOT/'runplan.json').read_text())
    for fold in (2,1,3):
        group=plan['folds'][fold-1]['evaluation']
        if sys.argv[1:] and group not in sys.argv[1:]:continue
        directory=ROOT/'adapters/official_re/groups'/group;directory.mkdir(parents=True,exist_ok=True)
        batch,index,_=load_role(fold,'eval');n=len(index)
        signature={'source':sha(__file__),'observed_io':sha(ROOT/'code/observed_io.py'),'runplan':sha(ROOT/'runplan.json'),'n':n,'metadata_order_sha256':hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'weights':{str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'assets/official_weights').glob('*/*')}}
        receipt=directory/'receipt.json'
        if receipt.exists():
            previous=json.loads(receipt.read_text());assert previous['signature']==signature
            for name,digest in previous['outputs'].items():assert sha(directory/(name+'.npy'))==digest
            continue
        ego,neighbors,mask=load_observed_role(fold,'eval',steps=8,count=49,include_ego=True);neighbors[mask==0]=1e8
        outputs={f'official_re_{variant}_{rule}':np.empty((n,12,2),np.float64) for variant in ('default','no_neighbors','history_reversed') for rule in ('sample0','mean20')}
        started=time.monotonic();records=[]
        for key in ('rehotel','reuniv13','rezara1','rezara2'):
            chosen=np.array([i for i,r in enumerate(index) if checkpoint_for(r['dataset'])==key],dtype=np.int64)
            if not len(chosen):continue
            path=SOURCE/'assets/official_weights'/key;weight=next(path.glob('*.pt'))
            args=qpid.args.Args(['--load',str(path)],is_temporary=True);manager=BaseManager(args);AnnotationManager(manager,'coordinate');net=CoordinateModel(manager).cuda().eval()
            loaded=net.load_state_dict(torch.load(weight,weights_only=True,map_location='cuda'),strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
            for variant in ('default','no_neighbors','history_reversed'):
                tick=time.monotonic()
                for start in range(0,len(chosen),32):
                    ids=chosen[start:start+32];e=ego[ids];ne=neighbors[ids].copy()
                    if variant=='no_neighbors':ne[:]=1e8;ne[:,0]=e
                    if variant=='history_reversed':ne[:,:,:7]=ne[:,:,6::-1]
                    torch.manual_seed(20261109+start);torch.cuda.manual_seed_all(20261109+start)
                    with torch.inference_mode():values=net.implement([torch.as_tensor(e,device='cuda'),torch.as_tensor(ne,device='cuda')],training=False)[0].cpu().numpy()
                    assert values.shape==(len(ids),20,12,2) and np.isfinite(values).all()
                    world=values+batch.origin[ids,None,None]
                    outputs[f'official_re_{variant}_sample0'][ids]=world[:,0]
                    outputs[f'official_re_{variant}_mean20'][ids]=world.mean(1)
                row={'checkpoint':key,'variant':variant,'n':len(chosen),'seconds':time.monotonic()-tick};records.append(row);print(json.dumps({'group':group,**row}),flush=True)
            del net
        observed=np.matmul(batch.ego[:,:,:2],batch.rotation.transpose(0,2,1))*batch.scale[:,None,None]+batch.origin[:,None]
        fps=np.array([row['fps'] for row in index])
        for method in ('constant_velocity','kalman'):
            output=np.empty((n,12,2),np.float64)
            for rate in np.unique(fps):
                ids=np.flatnonzero(fps==rate);output[ids]=np.einsum('nip,ij->njp',observed[ids,-8:],weights(8,12,1/rate,method))
            outputs[method+'_8obs']=output
        hashes={}
        for name,values in outputs.items():
            assert np.isfinite(values).all();path=directory/(name+'.npy');np.save(path,values);hashes[name]=sha(path)
        receipt.write_text(json.dumps({'signature':signature,'outputs':hashes,'seconds':time.monotonic()-started,'records':records,'GT_read':False,'API':0,'GPU_physical':os.environ['CUDA_VISIBLE_DEVICES'],'condition':'8 of 9 observed steps; unchanged pretrained weights; source-group overlap possible; CFF/LCAS/WildTrack use prespecified rezara2 fallback; sample0 and mean20 are single predictions, never oracle minADE','new_fit':False},indent=2))


if __name__=='__main__':main()
