"""역할별 저장 예측의 독립 ADE/FDE와 공식 GT 충돌 범위를 채점한다."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import time
import numpy as np
from adapter_io import ROOT,sha
from collision_metric import collision_segments
from freeze_predictions import freeze


def main():
    fold=int(sys.argv[1]);role=sys.argv[2];assert role in ('dev','eval')
    if role=='eval':
        frozen=ROOT/'all_fold_selections_frozen.json';assert frozen.exists(),'eval_scores_locked_until_all_fold_selections'
        lock=json.loads(frozen.read_text());assert lock['runplan']==sha(ROOT/'runplan.json') and not lock['eval_scores_used']
        assert lock['candidate_catalog']==sha(ROOT/'ours_candidate_catalog.json') and lock['attention_transfer_amendment']==sha(ROOT/'attention_transfer_amendment.json')
        assert set(lock['folds'])=={'fold1.json','fold2.json','fold3.json'}
        for name,digest in lock['folds'].items():assert sha(ROOT/'selections'/name)==digest
    plan=json.loads((ROOT/'runplan.json').read_text());group=plan['folds'][fold-1][{'dev':'development','eval':'evaluation'}[role]]
    paths=freeze(fold,role);assert paths
    arrays={name:np.load(path,mmap_mode='r') for name,path in paths.items()};names=list(arrays)
    total=sum(r['scenes'] for r in plan['groups'][group]);assert all(a.shape==(total,12,2) for a in arrays.values())
    base=ROOT/'scores'/f'fold{fold}'/role;base.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'prediction_manifests'/f'fold{fold}_{role}.json'
    signature={'predictions':{name:sha(path) for name,path in paths.items()},'source':sha(__file__),'metric_contract':sha(ROOT/'metric_amendment.json'),'collision_source':sha(ROOT/'code/collision_metric.py'),'runplan':sha(ROOT/'runplan.json'),'prediction_manifest_sha256':sha(manifest)}
    version=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
    result=base/'runs'/version;result.mkdir(parents=True,exist_ok=True)
    if (result/'receipt.json').exists():
        previous=json.loads((result/'receipt.json').read_text())
        for row in previous['files']:
            assert sha(ROOT/row['artifact'])==row['score_sha256']
            for folder,suffix,key in [('data/targets','_labels.npz','GT_sha256'),('data/indices','_index.json','index_sha256'),('evaluation_only','.npz','evaluation_cache_sha256')]:assert sha(ROOT/folder/(row['file']+suffix))==row[key]
        assert sha(result/'summary.json')==previous['summary_sha256']
        for name in ('summary.json','receipt.json'):shutil.copy2(result/name,base/name)
        print(json.dumps({'cached_complete_score':str(result),'scenes':total,'methods':len(names)}));return
    if (base/'receipt.json').exists() and 'version' not in json.loads((base/'receipt.json').read_text()):
        legacy=base/'runs/initial_unversioned';legacy.mkdir(exist_ok=True)
        for old in base.iterdir():
            if old.is_file():shutil.copy2(old,legacy/old.name)
    prior=json.loads((base/'receipt.json').read_text()) if (base/'receipt.json').exists() else {}
    reusable=prior.get('fold')==fold and prior.get('role')==role and all(prior.get(key)==signature[key] for key in ('source','metric_contract','collision_source','runplan'))
    old_files={row['file']:row for row in prior['files']} if reusable else {}
    archive=ROOT/'scores/source_archive';archive.mkdir(exist_ok=True);(archive/(signature['source']+'.py')).write_bytes(Path(__file__).read_bytes())
    shutil.copy2(manifest,result/'prediction_manifest.json')
    accumulator=np.zeros((len(names),5));offset=0;started=time.monotonic();file_receipts=[]
    for item in plan['groups'][group]:
        stem=Path(item['file']).stem;index=json.loads((ROOT/'data/indices'/f'{stem}_index.json').read_text());n=len(index)
        receipt=json.loads((ROOT/'auxiliary_receipts'/f'{stem}.json').read_text());rawpath=ROOT/'evaluation_only'/f'{stem}.npz';assert sha(rawpath)==receipt['outputs'][str(rawpath.relative_to(ROOT))]
        raw=np.load(rawpath);tracks=raw['tracks'];assert np.isfinite(tracks).all();scenes={int(r[0]):int(r[1]) for r in raw['scenes']}
        order=np.argsort(tracks[:,0],kind='stable');tracks=tracks[order];frames,first,counts=np.unique(tracks[:,0],return_index=True,return_counts=True);by_frame={int(f):tracks[start:start+count] for f,start,count in zip(frames,first,counts)}
        pred=np.stack([a[offset:offset+n] for a in arrays.values()]);assert np.isfinite(pred).all()
        data_receipt=json.loads((ROOT/'data/receipts'/f'{stem}.json').read_text());labelpath=ROOT/'data/targets'/f'{stem}_labels.npz';indexpath=ROOT/'data/indices'/f'{stem}_index.json'
        for source_path in (labelpath,indexpath):assert sha(source_path)==data_receipt['outputs'][str(source_path.relative_to(ROOT))]
        truth=np.load(labelpath)['xy'];error=np.linalg.norm(pred-truth[None],axis=-1);ade=error.mean(-1);fde=error[:,:,-1]
        collision=np.zeros((len(names),n,2),bool)
        reused=[];old=old_files.get(stem)
        if old and old['GT_sha256']==sha(labelpath) and old['index_sha256']==sha(indexpath) and old['evaluation_cache_sha256']==sha(rawpath):
            old_path=ROOT/old['artifact'];assert sha(old_path)==old['score_sha256']
            with np.load(old_path) as cached:
                assert cached['sample_ids'].tolist()==[r['sample_id'] for r in index]
                by_digest={prior['predictions'][str(name)]:i for i,name in enumerate(cached['methods'])}
                for j,name in enumerate(names):
                    previous=by_digest.get(signature['predictions'][name])
                    if previous is not None:
                        assert np.array_equal(ade[j],cached['ADE'][previous]) and np.array_equal(fde[j],cached['FDE'][previous])
                        collision[j]=cached['collision'][previous];reused.append(j)
        fresh=np.array([j for j in range(len(names)) if j not in reused],dtype=int)
        for i,row in enumerate(index if len(fresh) else []):
            primary=scenes[int(row['sample_id'].rsplit('_',1)[1])]
            visible=np.unique(np.concatenate([by_frame[f][:,1] for f in row['observed_frames']]))
            pieces=[]
            for step,f in enumerate(row['future_frames']):
                values=by_frame[f];values=values[values[:,1]!=primary]
                pieces.append(np.column_stack((np.full(len(values),step),values[:,1:])))
            future=np.concatenate(pieces);future=future[np.argsort(future[:,1],kind='stable')]
            paired=future[:-1,1]==future[1:,1];left=future[:-1][paired];right=future[1:][paired]
            assert np.all(left[:,0]<right[:,0])
            all_hit,observed_hit=collision_segments(pred[fresh,i],left[:,0].astype(int),right[:,0].astype(int),left[:,2:],right[:,2:],np.isin(left[:,1],visible))
            collision[fresh,i,0]=all_hit;collision[fresh,i,1]=observed_hit
        np.savez(result/(stem+'.npz'),ADE=ade,FDE=fde,collision=collision,methods=np.array(names),sample_ids=np.array([r['sample_id'] for r in index]))
        accumulator+=np.column_stack((np.full(len(names),n),ade.sum(1),fde.sum(1),collision[:,:,0].sum(1),collision[:,:,1].sum(1)))
        file_receipts.append({'file':stem,'n':n,'score_sha256':sha(result/(stem+'.npz')),'artifact':str((result/(stem+'.npz')).relative_to(ROOT)),'GT_sha256':sha(labelpath),'index_sha256':sha(indexpath),'evaluation_cache_sha256':sha(rawpath),'reused_method_count':len(reused),'reused_score_artifact':old['artifact'] if reused else None});offset+=n
        print(json.dumps({'fold':fold,'role':role,'file':stem,'methods':len(names),'reused_methods':len(reused),'fresh_methods':len(fresh),'processed':offset,'total':total,'seconds':time.monotonic()-started}),flush=True)
    summary=[{'method':name,'requested':total,'provided':int(v[0]),'missing':0,'ADE':v[1]/v[0],'FDE':v[2]/v[0],'ColI_all_GT_percent':100*v[3]/v[0],'ColI_observed_GT_percent':100*v[4]/v[0],'ColII_predicted_neighbors':None,'units':'source coordinate units; calibration not independently verified'} for name,v in zip(names,accumulator)]
    (result/'summary.json').write_text(json.dumps(summary,indent=2))
    (result/'receipt.json').write_text(json.dumps({'fold':fold,'role':role,'version':version,'scenes':total,'methods':len(names),'seconds':time.monotonic()-started,**signature,'prediction_manifest':str((result/'prediction_manifest.json').relative_to(ROOT)),'files':file_receipts,'summary_sha256':sha(result/'summary.json')},indent=2))
    for name in ('summary.json','receipt.json'):shutil.copy2(result/name,base/name)


if __name__=='__main__':main()
