# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: 기존 b1 model/env/bin/python -B score.py
"""동결된 예측만 별도 프로세스에서 전수 채점·독립 집계한다."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from prepare import OUT, ROOT, SOURCE
from followup_common import sha,save,guard,metric
from head import current_metric,features


def main() -> None:
    guard(OUT,False)
    torch.set_num_threads(2)
    lock=json.loads((OUT/'selection_lock.json').read_text())
    inference=json.loads((OUT/'inference_receipt.json').read_text())
    assert inference['selection_lock_sha256']==sha(OUT/'selection_lock.json')
    assert all(sha(OUT/p)==h for p,h in inference['outputs'].items())
    assert not lock['outer_labels_read'] and not inference['GT_read'] and inference['negative_GT_open_blocked']
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    assert inference['sample_ids']==[r['sample_id'] for r in manifest]
    labels={}
    for name in ('train','dev','eval'):
        labels.update({r['sample_id']:r for r in json.loads((SOURCE/f'labels_{name}.json').read_text())})
    y=np.array([labels[r['sample_id']]['future'] for r in manifest],np.float64)
    c=np.array([labels[r['sample_id']]['current'] for r in manifest],np.float64)
    splits=np.array([r['split'] for r in manifest])
    with np.load(SOURCE/'inputs.npz') as saved:
        rec=saved['recognition']
    smooth=(rec[:,-3:]*np.array([.2,.3,.5])[None,:,None]).sum(1)
    predictions={'baseline_hold':(rec[:,-1],np.repeat(rec[:,-1:, :],4,1)),
        'previous_smoothing':(smooth,np.repeat(smooth[:,None],4,1))}
    with np.load(SOURCE/'predictions/direct_gru_h8_contrast0.npz') as old:
        predictions['previous_GRU']=(None,old['prediction'])
    for method in lock['models']:
        with np.load(OUT/f'{method}_prediction.npz') as data:
            predictions[method]=(data['current'],data['prediction'])
    table,group_rows,independent=[],[],[]
    for method,(cur,p) in predictions.items():
        assert p.shape==y.shape and np.isfinite(p).all()
        for subset in ('train','dev','official_test','additional'):
            mask=splits==subset
            current_result={'current_MAE':'','current_CCC':''}
            if cur is not None:
                a,b=current_metric(cur[mask],c[mask])
                current_result={'current_MAE':a,'current_CCC':b}
            result={'method':method,'split':subset,'n':int(mask.sum()),'seed':20261111 if method.startswith('B2-D') else 'historical','epoch':'analytic1' if method.startswith('B2-D') else '',**current_result,**metric(p[mask],y[mask])}
            table.append(result)
            flat=p[mask].clip(-1,1).reshape(-1,8).astype(np.float64)
            truth=y[mask].reshape(-1,8)
            ccc=[]
            for j in range(8):
                a,b=flat[:,j],truth[:,j]
                n=len(a)
                covariance=(np.dot(a,b)-a.sum()*b.sum()/n)/n
                denominator=(np.dot(a,a)+np.dot(b,b))/n-2*a.mean()*b.mean()
                ccc.append(2*covariance/max(denominator,1e-15))
            difference=max(abs(np.abs(flat-truth).sum()/flat.size-result['MAE']),abs(np.mean(ccc)-result['CCC']))
            assert difference<1e-10
            independent.append(float(difference))
            groups=np.array([r['group'] for r in manifest])
            for group in sorted(set(groups[mask])):
                gm=mask&(groups==group)
                cm={'current_MAE':'','current_CCC':''}
                if cur is not None:
                    a,b=current_metric(cur[gm],c[gm]);cm={'current_MAE':a,'current_CCC':b}
                group_rows.append({'method':method,'split':subset,'group':group,'n':int(gm.sum()),**cm,**metric(p[gm],y[gm])})
    previous=json.loads((SOURCE/'scores.json').read_text())
    baseline_diffs=[]
    for old in previous:
        if old['method'] in ('hold','direct_gru_h8_contrast0'):
            method='baseline_hold' if old['method']=='hold' else 'previous_GRU'
            new=next(r for r in table if r['method']==method and r['split']==old['split'])
            baseline_diffs.append(max(abs(new['MAE']-old['MAE']),abs(new['CCC']-old['CCC_mean8'])))
    assert max(baseline_diffs)<1e-7
    replay=[]
    transform_checks=[]
    split=json.loads((OUT/'split.json').read_text())
    for phase in ('inner','full'):
        train=np.array(split['fit' if phase=='inner' else 'train_full'])
        x,hold=features(train,OUT/phase)
        with np.load(SOURCE/'inputs.npz') as raw:
            visual=np.concatenate((raw['x'][:,-1],raw['x'].mean(1)),1).astype(np.float64)
            numeric=raw['numeric']
        with np.load(OUT/phase/'transform.npz') as transform:
            assert np.array_equal(train,transform['fit_ids'])
            mean=visual[train].sum(0)/len(train)
            scale=np.maximum(np.sqrt(((visual[train]-mean)**2).sum(0)/len(train)),1e-4)
            assert np.allclose(mean,transform['mean'],atol=1e-12,rtol=0)
            assert np.allclose(scale,transform['scale'],atol=1e-12,rtol=0)
            orthogonality=float(np.abs(transform['basis'].T@transform['basis']-np.eye(16)).max())
            assert orthogonality<1e-5
            joined=np.concatenate((((visual-mean)/scale)@transform['basis'],numeric),1)
            fm=joined[train].sum(0)/len(train)
            fs=np.maximum(np.sqrt(((joined[train]-fm)**2).sum(0)/len(train)),1e-4)
            assert np.allclose(fm,transform['fm'],atol=1e-10,rtol=0)
            assert np.allclose(fs,transform['fs'],atol=1e-10,rtol=0)
            transform_checks.append({'phase':phase,'fit_n':len(train),'fit_ids_exact':True,'PCA_mean_scale_maxdiff':float(max(np.abs(mean-transform['mean']).max(),np.abs(scale-transform['scale']).max())),
                'feature_mean_scale_maxdiff':float(max(np.abs(fm-transform['fm']).max(),np.abs(fs-transform['fs']).max())),'basis_orthogonality_maxdiff':orthogonality})
        affine=np.column_stack((hold,np.ones(len(hold))))
        for method in lock['models']:
            parent=lock['D04_parent'] if method=='B2-D04' else method
            design=affine if parent=='B2-D01' else x
            with np.load(OUT/phase/f'{parent}.npz') as data:
                current=hold+design@data['coefficient']
            prediction=np.repeat(current[:,None],4,1)
            if method=='B2-D04':
                with np.load(OUT/phase/'B2-D04.npz') as data:
                    prediction=prediction+(x@data['coefficient']).reshape(-1,4,2)
            path=OUT/'inner/predictions.npz' if phase=='inner' else OUT/f'{method}_prediction.npz'
            with np.load(path) as saved:
                reference=saved[method if phase=='inner' else 'prediction']
            difference=float(np.abs(reference-prediction).max())
            assert difference<1e-12
            replay.append({'phase':phase,'method':method,'n':len(manifest),'maxdiff':difference})
    for filename,rows in [('scores.csv',table),('scores_by_group.csv',group_rows)]:
        with (OUT/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(OUT/'scores.json',table)
    registration=json.loads((OUT/'registration.json').read_text())
    preservation={p:sha(Path(p))==h for p,h in registration['protected'].items()}
    assert all(preservation.values())
    save(OUT/'verification.json',{'samples':len(manifest),'split_counts':{s:int((splits==s).sum()) for s in set(splits)},'score_rows':len(table),'group_rows':len(group_rows),
        'missing_samples':0,'failed_samples':0,'independent_aggregation_maxdiff':max(independent),'historical_score_maxdiff':max(baseline_diffs),'replay':replay,'protected':preservation,
        'selection':lock['selected'],'transform_checks':transform_checks,'negative_GT_open_blocked':inference['negative_GT_open_blocked'],'source_sha256':sha(Path(__file__))})
    print(json.dumps([r for r in table if r['split']=='official_test']),flush=True)


if __name__=='__main__':
    main()
