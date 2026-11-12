# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: 기존 b1 model/env/bin/python -B train.py
"""정답은 train만 읽고 이중 내부 검증으로 선택을 동결한다."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import torch
from prepare import OUT, ROOT, SOURCE
from followup_common import sha,save,guard,metric
from head import features,ridge,current_metric


def main() -> None:
    guard(OUT,True)
    torch.set_num_threads(2)
    registration = json.loads((OUT/'registration.json').read_text())
    assert all(sha(Path(p))==h for p,h in registration['protected'].items())
    assert sha(OUT/'DECISIONS.md')==registration['decisions_sha256']
    assert json.loads((OUT/'restored_pixels/input_check.json').read_text())['all_passed']
    sources = {str(p):sha(p) for p in OUT.glob('*.py')}
    save(OUT/'training_protocol_lock.json',{'sources':sources,'registration':sha(OUT/'registration.json'),'input_amendment_sha256':sha(OUT/'INPUT_PROTOCOL_AMENDMENT.md'),
        'input_check_sha256':sha(OUT/'restored_pixels/input_check.json'),'external_labels_read':False})
    rows = json.loads((SOURCE/'manifest.json').read_text())
    split = json.loads((OUT/'split.json').read_text())
    train,fit = np.array(split['train_full']),np.array(split['fit'])
    group = np.array([r['group'] for r in rows])
    labels = {r['sample_id']:r for r in json.loads((SOURCE/'labels_train.json').read_text())}
    current, future = np.zeros((len(rows),2)),np.zeros((len(rows),4,2))
    for i in train:
        current[i],future[i] = labels[rows[i]['sample_id']]['current'],labels[rows[i]['sample_id']]['future']
    x,hold = features(fit,OUT/'inner')
    affine = np.column_stack((hold,np.ones(len(hold))))
    val_rows, predictions, currents = [],{},{}
    models = ['B2-D01','B2-D02','B2-D03']
    coefficients = {}
    for method in models:
        design = affine if method=='B2-D01' else x
        weights = np.ones(len(fit))
        if method=='B2-D03':
            counts = {g:int((group[fit]==g).sum()) for g in set(group[fit])}
            weights = np.array([1/counts[g] for g in group[fit]])
        coef = ridge(design[fit],current[fit]-hold[fit],weights)
        coefficients[method] = coef
        c = hold+design@coef
        currents[method] = c
        predictions[method] = np.repeat(c[:,None],4,1)
        np.savez(OUT/'inner'/f'{method}.npz',coefficient=coef,fit_ids=fit)
    currents['baseline_hold'],predictions['baseline_hold'] = hold,np.repeat(hold[:,None],4,1)
    baseline = {}
    for subset in ('inner_group','inner_time'):
        ids = np.array(split[subset])
        a,b = current_metric(hold[ids],current[ids])
        baseline[subset] = {'current_MAE':a,'current_CCC':b,**metric(predictions['baseline_hold'][ids],future[ids])}
    eligible_current=[]
    for method in models:
        values=[]
        for subset in baseline:
            ids=np.array(split[subset])
            a,b=current_metric(currents[method][ids],current[ids])
            values.append((a,b,baseline[subset]))
        if all(a<=base['current_MAE'] and b>=base['current_CCC'] for a,b,base in values):
            eligible_current.append((max(a/base['current_MAE'] for a,b,base in values),method))
    delta_parent = min(eligible_current)[1] if eligible_current else None
    if delta_parent is not None:
        models.append('B2-D04')
        delta = ridge(x[fit],(future[fit]-current[fit,None]).reshape(-1,8),np.ones(len(fit)))
        coefficients['B2-D04'] = delta
        currents['B2-D04'] = currents[delta_parent]
        predictions['B2-D04'] = predictions[delta_parent]+(x@delta).reshape(-1,4,2)
        np.savez(OUT/'inner/B2-D04.npz',coefficient=delta,fit_ids=fit,parent=np.array(delta_parent))
    passing=[]
    for method in ['baseline_hold']+models:
        for subset in baseline:
            ids=np.array(split[subset])
            a,b=current_metric(currents[method][ids],current[ids])
            val_rows.append({'method':method,'subset':subset,'n':len(ids),'seed':20261111,'epoch':'analytic1','current_MAE':a,'current_CCC':b,**metric(predictions[method][ids],future[ids])})
        actual=[r for r in val_rows if r['method']==method]
        if method!='baseline_hold' and all(r['MAE']<=baseline[r['subset']]['MAE'] and r['CCC']>=baseline[r['subset']]['CCC'] for r in actual):
            passing.append((max(r['MAE']/baseline[r['subset']]['MAE'] for r in actual),method))
    selected=min(passing)[1] if passing else 'baseline_hold'
    save(OUT/'selection_lock.json',{'selected':selected,'models':models,'D04_parent':delta_parent,'D04_skipped':delta_parent is None,'validation_scores':val_rows,'fit_ids':fit.tolist(),
        'outer_labels_read':False,'registration_sha256':sha(OUT/'registration.json'),'sources':sources})
    with (OUT/'inner_scores.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(val_rows[0]));writer.writeheader();writer.writerows(val_rows)
    np.savez(OUT/'inner/predictions.npz',**{m:predictions[m] for m in predictions},**{m+'_current':currents[m] for m in currents})
    full_x,full_hold=features(train,OUT/'full')
    full_affine=np.column_stack((full_hold,np.ones(len(full_hold))))
    for method in models[:3]:
        design=full_affine if method=='B2-D01' else full_x
        weights=np.ones(len(train))
        if method=='B2-D03':
            counts={g:int((group[train]==g).sum()) for g in set(group[train])}
            weights=np.array([1/counts[g] for g in group[train]])
        coef=ridge(design[train],current[train]-full_hold[train],weights)
        np.savez(OUT/'full'/f'{method}.npz',coefficient=coef,fit_ids=train)
    if delta_parent is not None:
        delta=ridge(full_x[train],(future[train]-current[train,None]).reshape(-1,8),np.ones(len(train)))
        np.savez(OUT/'full/B2-D04.npz',coefficient=delta,fit_ids=train,parent=np.array(delta_parent))
    outputs={str(p.relative_to(OUT)):sha(p) for p in list((OUT/'inner').glob('*.npz'))+list((OUT/'full').glob('*.npz'))}
    save(OUT/'training_receipt.json',{'source_sha256':sha(Path(__file__)),'selection_lock_sha256':sha(OUT/'selection_lock.json'),'outputs':outputs,'fit_groups':sorted(set(group[fit])),'full_train_groups':sorted(set(group[train])),
        'current_only_fits':6,'delta_fits':2 if delta_parent is not None else 0,'epochs':'analytic1','outer_labels_read':False,'targets_in_predict_signature':False})
    print(json.dumps({'selected':selected,'D04_parent':delta_parent,'inner_scores':val_rows}),flush=True)


if __name__=='__main__':
    main()
