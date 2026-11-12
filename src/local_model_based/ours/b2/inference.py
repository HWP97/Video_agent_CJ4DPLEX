"""Sol 입력 없이 기존 localpool Ridge를 추론한다. 이미 본 final의 사후 선택이다."""
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from typing_extensions import assert_never

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'baseline/b2'))
import runtime as baseline
from bundle_paths import REPORTS, RESULTS, artifact_path

DATA = ROOT/'data/b2/ours'
METHOD = 'localpool_ridge'

def pool_predict(features, numeric):
    with np.load(DATA/'projection.npz') as projection, np.load(DATA/'pool.npz') as model:
        visual=(((features-projection['p0'])/projection['p1'])@projection['p2']/np.sqrt(32.)).astype(np.float32)
        normalized=((numeric-projection['p3'])/projection['p4']).astype(np.float32)
        pooled=np.concatenate((visual.mean(1),visual[:,-1],normalized),1)
        values=((pooled-model['mean'])@model['coef']+model['center']).reshape(-1,4,2)
    if not np.isfinite(values).all():
        raise baseline.InvalidCache('nonfinite_head_prediction')
    return np.clip(values,-1,1)

def predictions(sample_id=None):
    ids=baseline.read(DATA/'sample_ids.json')
    indices=[i for i,sid in enumerate(ids) if sample_id is None or sid==sample_id]
    if not indices:
        raise baseline.InvalidCache('unknown_sample_id')
    manifest={row['sample_id']:row for row in baseline.read(ROOT/'data/b2/inputs/forecast.json')}
    for i in indices:
        row=manifest[ids[i]]
        if len(row['frames'])!=15 or any(frame['pts_s']>row['cutoff_s'] for frame in row['frames']):
            raise baseline.InvalidCache('future_or_missing_observation')
    with np.load(DATA/'observations.npz') as values:
        output=pool_predict(values['x'][indices],values['numeric'][indices])
    return [{'sample_id':ids[i],'prediction':prediction.tolist()} for i,prediction in zip(indices,output)]

def scores(rows):
    result=baseline.scores(rows)
    truth={(r['sample_id'],r['horizon_s']):r['VA'] for r in baseline.read(ROOT/'data/b2/ground_truth/forecast.json')}
    splits={r['sample_id']:r['split'] for r in baseline.read(ROOT/'data/b2/shared_split.json')}
    for scope,chosen in [('selection236',{'selection'}),('old_holdout657',{'selection','final'})]:
        subset=[r for r in rows if splits.get(r['sample_id']) in chosen]
        p=np.array([r['prediction'] for r in subset]).reshape(len(subset),8)
        y=np.array([[truth[r['sample_id'],h] for h in baseline.HORIZONS] for r in subset]).reshape(len(subset),8)
        covariance=np.mean((p-p.mean(0))*(y-y.mean(0)),axis=0)
        ccc=2*covariance/(p.var(0)+y.var(0)+(p.mean(0)-y.mean(0))**2)
        result['scores'][scope]={'n':len(subset),'MAE':float(np.abs(p-y).mean()),'CCC':float(ccc.mean())}
    result.update(method=METHOD,selection_rule='posthoc_seen_final_MAE',independent_validation=False)
    return result

def verify():
    result=baseline.verify()
    records=baseline.read(REPORTS/'b2/ours_assets.json')
    for row in records:
        path=artifact_path(row['destination'])
        if path.is_symlink() or path.stat().st_nlink!=1 or path.stat().st_size!=row['bytes'] or baseline.digest(path)!=row['sha256']:
            raise baseline.InvalidCache('ours_asset_hash_mismatch:'+row['destination'])
    return {**result,'ours_verified_files':len(records),'ours_verified_bytes':sum(r['bytes'] for r in records)}

def main():
    parser=argparse.ArgumentParser(description='기존 meanpool head 추론 전용; 이미 본 final MAE 기준 사후 선택')
    parser.add_argument('action',choices=['status','replay','predict','evaluate','verify'])
    parser.add_argument('--sample-id',help='저장 관측 특징의 단일 표본 ID; 생략하면 전체')
    parser.add_argument('--output',type=Path,help='results/b2/새이름 또는 해당 results의 절대경로')
    args=parser.parse_args()
    if args.action in ('predict','replay'):
        def reject_gt(event,arguments):
            if event=='open' and isinstance(arguments[0],(str,bytes)):
                value=str(arguments[0])
                if '/ground_truth/' in value or '/labels_' in value or '/b2/reference/' in value:
                    raise PermissionError('future_GT_forbidden_during_inference')
        sys.addaudithook(reject_gt)
    match args.action:
        case 'status':
            result={'method':METHOD,'stage':'ours','selection_rule':'posthoc_seen_final_MAE',
                    'independent_validation':False,'inputs':len(baseline.read(DATA/'sample_ids.json')),'paid_calls':0,'training':False}
        case 'verify':
            result=verify()
        case 'replay' | 'predict':
            result={'method':METHOD,'predictions':predictions(args.sample_id),'paid_calls':0,'feature_source':'stored_observations','GT_access':False}
        case 'evaluate':
            if args.sample_id is not None:
                parser.error('evaluate는 전체 분모를 보존하므로 --sample-id를 받지 않습니다')
            result=scores(predictions())
        case unreachable:
            assert_never(unreachable)
    destination=artifact_path(args.output) if args.output else RESULTS/'b2'/f'{args.action}-ours-{uuid4().hex}'
    destination=destination.resolve()
    if not destination.is_relative_to(RESULTS/'b2'):
        raise baseline.InvalidCache('output_must_be_inside_family_results_b2')
    destination.mkdir(parents=True,exist_ok=False)
    path=destination/'result.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'artifact':str(path),'action':args.action,**({} if 'predictions' in result else result)}))
