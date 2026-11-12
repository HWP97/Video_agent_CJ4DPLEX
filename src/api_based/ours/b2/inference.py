"""selection236 MAE로 선정한 기존 observation residual Ridge 실행기."""
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
from bundle_paths import RESULTS, artifact_path

METHOD = 'observation_residual_lambda1'

def predictions(sample_id=None):
    frozen=baseline.read(ROOT/'data/b2/ridge/frozen_model.json')
    selection=baseline.read(Path(__file__).with_name('selection236.json'))
    if frozen['selected']!=METHOD or selection['method']!=METHOD:
        raise baseline.InvalidCache('frozen_selection_method_mismatch')
    rows=baseline.predictions('final')
    chosen=[row for row in rows if sample_id is None or row['sample_id']==sample_id]
    if not chosen:
        raise baseline.InvalidCache('unknown_sample_id')
    return chosen

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
    result.update(method=METHOD,selection_rule='selection236_MAE',independent_validation=False)
    return result

def verify():
    result=baseline.verify()
    selected=baseline.read(Path(__file__).with_name('selection236.json'))
    if selected['method']!=METHOD:
        raise baseline.InvalidCache('selection_identity_mismatch')
    return {**result,'method':METHOD,'selection_rule':'selection236_MAE'}

def main():
    parser=argparse.ArgumentParser(description='selection236으로 선택한 동결 Ridge; 신규 API/학습 없음')
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
            result={'method':METHOD,'stage':'ours','selection_rule':'selection236_MAE',
                    'independent_validation':False,'inputs':len(baseline.read(ROOT/'data/b2/ridge/inputs.json')),'paid_calls':0,'training':False}
        case 'verify':
            result=verify()
        case 'replay' | 'predict':
            result={'method':METHOD,'predictions':predictions(args.sample_id),'paid_calls':0,'feature_source':'cached_Sol_and_observed_local_VA','GT_access':False}
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
