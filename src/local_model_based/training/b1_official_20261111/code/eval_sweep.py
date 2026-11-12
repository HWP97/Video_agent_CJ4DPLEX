# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
# 실행: 기존 AG Python -B code/eval_sweep.py development|test
"""로컬 세부 설정을 같은 bank로 평가하고 개발에서만 선택한다."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from functools import lru_cache
import copy
import csv
import hashlib
import json
import os
import socket
import sys
import time
import numpy as np
import torch
from dataclasses import replace
from sweep_data import HERE,SOURCE
from sweep_predict import METHODS,load,predict
from provenance import freeze,digest,training_signature
from v2_reference import signature as reference_signature,matching_statistics
sys.path.insert(0,'/mnt/RG2/paper_experiments_20261109/b1/legacy')
sys.path.insert(0,'/mnt/RG2/my_works')
from scoring import capture,statistics,merge,metrics,evaluate
from selector import select
from forecasting.ag_vlm.evaluate import JsonlIndex
DATA=Path('/mnt/RG2/HWP_CJ4DPLEX_AI/src/local_model_based/data/b1')
ACTIVE=[False]
REUSE_V2=True


def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo') or (ACTIVE[0] and event=='open'):
        raise PermissionError('Local prediction cannot read files or use network')


@lru_cache(maxsize=4)
def source_index(mode,name):
    cohort='filtered_small' if mode=='sgcls' else 'unfiltered'
    return JsonlIndex(DATA/f'benchmark/protocol/{cohort}/{name}.jsonl','case_id')


def case(task):
    stage,row,freeze_sha=task;torch.set_num_threads(1)
    sid=row['sample_id'];stem=sid.replace(':','__');mode=row['mode']
    destination=HERE/f'results/{stage}_records/{stem}.json'
    saved_prediction=HERE/f'results/{stage}_predictions/{stem}.pt'
    if destination.exists():
        result=json.loads(destination.read_text())
        assert result['freeze_sha256']==freeze_sha,'Case belongs to different frozen inputs'
        if result['status']=='success':
            assert hashlib.sha256(saved_prediction.read_bytes()).hexdigest()==result['prediction_sha256']
        return result
    if stage=='development':
        saved=torch.load(SOURCE/f'assets/compact_train/{mode}_{row["video"]}.pt',weights_only=False)
        graph,official=saved['graph'],saved['official_prediction']
        future=[int(Path(n).stem) for n in row['future_frame_ids']]
    else:
        receipt=DATA/f'impartail/receipts/{stem}.json'
        if not receipt.exists():
            result=dict(row,status='missing_observation_features',methods={},freeze_sha256=freeze_sha);destination.write_text(json.dumps(result));return result
        receipt=json.loads(receipt.read_text())
        graph_path=DATA/'impartail'/receipt['compact_path'];pred_path=DATA/'impartail'/receipt['prediction_path']
        assert hashlib.sha256(graph_path.read_bytes()).hexdigest()==receipt['compact_sha256']
        assert hashlib.sha256(pred_path.read_bytes()).hexdigest()==receipt['prediction_sha256']
        graph=torch.load(graph_path,weights_only=False);official=torch.load(pred_path,weights_only=False)
        future=[int(Path(n).stem) for n in source_index(mode,'cases').get(sid.split(':',1)[1])['future_frame_ids']]
    models=load(mode,'development' if stage=='development' else 'final')
    start=time.monotonic();ACTIVE[0]=True
    try:
        outputs=predict(graph,future,official,models)
    finally:
        ACTIVE[0]=False
    temporary=saved_prediction.with_suffix('.tmp');torch.save(outputs,temporary);temporary.replace(saved_prediction)
    reused=matching_statistics(stem,outputs) if stage=='test' and REUSE_V2 else {}
    truth=json.loads((SOURCE/f'assets/supervision/{mode}_{row["video"]}.json').read_text())['future_GT'] if stage=='development' else source_index(mode,'future_gt').get(sid.split(':',1)[1])['frames']
    values={}
    baseline_frames={} if reused else capture(copy.deepcopy(outputs['impartail_ode']),truth,mode,np.full(len(outputs['impartail_ode']['boxes']),-1))
    for name,output in outputs.items():
        if name in reused:
            values[name]=reused[name]
            continue
        scored=capture(copy.deepcopy(output),truth,mode,np.full(len(output['boxes']),-1))
        if name=='log_rank_residual':
            multipliers=output['post_softmax_multiplier'].reshape(len(future),-1,26).numpy()
            for constraint,frames in baseline_frames.items():
                assert len(frames)==len(future)
                adjusted=[]
                for frame,p in zip(frames,multipliers):
                    count=len(p);relations=frame.entry.relations.copy()
                    assert len(relations)==3*count
                    for group,(first,last) in enumerate(((0,3),(3,9),(9,26))):
                        relations[group*count:(group+1)*count,first:last]*=p[:,first:last]
                    adjusted.append(evaluate(replace(frame.entry,relations=relations),frame.truth,mode,constraint))
                scored[constraint]=adjusted
        if name in ('order_baseline','diversity_baseline','diversity_linear'):
            scored['with']=[evaluate(select(frame.entry,name!='order_baseline'),frame.truth,mode,'with') for frame in scored['with']]
        values[name]={constraint:statistics(frames) for constraint,frames in scored.items()}
    assert values['order_baseline']['no']==values['impartail_ode']['no']
    assert values['diversity_baseline']['no']==values['impartail_ode']['no']
    result=dict(row,status='success',methods=values,seconds=time.monotonic()-start,
                v2_reused_methods=list(reused),v2_raw_tensor_exact_count=len(reused),newly_scored_methods=[name for name in outputs if name not in reused],
                prediction_sha256=hashlib.sha256(saved_prediction.read_bytes()).hexdigest(),future_GT_file_access_blocked=True,freeze_sha256=freeze_sha)
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(result));temporary.replace(destination)
    return result


def initialize():
    torch.set_num_threads(1);sys.addaudithook(guard)


def main() -> None:
    stage=sys.argv[1];assert stage in ('development','test')
    if stage=='test':
        assert (HERE/'protocol/selection.json').exists()
    for folder in (f'{stage}_records',f'{stage}_predictions'):
        (HERE/'results'/folder).mkdir(exist_ok=True)
    rows=[r for r in json.loads((SOURCE/'protocol/train_manifest.json').read_text()) if r['status']=='ready' and r['split']=='development'] if stage=='development' else [json.loads(s) for s in (DATA/'samples.jsonl').read_text().splitlines() if json.loads(s)['future_count']]
    frozen={'stage':stage,'cases':len(rows),'methods':METHODS,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'models':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'assets').glob(('development' if stage=='development' else 'final')+'*.pt')},
            'runplan_sha256':hashlib.sha256((HERE/'runplan.json').read_bytes()).hexdigest()}
    frozen['input_signature']=training_signature(HERE,SOURCE,stage)
    if stage=='test':
        frozen['v2_reference']=reference_signature()
        for path in (HERE/'assets').glob('development_*_linear_none.pt'):
            frozen['models'][path.name]=digest(path)
    frozen['selection_sha256']=digest(HERE/'protocol/selection.json') if stage=='test' else None
    frozen['test_inputs']={str(p):digest(p) for p in [DATA/'samples.jsonl',*sorted((DATA/'benchmark/protocol').glob('*/*.jsonl')),*sorted((DATA/'impartail/receipts').glob('*.json'))]} if stage=='test' else {}
    freeze_sha=freeze(HERE/f'protocol/{stage}_freeze.json',frozen)
    if len(sys.argv)>2 and sys.argv[2]=='--smoke':
        initialize()
        for mode in ('sgcls','predcls'):
            checked=case((stage,next(r for r in rows if r['mode']==mode),freeze_sha))
            print(json.dumps({'sample_id':checked['sample_id'],'reused':checked['v2_raw_tensor_exact_count'],'newly_scored':len(checked['newly_scored_methods'])}),flush=True)
        return
    totals={};counts={};missing=[]
    workers=2 if stage=='development' else int(os.environ.get('B1_EVAL_WORKERS','24'))
    with (HERE/'evidence/executions.jsonl').open('a') as stream:
        stream.write(json.dumps({'stage':stage,'pid':os.getpid(),'workers':workers,'started_unix':time.time(),'freeze_sha256':freeze_sha})+'\n')
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        for i,result in enumerate(pool.map(case,[(stage,r,freeze_sha) for r in rows]),1):
            if result['status']!='success':missing.append(result['sample_id'])
            for name,constraints in result['methods'].items():
                for constraint,stats in constraints.items():
                    key=(result['mode'],result.get('ratio',.5),name,constraint)
                    totals[key]=merge([totals[key],stats]) if key in totals else stats;counts[key]=counts.get(key,0)+1
            if i%25==0:print(json.dumps({'stage':stage,'processed':i,'total':len(rows),'missing':len(missing)}),flush=True)
    table=[dict(mode=key[0],ratio=key[1],method=key[2],constraint=key[3],cases=counts[key],**{f'{metric}@{k}':v for k,values in metrics(stats).items() for metric,v in values.items()}) for key,stats in sorted(totals.items())]
    (HERE/f'results/{stage}_summary.json').write_text(json.dumps(table,indent=2))
    with (HERE/f'results/{stage}_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if stage=='development':
        selection={}
        for mode in ('sgcls','predcls'):
            focus=[r for r in table if r['mode']==mode and r['constraint']=='no']
            baseline=next(r for r in focus if r['method']=='impartail_ode')
            valid=[r for r in focus if r['R@20']>=baseline['R@20'] and r['mR@20']>baseline['mR@20']]
            chosen=max(valid,key=lambda r:(r['mR@20'],r['method'])) if valid else baseline
            selection[mode]={'method':chosen['method'],'adopted':bool(valid),'baseline':baseline,'selected':chosen,'eligible':[r['method'] for r in valid]}
        freeze(HERE/'protocol/selection.json',selection)
    (HERE/f'results/{stage}_receipt.json').write_text(json.dumps({'cases':len(rows),'methods':len(METHODS),'missing':missing,'summary_rows':len(table)},indent=2))


if __name__=='__main__':
    main()
