# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
# 실행: AG_PYTHON -B offline_case.py predict SAMPLE OUTPUT [EXPERIMENT_ROOT] [CODE_ROOT]
# 실행: AG_PYTHON -B offline_case.py score PREDICTION_DIRECTORY OUTPUT [EXPERIMENT_ROOT] [CODE_ROOT]
"""동결 가중치의 단일 관측 추론과 저장 예측 전용 재채점을 분리한다."""
from pathlib import Path
import hashlib
import json
import sys
import torch


def main() -> None:
    action,value,destination=sys.argv[1:4]
    root=Path(sys.argv[4]).resolve() if len(sys.argv)>4 else Path(__file__).resolve().parent
    code=Path(sys.argv[5]).resolve() if len(sys.argv)>5 else root/'code'
    output=Path(destination).resolve()
    assert output.is_relative_to(root),'Outputs must remain in experiment support directory'
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(code))
    import sweep_data
    sweep_data.HERE=root
    import eval_sweep as runtime
    from sweep_predict import load,predict
    frozen=json.loads((root/'protocol/test_freeze.json').read_text())
    assert hashlib.sha256((root/'protocol/selection.json').read_bytes()).hexdigest()==frozen['selection_sha256']
    for name,expected in frozen['models'].items():
        assert hashlib.sha256((root/'assets'/name).read_bytes()).hexdigest()==expected,name
    for name,expected in frozen['sources'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected,name
    runtime.initialize()
    if action=='predict':
        mode,case=value.split(':',1)
        receipt=json.loads((runtime.DATA/f'impartail/receipts/{mode}__{case}.json').read_text())
        graph_path=runtime.DATA/'impartail'/receipt['compact_path']
        original_path=runtime.DATA/'impartail'/receipt['prediction_path']
        assert hashlib.sha256(graph_path.read_bytes()).hexdigest()==receipt['compact_sha256']
        assert hashlib.sha256(original_path.read_bytes()).hexdigest()==receipt['prediction_sha256']
        graph=torch.load(graph_path,weights_only=False);original=torch.load(original_path,weights_only=False)
        future=[int(Path(n).stem) for n in runtime.source_index(mode,'cases').get(case)['future_frame_ids']]
        models=load(mode,'final')
        runtime.ACTIVE[0]=True
        try:
            outputs=predict(graph,future,original,models)
        finally:
            runtime.ACTIVE[0]=False
        path=output/'predictions.pt';torch.save(outputs,path)
        result={'action':action,'sample_id':value,'methods':list(outputs),'future_GT_access_blocked':True,'API_calls':0,
                'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'frozen_selection_sha256':hashlib.sha256((root/'protocol/selection.json').read_bytes()).hexdigest()}
    elif action=='score':
        source=Path(value).resolve();receipt=json.loads((source/'receipt.json').read_text())
        path=source/'predictions.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['prediction_sha256']
        assert hashlib.sha256((root/'protocol/selection.json').read_bytes()).hexdigest()==receipt['frozen_selection_sha256']
        cached=torch.load(path,weights_only=False)
        sample=next(json.loads(line) for line in (runtime.DATA/'samples.jsonl').read_text().splitlines() if json.loads(line)['sample_id']==receipt['sample_id'])
        for folder in ('test_records','test_predictions'):
            (output/'results'/folder).mkdir(parents=True)
        runtime.HERE=output
        runtime.REUSE_V2=False
        runtime.predict=lambda *args:cached
        def deny_forward(*args,**kwargs):
            raise PermissionError('Saved-prediction scoring forbids model forward')
        torch.nn.Module._call_impl=deny_forward
        scored=runtime.case(('test',sample,receipt['prediction_sha256']))
        result={'action':action,'sample_id':sample['sample_id'],'methods':len(scored['methods']),'model_forward_blocked':True,'API_calls':0,
                'metrics':{method:{constraint:runtime.metrics(stats) for constraint,stats in values.items()} for method,values in scored['methods'].items()}}
    else:
        raise SystemExit('Expected predict or score')
    (output/'receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))


if __name__=='__main__':
    main()
