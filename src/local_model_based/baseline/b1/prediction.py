# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "opencv-python"]
# ///
# 실행: CUDA_VISIBLE_DEVICES=0 entry.py predict
"""관측 영상·관측 GT bbox/class와 미래 시각 목록으로 실제 ImparTail 추론."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def predict(sample: str, output: Path):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(20261109)
    sys.path.insert(0,str(ROOT / 'baseline/b1/vendor/benchmark/code'))
    from model_infer import load_models, observed_entry, SOURCE
    from model_prefix import anticipate_prefix
    mode,case_id = sample.split(':',1)
    cohort = {'sgcls':'filtered_small','predcls':'unfiltered'}[mode]
    protocol = ROOT / f'data/b1/benchmark/protocol/{cohort}'
    case = next(r for r in map(json.loads,(protocol / 'cases.jsonl').open()) if r['case_id'] == case_id)
    observation = next(r for r in map(json.loads,(protocol / 'observations.jsonl').open()) if r['case_id'] == case_id)
    def guard(event: str, args: tuple) -> None:
        if event == 'open' and isinstance(args[0], (str,bytes)):
            path = os.fsdecode(args[0])
            if 'future_gt' in path or '/sol/' in path:
                raise PermissionError(path)
    sys.addaudithook(guard)
    os.chdir(SOURCE)
    started = time.monotonic()
    model,detector,matcher = load_models(mode)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        entry,hashes = observed_entry(detector,matcher,case,observation,mode)
        assert not {'gt_annotation','attention_gt','spatial_gt','contacting_gt'} & entry.keys()
        prediction = anticipate_prefix(model,entry,[int(Path(name).stem) for name in case['future_frame_ids']])
    torch.cuda.synchronize()
    prediction = {key:value.cpu() for key,value in prediction.items()}
    torch.save(prediction,output / 'prediction.pt')
    reference = torch.load(ROOT / 'data/b1/impartail/predictions' / (sample.replace(':','__')+'.pt'),map_location='cpu',weights_only=False)
    differences = {key:float((value-reference[key]).abs().max()) if value.numel() else 0.0 for key,value in prediction.items()}
    close = all(torch.allclose(value,reference[key],rtol=1e-4,atol=1e-5) for key,value in prediction.items())
    result = {'sample_id':sample,'observed_frames':len(hashes),'frame_hashes':hashes,'future_frames':len(case['future_frame_ids']),'future_GT_blocked':True,'reference_allclose':close,'max_abs_difference':differences,'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (output / 'comparison.json').write_text(json.dumps(result,indent=2))
    if not close:
        raise AssertionError(differences)
    return result
