# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: entry.py predict --sample sgcls:00607.mp4_0.5
"""GT 없는 저장 Sol 응답 또는 관측 전용 연구 입력의 후처리."""
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]


def predict(args, output: Path):
    sample = args.sample or 'sgcls:00607.mp4_0.5'
    def guard(event: str, values: tuple) -> None:
        if event == 'open' and isinstance(values[0], (str,bytes)):
            path = os.fsdecode(values[0])
            if 'future_gt' in path or '/prepared/' in path:
                raise PermissionError(path)
    sys.addaudithook(guard)
    if not args.research:
        with (ROOT / 'data/b1/sol/index.final.jsonl').open() as stream:
            row = next(row for row in map(json.loads,stream) if row['sample_id'] == sample)
        if not row.get('valid'):
            raise KeyError(sample)
        (output / 'prediction.json').write_text(json.dumps(row['prediction']))
        return {'sample_id':sample,'method':'raw_sol','future_GT_blocked':True,'source':'saved_parsed_Sol'}
    import torch
    from adaptive import Adaptive, Decoder, predict as transform
    torch.set_num_threads(1)
    path = args.input or ROOT / 'data/b1/inference' / (sample.replace(':','__')+'.pkl')
    with path.open('rb') as stream:
        record = pickle.load(stream)
    checkpoint = torch.load(ROOT / f'data/b1/models/{sample.split(":")[0]}_rank_residual.pt',map_location='cpu',weights_only=False)
    model = Adaptive(checkpoint['residual'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    decoder = Decoder(model,checkpoint['support'])
    result = {constraint:[transform(entry,feature,decoder) for entry,feature in values] for constraint,values in record.items()}
    with (output / 'prediction.pkl').open('xb') as stream:
        pickle.dump(result,stream,protocol=5)
    return {'sample_id':sample,'method':'rank_residual','future_GT_blocked':True,'constraints':list(result)}
