# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: ours_cli.py에서 호출
"""동결 모델 확인과 미래 GT 없는 후처리."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import torch
from adaptive import Adaptive, Decoder, predict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/b1'


def decoder(mode: str, name: str) -> Decoder:
    checkpoint = torch.load(DATA / f'models/{mode}_{name}.pt',map_location='cpu',weights_only=False)
    model = Adaptive(checkpoint['residual'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    return Decoder(model,checkpoint['support'])


def execute(args, output: Path):
    torch.set_num_threads(1)
    name = args.research or 'macro_direct'
    if args.action == 'status':
        return {'scheduled':13948,'eligible':13635,'empty_future':313}
    if args.action == 'verify':
        rows = json.loads((DATA / 'ours_assets.json').read_text())
        for row in rows:
            path = ROOT / row['destination']
            assert path.stat().st_nlink == 1 and not path.is_symlink()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['destination_sha256']
        return {'new_assets_checked':len(rows),'failures':0}
    if args.action in ('evaluate','replay'):
        from ours_scores import score
        return score(args,output)
    sample = args.sample or 'sgcls:00607.mp4_0.5'
    def guard(event: str, values: tuple) -> None:
        if event == 'open' and isinstance(values[0],(str,bytes)):
            path = os.fsdecode(values[0])
            if 'future_gt' in path or '/prepared/' in path:
                raise PermissionError(path)
    sys.addaudithook(guard)
    path = args.input or DATA / 'inference' / (sample.replace(':','__')+'.pkl')
    with path.open('rb') as stream:
        record = pickle.load(stream)
    model = decoder(sample.split(':',1)[0],name)
    result = {constraint:[predict(entry,feature,model) for entry,feature in values] for constraint,values in record.items()}
    with (output / 'prediction.pkl').open('xb') as stream:
        pickle.dump(result,stream,protocol=5)
    return {'sample_id':sample,'future_GT_blocked':True,'constraints':list(result)}
