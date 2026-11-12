# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
# 실행: train_sweep.py가 호출
"""기존 공식 train 관측 특징과 target을 명시적으로 분리해 포장한다."""
from pathlib import Path
import json
import torch
from supervision import targets
from compact_graph import histories
HERE=Path(__file__).resolve().parents[1]
SOURCE=Path('/mnt/RG2/paper_experiments_20261109/combinations')


def context(history: torch.Tensor) -> torch.Tensor:
    return torch.cat((history[:,-1,:26],history[:,:,:26].mean(1),history[:,-1,26:63],history[:,-1,63:71]),-1)


def probabilities(prediction) -> torch.Tensor:
    return torch.cat((prediction['attention_distribution'].softmax(-1),prediction['spatial_distribution'],prediction['contacting_distribution']),-1).clamp(1e-5,1-1e-5)


def pack(mode: str, stage: str):
    rows=[r for r in json.loads((SOURCE/'protocol/train_manifest.json').read_text()) if r['mode']==mode and r['status']=='ready' and (stage=='final' or r['split']=='training')]
    arrays={k:[] for k in ('history','delta','base','target','middle','middle_delta','context','profile')}
    for row in rows:
        saved=torch.load(SOURCE/'assets/compact_train'/f'{mode}_{row["video"]}.pt',weights_only=False)
        record=json.loads((SOURCE/'assets/supervision'/f'{mode}_{row["video"]}.json').read_text())
        data=targets(saved['graph'],record)
        history=data['predicted_history'];c=data['coverage']['candidates'];n=len(data['delta'])//c
        y=data['future'].reshape(n,c,26);middle=torch.arange(n)//2
        base=probabilities(saved['official_prediction'])
        assert len(base)==len(history)
        arrays['history'].append(history);arrays['delta'].append(data['delta']);arrays['base'].append(base);arrays['target'].append(data['future'])
        arrays['middle'].append(y[middle].flatten(0,1));arrays['middle_delta'].append(data['delta'].reshape(n,c)[middle].flatten())
        arrays['context'].append(context(history[:c]))
        bins=torch.linspace(0,n,4).round().long()
        arrays['profile'].append(torch.stack([y[min(int(bins[i]),n-1):max(int(bins[i+1]),min(int(bins[i]),n-1)+1)].mean(0) for i in range(3)],-1))
    return {k:torch.cat(v) for k,v in arrays.items()},rows
