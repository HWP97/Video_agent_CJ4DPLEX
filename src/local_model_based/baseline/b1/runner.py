# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: 기존 AG Python entry.py status|replay|evaluate|verify|predict
"""로컬 관측 GT bbox benchmark CLI."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / 'baseline/b1/vendor'))
from bundle_paths import RESULTS


def main(stage: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status','replay','evaluate','verify','predict'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, help='생략하면 전체 유효 사례를 재채점')
    parser.add_argument('--sample', help='predict 기본값 predcls:00607.mp4_0.5; 제공 프레임은 이 영상의 0.5 관측까지')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit은 양수여야 합니다')
    output = (args.output or RESULTS / 'b1' / f'{stage}_{args.action}_{time.time_ns()}').resolve()
    if not (output.is_relative_to(ROOT) or output.is_relative_to(RESULTS)):
        parser.error('출력은 해당 계열 내부여야 합니다')
    output.mkdir(parents=True, exist_ok=False)
    if args.action == 'predict':
        from prediction import predict
        result = predict(args.sample or 'predcls:00607.mp4_0.5',output)
    else:
        from operations import execute
        result = execute(args,output)
    result.update(stage=stage, paid_api_calls=0, training_calls=0,method='impartail_selected')
    (output / 'receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
