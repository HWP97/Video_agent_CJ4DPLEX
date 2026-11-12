# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: 기존 AG Python entry.py status|replay|evaluate|verify|predict
"""b1 캐시 작업 CLI. 새 API 요청이나 학습 경로가 없다."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / 'baseline/b1/vendor'))
from bundle_paths import RESULTS


def main(stage: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status','replay','evaluate','verify','predict'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, help='생략하면 전체 유효 사례를 재채점')
    parser.add_argument('--sample', help='예: sgcls:00607.mp4_0.5')
    parser.add_argument('--research', choices=['rank_residual'])
    parser.add_argument('--input', type=Path, help='rank_residual predict용 관측 전용 entry/features pickle')
    args = parser.parse_args()
    if args.research and stage != 'final':
        parser.error('연구 옵션은 final에서만 사용합니다')
    if args.research:
        sys.path.insert(0,str(ROOT / 'ours/b1'))
    if args.limit is not None and args.limit < 1:
        parser.error('--limit은 양수여야 합니다')
    output = args.output or RESULTS / 'b1' / f'{stage}_{args.action}_{time.time_ns()}'
    output = output.resolve()
    if not (output.is_relative_to(ROOT) or output.is_relative_to(RESULTS)):
        parser.error('출력은 해당 계열 내부여야 합니다')
    output.mkdir(parents=True, exist_ok=False)
    if args.action == 'predict':
        from prediction import predict
        result = predict(args, output)
    else:
        from operations import execute
        result = execute(args, output)
    result.update(stage=stage, paid_api_calls=0, training_calls=0)
    (output / 'receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

