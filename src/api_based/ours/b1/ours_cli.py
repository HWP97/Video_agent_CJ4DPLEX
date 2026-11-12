# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: python main.py b1 ours evaluate --limit 1
"""기존 동결 API 후처리 방법의 명시적 실행 경계."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT / 'baseline/b1/vendor'),str(Path(__file__).resolve().parent)]
from bundle_paths import RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','replay','evaluate','predict','verify'])
    parser.add_argument('--research',choices=['rank_residual'],help='기존 WC 연구 옵션; 기본은 macro_direct')
    parser.add_argument('--sample')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--input',type=Path,help='관측 전용 entry/features pickle; GT 포함 캐시는 금지')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit은 양수여야 합니다')
    output = (args.output or RESULTS / 'b1' / f'ours_{args.action}_{time.time_ns()}').resolve()
    if not (output.is_relative_to(ROOT) or output.is_relative_to(RESULTS)):
        parser.error('출력은 현재 계열 소스 또는 결과 폴더 내부여야 합니다')
    output.mkdir(parents=True,exist_ok=False)
    from ours_method import execute
    result = execute(args,output)
    result.update(stage='ours',method=args.research or 'macro_direct',paid_api_calls=0,training_calls=0,
                  selection='기존 공개 성적을 본 뒤 NC 관측 0.5 성능으로 사후 선택',
                  original_WC_gate_passed=bool(args.research),
                  tradeoff='macro_direct는 기존 WC 개발 채택 실패 및 WC R 하락; 전 조건 동일 방법, 자동 라우팅 없음')
    (output / 'receipt.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False))
