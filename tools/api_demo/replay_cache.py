"""demo/data/api_cache 의 15개 Sol 응답 파일 존재·형식·catalog 매칭을 확인한다.

**이 도구는 지표를 재계산하지 않는다.** 각 entry 에 대해:
  1) input/request/response.json 세 파일 존재 여부
  2) response.status == 'completed'
  3) response.output[*].content 중 output_text 의 JSON 파싱 가능 여부
  4) index.json 의 sample_id 와 catalog 의 sample_id 일치 여부
를 확인하고, 보고에는 **catalog 에 이미 저장된 baseline_metric / ours_metric 을 그대로 출력**한다.
수치 재계산·채점·저장값 대조는 각 head 의 ours 파이프라인(src/api_based/ours/...)에서 수행한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO_CACHE = ROOT / 'demo/data/api_cache'
CATALOGS = {
    'affect_forecasting': ROOT / 'demo/api_based/affect_forecasting/catalog.json',
    'scene_graph_anticipation': ROOT / 'demo/api_based/scene_graph_anticipation/catalog.json',
    'trajectory_forecasting': ROOT / 'demo/api_based/trajectory_forecasting_reselected/catalog.json',
}

# 활성 CACHE 는 CLI/env 로 교체 가능하다. 기본은 Repository 내장 15개 서브셋.
CACHE = DEMO_CACHE


def catalog_row(task: str, rank: int) -> dict:
    for row in json.loads(CATALOGS[task].read_text()):
        if row.get('rank') == rank:
            return row
    raise SystemExit(f'catalog에 {task} rank {rank} 없음')


def extract_model_text(response: dict) -> str | None:
    """OpenAI Responses API v1의 출력에서 모델이 쓴 JSON 문자열을 추출."""
    for item in response.get('output', []):
        if item.get('type') == 'message':
            for part in item.get('content', []):
                if part.get('type') == 'output_text' and part.get('text'):
                    return part['text']
    return None


def check_one(entry: dict) -> tuple[bool, str]:
    task, rank, sample_id, signature = entry['task'], entry['rank'], entry['sample_id'], entry['signature']
    folder = CACHE / task / signature
    missing = [name for name in ('input.json', 'request.json', 'response.json') if not (folder / name).is_file()]
    if missing:
        return False, f'cache 누락: {missing}'
    response = json.loads((folder / 'response.json').read_text())
    if response.get('status') != 'completed':
        return False, f'응답 상태={response.get("status")}'
    text = extract_model_text(response)
    if text is None:
        return False, '모델 출력 텍스트 없음'
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False, '모델 출력이 유효한 JSON 아님'
    row = catalog_row(task, rank)
    if row.get('sample_id') != sample_id:
        return False, f'sample_id 불일치: catalog={row.get("sample_id")}, cache={sample_id}'
    metric = row.get('metric', '?')
    baseline = row.get('baseline_metric')
    ours = row.get('ours_metric')
    if baseline is not None and ours is not None:
        return True, f'{metric} · Baseline {baseline:.4f} → Ours {ours:.4f}'
    return True, metric


def main() -> int:
    global CACHE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(CATALOGS))
    parser.add_argument('--rank', type=int)
    parser.add_argument('--dataset', help='외부 저장소 루트. 지정 시 <dataset>/api_cache 를 사용한다. 기본: 환경변수 VFA_DATASET_DIR, 둘 다 없으면 Repository 내장 서브셋.')
    args = parser.parse_args()
    override = args.dataset or os.environ.get('VFA_DATASET_DIR')
    if override:
        candidate = Path(override) / 'api_cache'
        if not (candidate / 'index.json').is_file():
            raise SystemExit(f'지정한 --dataset 에 {candidate}/index.json 가 없습니다. 외부 묶음의 index.json 경로를 확인하세요.')
        CACHE = candidate
        print(f'[CACHE] {CACHE}\n', file=sys.stderr)
    index = json.loads((CACHE / 'index.json').read_text())
    if args.task:
        index = [e for e in index if e['task'] == args.task]
    if args.rank is not None:
        index = [e for e in index if e['rank'] == args.rank]
    if not index:
        print('매칭되는 cache entry 없음'); return 1
    ok = 0
    for entry in index:
        passed, detail = check_one(entry)
        flag = 'OK ' if passed else 'FAIL'
        print(f'{flag}  {entry["task"]:28s} rank {entry["rank"]:2d}  {entry["sample_id"]:40s}  {detail}')
        if passed: ok += 1
    print(f'\n{ok}/{len(index)} entries 검증 통과 (지표 재계산 아님, catalog 매칭만)')
    return 0 if ok == len(index) else 2


if __name__ == '__main__':
    sys.exit(main())
