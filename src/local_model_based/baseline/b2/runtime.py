"""동결 b2 캐시를 읽어 예측하고 별도 GT로 재채점한다."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from typing_extensions import assert_never

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bundle_paths import REPORTS, RESULTS, artifact_path

DATA = ROOT / 'data/b2'
HORIZONS = [0.5, 1.0, 2.0, 3.0]

class InvalidCache(ValueError):
    """유효하지 않은 동결 입력은 재호출하지 않고 중단한다."""

def read(path):
    return json.loads(path.read_text())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()

def hold(current):
    value = np.asarray(current, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise InvalidCache('current_VA_requires_two_finite_values')
    return np.broadcast_to(np.clip(value, -1, 1), (4, 2)).copy()

def predictions(stage):
    manifest = read(DATA / 'inputs/forecast.json')
    for row in manifest:
        if any(frame['pts_s'] > row['cutoff_s'] for frame in row['frames']):
            raise InvalidCache('future_observation')
    histories = {row['sample_id']: row['raw_recognition_states'] for row in read(DATA / 'local_predictions/forecast_states.json')}
    output = []
    for row in manifest:
        states = histories[row['sample_id']]
        observed = {(frame['frame_index'], frame['pts_s']) for frame in row['frames']}
        if len(states) != 3 or any((state['frame'], state['pts_s']) not in observed for state in states):
            raise InvalidCache('recognition_not_in_observations')
        output.append({'sample_id': row['sample_id'], 'prediction': hold(states[-1]['VA']).tolist()})
    return output

def scores(rows):
    truth = {(r['sample_id'], r['horizon_s']): r['VA'] for r in read(DATA / 'ground_truth/forecast.json')}
    split = {r['sample_id']: r['split'] for r in read(DATA / 'shared_split.json')}
    result = {}
    for scope in ('all_available', 'shared_final421', 'shared908'):
        selected = [r for r in rows if scope == 'all_available' or
                    (scope == 'shared_final421' and split.get(r['sample_id']) == 'final') or
                    (scope == 'shared908' and r['sample_id'] in split)]
        p = np.array([r['prediction'] for r in selected]).reshape(len(selected), 8)
        y = np.array([[truth[(r['sample_id'], h)] for h in HORIZONS] for r in selected]).reshape(len(selected), 8)
        covariance = np.mean((p - p.mean(0)) * (y - y.mean(0)), axis=0)
        ccc = 2 * covariance / (p.var(0) + y.var(0) + (p.mean(0) - y.mean(0)) ** 2)
        result[scope] = {'n': len(selected), 'MAE': float(np.abs(p-y).mean()), 'CCC': float(ccc.mean()),
                         'MAE_by_horizon_axis': np.abs(p-y).mean(0).reshape(4,2).tolist(),
                         'CCC_by_horizon_axis': ccc.reshape(4,2).tolist()}
    return {'scores': result, 'requested': 1258, 'success': len(rows), 'excluded': 1258-len(rows),
            'api_success': 908, 'api_target_abstention': 287, 'api_identity_unresolved': 63,
            'seen_final': True, 'paid_calls': 0}

def verify():
    records = read(REPORTS / 'b2/copy_manifest.json')
    for row in records:
        path = artifact_path(row['destination'])
        if path.is_symlink() or path.stat().st_nlink != 1 or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise InvalidCache('copied_asset_mismatch:' + row['destination'])
    return {'verified_files': len(records), 'verified_bytes': sum(r['bytes'] for r in records), 'paid_calls': 0}

def main(stage):
    parser = argparse.ArgumentParser(description='저장 캐시 전용 b2 재현. API 호출·학습 없음.')
    parser.add_argument('action', choices=['status','replay','evaluate','verify','predict'])
    parser.add_argument('--output', type=Path, help='프로젝트 results/local_model_based/b2 내부의 새 디렉터리')
    parser.add_argument('--current-va', nargs=2, type=float, help='로컬 현재 valence arousal → hold')
    parser.add_argument('--recognize', action='store_true', help='로컬 실제 관측 JPEG 5장으로 VEATIC 추론')
    parser.add_argument('--sample-id', default='v000_t0050.00')
    args = parser.parse_args()
    if args.action in ('predict','replay'):
        def reject_gt(event, arguments):
            if event == 'open' and isinstance(arguments[0], (str, bytes)):
                value = str(arguments[0])
                if '/ground_truth/' in value or '/labels_' in value or '/b2/reference/' in value:
                    raise PermissionError('future_GT_forbidden_during_inference')
        sys.addaudithook(reject_gt)
    match args.action:
        case 'status':
            result = {'family': 'local', 'stage': stage, 'requested': 1258,
                      'method': 'recognition_hold',
                      'api_calls_allowed': False, 'training_allowed': False}
        case 'verify':
            result = verify()
        case 'predict' if args.current_va is not None:
            result = {'prediction': hold(args.current_va).tolist(), 'source': 'current_VA', 'paid_calls': 0}
        case 'predict' if args.recognize:
            from recognize import recognize
            result = recognize(args.sample_id)
        case 'replay' | 'predict' | 'evaluate':
            rows = predictions(stage)
            result = scores(rows) if args.action == 'evaluate' else {'predictions': rows, 'paid_calls': 0}
        case unreachable:
            assert_never(unreachable)
    destination = artifact_path(args.output) if args.output else RESULTS / 'b2' / (args.action + '-' + stage + '-' + uuid4().hex)
    destination = destination.resolve()
    if not destination.is_relative_to(RESULTS / 'b2'):
        raise InvalidCache('output_must_be_inside_family_results_b2')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'artifact': str(destination / 'result.json'), 'action': args.action,
                      **({} if 'predictions' in result else result)}, allow_nan=False))
