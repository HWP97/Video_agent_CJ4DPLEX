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
API = True
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

def cached(signature):
    folder = DATA / 'sol/cache' / signature
    metadata = read(folder / 'request.json')
    identity = read(folder / 'identity_v2.json') if (folder / 'identity_v2.json').exists() else metadata
    if identity['signature'] != signature or metadata['endpoint'] != 'https://api.openai.com/v1/responses' or metadata['model'] != 'gpt-6.1-sol':
        raise InvalidCache('endpoint_model_signature_mismatch')
    if digest(folder / 'response.json') != metadata['response_sha256']:
        raise InvalidCache('response_hash_mismatch')
    if metadata.get('input_record_sha256') and digest(folder / 'input.json') != metadata['input_record_sha256']:
        raise InvalidCache('input_record_hash_mismatch')
    response = read(folder / 'response.json')
    texts = [part['text'] for item in response['output'] if item['type'] == 'message'
             for part in item['content'] if part['type'] == 'output_text']
    if len(texts) != 1:
        raise InvalidCache('missing_structured_output')
    return json.loads(texts[0])

def predictions(stage):
    manifest = read(DATA / 'inputs/forecast.json')
    for row in manifest:
        if any(frame['pts_s'] > row['cutoff_s'] for frame in row['frames']):
            raise InvalidCache('future_observation')
    if API:
        rows = read(DATA / 'ridge/inputs.json')
        signatures = read(DATA / 'sol/request_signatures.json')['signatures']
        cache = {signature: cached(signature) for signature in set(signatures.values())}
        raw = []
        current = []
        for row in rows:
            value = cache[row['source_signature']]
            if not value['target_confirmed'] or [v['horizon_s'] for v in value['future']] != HORIZONS:
                raise InvalidCache('invalid_target_or_horizons')
            raw.append([[v['valence'], v['arousal']] for v in value['future']])
            current.append([value['current']['valence'], value['current']['arousal']])
        baseline = np.asarray(raw)
        if not np.array_equal(baseline, np.array([r['baseline'] for r in rows])) or not np.array_equal(current, [r['current'] for r in rows]):
            raise InvalidCache('parsed_frozen_input_mismatch')
        output = baseline
        if stage == 'final':
            from ridge_model import Inputs, predict
            frozen = read(DATA / 'ridge/frozen_model.json')
            inputs = Inputs(baseline, np.asarray(current), np.array([r['local'] for r in rows]), np.array(HORIZONS))
            _, output = predict(inputs, frozen['models'][frozen['selected']])
        return [{'sample_id': row['sample_id'], 'prediction': prediction.tolist()} for row, prediction in zip(rows, output)]

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
    parser.add_argument('--output', type=Path, help='프로젝트 results/api_based/b2 내부의 새 디렉터리')
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
            result = {'family': 'api', 'stage': stage, 'requested': 1258,
                      'method': 'raw_sol' if stage == 'baseline' else 'observation_residual_lambda1',
                      'api_calls_allowed': False, 'training_allowed': False}
        case 'verify':
            result = verify()
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
