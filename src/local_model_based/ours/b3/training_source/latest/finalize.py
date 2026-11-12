# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "tokenizers"]
# ///
# How to run: python -B finalize.py
"""전수 통계의 별도 집계와 원본 보존 확인 뒤 한국어 결과를 작성한다."""
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from common import FOLDER, OLD, ROOT, ORIGINAL, SEEDS, EPOCHS, sha, save


def write_csv(path: Path, records: list) -> None:
    """동일 스키마의 실험 수치를 CSV 부록에 남긴다."""
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    rows, aggregated, independent, full_epoch0 = [], {}, [], []
    for fold in (1, 2, 3):
        inference = json.loads((FOLDER / f'inference_fold{fold}.json').read_text())
        assert all(item['maxdiff'] == 0 for item in inference['replay'])
        for item in inference['epoch0_full_preservation_vs_historical_F0']:
            assert item['parent_state_and_transform_bitwise_equal']
            full_epoch0.append({'fold': fold, **item})
        folder = FOLDER / 'scores' / f'fold{fold}'
        receipt = json.loads((folder / 'receipt.json').read_text())
        assert all(sha(folder / p) == h for p, h in receipt['outputs'].items())
        summary = json.loads((folder / 'summary.json').read_text())
        arrays = [np.load(folder / p) for p in receipt['outputs'] if p.endswith('.npz')]
        names = arrays[0]['methods'].tolist()
        ids = np.concatenate([a['sample_ids'] for a in arrays])
        assert len(ids) == len(set(ids.tolist())) == receipt['n']
        for index, name in enumerate(names):
            ade = np.concatenate([a['ADE'][index] for a in arrays])
            fde = np.concatenate([a['FDE'][index] for a in arrays])
            hits = np.concatenate([a['collision'][index, :, 1] for a in arrays])
            metrics = {'n': len(ids), 'ADE': float(ade.mean()), 'FDE': float(fde.mean()), 'ColI_obs': float(hits.mean() * 100)}
            assert all(abs(metrics[k] - summary[name][k]) < 1e-12 for k in ('ADE', 'FDE', 'ColI_obs'))
            rows.append({'fold': fold, 'method': name, **metrics})
            aggregated.setdefault(name, []).append(metrics)
        for values in arrays:
            values.close()
        independent.append({'fold': fold, 'n': len(ids), 'duplicates': 0, 'missing': 0, 'max_allowed_difference': 1e-12})
    total = sum(r['n'] for r in independent)
    assert total == 240890
    aggregate = []
    for name, records in aggregated.items():
        assert sum(r['n'] for r in records) == total
        aggregate.append({'method': name, 'n': total, **{key: sum(r[key] * r['n'] for r in records) / total for key in ('ADE', 'FDE', 'ColI_obs')}})
    write_csv(FOLDER / 'scores_by_fold.csv', rows)
    write_csv(FOLDER / 'aggregate.csv', aggregate)
    save(FOLDER / 'aggregate.json', aggregate)
    trajectories, choices, optimizer = [], [], []
    for fold in (1, 2, 3):
        lock = json.loads((FOLDER / f'selection_fold{fold}.json').read_text())
        for seed in SEEDS:
            item = lock['seeds'][str(seed)]
            choices.append({'fold': fold, 'seed': seed, 'robust_epoch': item['robust_epoch'], 'single_epoch': item['single_epoch']})
            for candidate in item['candidates']:
                for split in (0, 1):
                    trajectories.append({'fold': fold, 'seed': seed, 'split': split, 'epoch': candidate['epoch'],
                         'ADE': candidate['ADE'][split], 'ADE_over_Kalman': candidate['ratios'][split], 'worst_ratio': candidate['worst_ratio']})
        for name in ('B3-D03', 'B3-D04'):
            receipt = json.loads((FOLDER / 'inner' / f'fold{fold}' / name / 'receipt.json').read_text())
            optimizer.extend({'fold': fold, 'method': name, 'selected_epoch': receipt['selected_epoch'], **r} for r in receipt['trace'])
    write_csv(FOLDER / 'internal_epoch_comparison.csv', trajectories)
    write_csv(FOLDER / 'selection_epochs.csv', choices)
    write_csv(FOLDER / 'finetune_traces.csv', optimizer)
    original = json.loads((FOLDER / 'preservation_before.json').read_text())
    changed = [p for p, h in original.items() if sha(Path(p)) != h]
    assert not changed
    sources = json.loads((FOLDER / 'training_source_lock.json').read_text())
    assert all(sha(Path(p)) == h for p, h in sources.items())
    inference_sources = json.loads((FOLDER / 'inference_source_lock.json').read_text())
    assert all(sha(Path(p)) == h for p, h in inference_sources.items())
    contracts = json.loads((FOLDER / 'contract_verification.json').read_text())
    assert contracts['failures'] == 0
    assert contracts['actual_new_fit_epoch0_rejections']
    deviation = json.loads((FOLDER / 'execution_deviation.json').read_text())
    assert deviation['validation_outputs_bitwise_equal'] and deviation['best_snapshot_byte_equal_including_optimizer_RNG']
    servers = []
    before_processes = json.loads((FOLDER / 'processes_before.json').read_text())['processes']
    current_processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,etimes,args'], text=True)
    current_lines = {line.split(None, 3)[0]: line.split(None, 3)[3] for line in current_processes.splitlines()[1:] if len(line.split(None, 3)) == 4}
    for line in before_processes.splitlines()[1:]:
        fields = line.split(None, 3)
        if len(fields) == 4 and any(marker in fields[3] for marker in ('status_report.py watch', 'status_report.py serve', '/C/main.py web', '/benchmark_viewer/server.py')):
            assert current_lines[fields[0]] == fields[3]
            servers.append(int(fields[0]))
    assert len(servers) == 4
    dependency_lock = json.loads((OLD / 'registration_fold1.json').read_text())['sources']
    assert all(sha(Path(p)) == h for p, h in dependency_lock.items())
    new_fits = list((FOLDER / 'inner').rglob('receipt.json')) + list((FOLDER / 'full').rglob('receipt.json'))
    for path in new_fits:
        record = json.loads(path.read_text())
        assert all(sha(path.parent / p) == h for p, h in record['checkpoints'].items())
    reference = {r['method']: r for r in aggregate}
    assert abs(reference['previous_F']['ADE'] - 1.0265881075446284) < 1e-10
    assert abs(reference['B3-D01']['ADE'] - 1.0578762368435914) < 1e-10
    assert abs(reference['B3-D02']['ADE'] - 1.0904305508043008) < 1e-10
    methods = ['baseline_attention', 'previous_F', 'B3-D01', 'B3-D02', 'B3-D03', 'B3-D04',
               'F_robust_seed20261111', 'F_single_seed20261111', 'F_robust_seed20261110', 'F_single_seed20261110']
    descriptions = {'baseline_attention': '원래 attention', 'previous_F': '기존 유망 F, 역사적 참조',
                    'B3-D01': '새 F 그대로, 추가학습0', 'B3-D02': '완료 C01 재사용, lr0.001 재시작',
                    'B3-D03': 'lr0.0001 재시작, epoch0 포함', 'B3-D04': 'lr0.0001 moment 유지, epoch0 포함'}
    table = ['| 조건 | 구성 | ADE↓ | FDE↓ | 관측 이웃 충돌률(%)↓ |', '|---|---|---:|---:|---:|']
    for name in methods:
        r = reference[name]
        table.append(f"| {name} | {descriptions.get(name, '그룹별 최악 비율 선택' if 'robust' in name else '기존 한 분리 ADE 선택')} | {r['ADE']:.6f} | {r['FDE']:.6f} | {r['ColI_obs']:.6f} |")
    epoch_lines = [f"- fold{r['fold']}, seed{r['seed']}: 두 분리 최악 비율 epoch{r['robust_epoch']}, 한 분리 최소 ADE epoch{r['single_epoch']}" for r in choices]
    additions = [f"- fold{r['fold']}: D03 epoch{r['D03_epoch']}, D04 epoch{r['D04_epoch']}" for r in contracts['optimizer_comparisons']]
    d1, d3, d4, best = (reference[n] for n in ('B3-D01', 'B3-D03', 'B3-D04', 'previous_F'))
    text = '# b3 진단 기반 후속 실험 결과\n\n'
    text += f'3fold 전체 {total:,}장면을 평가했다. 누락0, 중복0이다. 내부 선택 결과와 아래 최종 평가 결과는 구분한다. 반복 열람한 평가 영역이며 새 독립 holdout이 아니다.\n\n'
    text += '\n'.join(table) + '\n\n'
    text += '## 어느 단계가 문제였나\n\n'
    text += '기존 기본 F에서 새 기본 F로 바뀐 단계와 추가학습 단계에서 각각 성능이 달라졌다. epoch0을 제외한 과거 C01 선택기는 추가학습 전체가 나빠도 시작 모델을 남길 수 없었다. 실제 fold3 내부 예측과 완료된10epoch 기록에서도 이 문제가 확인됐다. 별도로 새 D03/D04 실행에서 모든 추가 epoch가 나쁜 실제 조건을 찾아 selected_epoch0, best.pt와 epoch0 가중치 동일, 저장 후 실제 예측 재생 동일을 검증했다. 역사적 counterfactual 계산과 새 학습 경로의 실제 검증은 receipt에서 구분한다.\n\n'
    text += '## 수정으로 무엇이 회복됐나\n\n'
    text += '\n'.join(additions) + '\n\n'
    text += f"D01의 ADE는 {d1['ADE']:.6f}이고 D03은 {d3['ADE']:.6f}, D04는 {d4['ADE']:.6f}다. epoch0 포함은 손해인 학습을 거절할 수 있게 하지만 최종 평가 개선을 보장하지 않는다. D03/D04의 parent 가중치·초기 점수·RNG·epoch별 배치 순서는 같고 D04는10개 parameter의 실제 optimizer moment/step을 복원했다. 이 두 조건 차이는 optimizer 상태 차이로 제한한다. D02는 과거 lr와 epoch0 제외 정책이 함께 달라 완전한 단일 요인 대조가 아니다.\n\n"
    text += '## 기본 F의 epoch·seed·검증 분리\n\n'
    text += '\n'.join(epoch_lines) + '\n\n'
    text += '동일 후보 풀1/3/5/10/20/25epoch에서 각 모델이 학습하지 않은 원본 그룹의 ADE/Kalman 비율을 계산했다. 두 내부 분리의 최악 비율로 고른 정책과 기존 한 분리 최소 ADE 정책을 모두 같은 두 seed로 비교했다. 각 fold는 자기 train의 결과만으로 선택했다. epoch별 재학습 없이 각 궤적에서 checkpoint를 저장했고, 전체 train refit 이전에 모든 선택을 동결했다. 세부 수치와 분리별 차이는 CSV에 남겼다. 두 seed의 최종 평가 차이는 같은 정책에서도 선택 epoch가 달라질 수 있어 순수 seed 효과로 읽지 않는다. 같은 분리·같은 epoch의 seed 대조는 내부 CSV에서 확인한다.\n\n'
    robust = reference['F_robust_seed20261111']
    text += f"기준 seed20261111에서는 두 그룹의 최악 비율 규칙도 세 fold 모두25epoch를 선택했다. 추가 검증 그룹을 넣었다고 짧은 학습이 자동으로 선택되지는 않았다. 이 정책의 최종 ADE/FDE/충돌률은 {robust['ADE']:.6f}/{robust['FDE']:.6f}/{robust['ColI_obs']:.6f}%다. 내부 그룹 선택과 최종 영역의 일반화 차이가 남는지 이 수치와 과거 F 참조를 함께 보되, 평가 결과로 epoch를 다시 고르지 않았다.\n\n"
    better = [r['method'] for r in aggregate if r['method'] not in ('baseline_attention', 'previous_F', 'B3-D01', 'B3-D02') and all(r[k] <= best[k] for k in ('ADE', 'FDE', 'ColI_obs'))]
    text += ('이번 조건에서 기존 F의 세 지표를 모두 회복한 조건은 ' + ', '.join(better) + '이다.\n\n') if better else '이번 조건 중 기존 F의 ADE/FDE/관측 이웃 충돌률을 모두 회복한 조건은 없다. 따라서 기존 최선의 복원을 달성했다고 보고하지 않는다.\n\n'
    text += '과거 F는 외부 개발 선택·다른 학습 길이·seed가 함께 다른 역사적 참조다. 이번 통제 실험에서 얻은 차이와 그 역사적 점수 차이를 특정 한 요인의 인과효과로 합치지 않는다. fold3의 train 원본 그룹은3개로 적어 두 분리의 일반화 판단에는 한계가 있다. G/P/U, 새 gate·충돌·증류 구조와 추가 탐색은 실행하지 않았다.\n\n'
    text += '## 검증과 산출물\n\n관측만 읽는 별도 추론에서 조건별512사례를 재생했다. 동일 seed에서 정책 epoch가 같으면 같은 모델·예측을 재사용했다. F0는 진단 예측 SHA와 parent checkpoint를 확인하고 원본 scorer로 충돌률을 새로 계산했다. 완료 참조 예측과 기존 채점의 SHA·입력 순서를 확인해 재사용했다. 최종 점수는 파일별 배열을 다시 연결해 독립 집계했고 합산 방식과1e-12 이내로 일치했다. 원본/진단/기존 lock과 모델을 SHA로 보존 확인했다. 신규 유료 API0건, 모델 다운로드0건, 서비스 기본값 변경0건이며 ZIP을 만들지 않았다.\n\n'
    text += '[fold별 점수](scores_by_fold.csv) · [내부 epoch별 비교](internal_epoch_comparison.csv) · [선택 epoch](selection_epochs.csv) · [추가학습 이력](finetune_traces.csv) · [실제 계약 검증](contract_verification.json) · [완료 receipt](COMPLETION_RECEIPT.json)\n'
    text += '\n실행 중 fold1 D04의 선계산과 주 실행기가 같은 폴더에 동시에 진입한 편차1건이 있었다. 중복 완료3epoch와 추가 부분 epoch 가능성을 기록하고 주 실행기를 중단했다. 해당 폴더 원시 증거를 quarantine에 보존한 뒤, 완료 helper의 정상 epoch5 snapshot SHA와10epoch trace·실제 내부 예측을 확인해 최선 checkpoint를 복구했다. 복구 best는 epoch5 snapshot과 바이트 단위로 같아 가중치·optimizer·RNG 상태를 함께 보존한다. 복구를 위한 새 학습은0회이며 이후 전체 refit은 직렬로 수행했다. [실행 편차와 복구 증거](execution_deviation.json)에 상세를 남겼다.\n'
    maximum = max(item['historical_F0_maxdiff'] for item in full_epoch0)
    text += f'\n전체 train에서도 epoch0을 택한 조건은 parent 가중치와 정규화 tensor가 완전히 같았다. 새 환경의512사례 자체 재생 오차는0이고, 역사적 F0 예측과의 전수 최대 절대차는 {maximum:.12g}다. 역사적 추론과 새 추론의 thread·결정론 설정 차이는 모델 수정 효과와 구분한다.\n'
    text += '\n추론 첫 시작 전에는 새 helper common과 원 adapter common의 모듈명 충돌로 import가 실패했다. 예측 생성 전의 실패이며 새 helper만 고유 모듈명으로 격리했다. 학습 source·checkpoint·원본은 변경하지 않았고, 재시작한3fold 실제 추론으로 수정 동작을 검증했다. 원 실패 로그와 별도 추론 import 편차 기록을 보존했다.\n'
    (FOLDER / 'RESULTS.md').write_text(text)
    unicode_check = subprocess.run(['rg', '-n', '--pcre2', r'[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]', str(FOLDER), '-g', '*.md'], capture_output=True, text=True)
    assert unicode_check.returncode == 1, unicode_check.stdout
    save(FOLDER / 'verification.json', {'independent_aggregation': independent, 'total': total, 'protected_files': len(original),
         'changed_protected_files': changed, 'all_model_checkpoints_SHA_verified': True, 'training_source_lock_unchanged': True,
         'unicode_script_check_passed': True, 'failures': 0, 'actual_epoch0_rejection': contracts['actual_new_fit_epoch0_rejections'],
         'historical_counterfactual': contracts['epoch0_rejection'], 'server_pids_preserved': servers,
         'historical_dependency_lock_verified': True, 'new_fit_count': len(new_fits),
         'full_epoch0_preservation': full_epoch0,
         'recovered_execution_deviations': 2, 'verification_adjustments': 1, 'outstanding_execution_failures': 0})
    outputs = {str(p.relative_to(FOLDER)): sha(p) for p in FOLDER.rglob('*') if p.is_file() and p.name != 'COMPLETION_RECEIPT.json' and '__pycache__' not in p.parts and not p.name.endswith('.log')}
    save(FOLDER / 'COMPLETION_RECEIPT.json', {'complete': True, 'n': total, 'folds': 3, 'missing': 0, 'duplicates': 0,
         'failures': 0, 'new_paid_API_calls': 0, 'model_downloads': 0, 'service_defaults_changed': False,
         'preserved_files': len(original), 'server_pids_preserved': servers,
         'recovered_execution_deviations': 2, 'verification_adjustments': 1, 'outstanding_execution_failures': 0, 'outputs': outputs})
    print(json.dumps({'complete': True, 'n': total, 'failures': 0}), flush=True)


if __name__ == '__main__':
    main()
