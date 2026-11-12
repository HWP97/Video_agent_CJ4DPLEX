"""기존 조건과 새 조합의 동일 내부 검증을 합쳐 선택을 고정한다."""
import json
from pathlib import Path
from prepare import ROOT, sha

references = {
    ROOT.parent: ('baseline', 'e01_linear_control', 'e08_memory_confidence_gate'),
    ROOT.parent / 'positive': ('e11_positive_balance0.125', 'e11_positive_balance0.25'),
    ROOT.parent / 'capacity': ('e12_cap1.5', 'e12_cap3'),
    ROOT: None,
}
inputs, rows = {}, []
for folder, methods in references.items():
    path = folder / 'evaluation/inner/summary.json'
    inputs[str(path)] = sha(path)
    for row in json.loads(path.read_text()):
        if row['constraint'] == 'no' and (methods is None or row['method'] in methods):
            rows.append(row)
selection = {}
for mode in ('sgcls', 'predcls'):
    candidates = {r['method']: r for r in rows if r['mode'] == mode}
    baseline = candidates['baseline']
    passing = [r for r in candidates.values() if r['R@20'] >= baseline['R@20'] and r['mR@20'] >= baseline['mR@20']]
    selected = max(passing, key=lambda row: row['mR@20'])
    selection[mode] = selected
result = {'selection': selection, 'inner_summaries': inputs, 'test_scores_read': False,
          'rule': '동일 내부 검증에서 R20>=baseline인 후보 중 mR20 최대', 'adaptive_campaign': True}
path = ROOT / 'combined_selection_lock.json'
if path.exists():
    assert json.loads(path.read_text()) == result
else:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(selection, ensure_ascii=False))
