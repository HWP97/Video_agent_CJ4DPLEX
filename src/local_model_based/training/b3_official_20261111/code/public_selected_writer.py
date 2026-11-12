"""세 동결 모델의 public 예측을 평균하고 공식 writer로 검증한다."""
import json
import numpy as np
from adapter_io import ROOT, sha
from public_submission import write_submission

directory = ROOT / 'public'
lock = json.loads((ROOT / 'all_fold_selections_frozen.json').read_text())
values = []
receipts = {}
for fold in (1, 2, 3):
    selection = ROOT / 'selections' / f'fold{fold}.json'
    assert lock['folds'][selection.name] == sha(selection)
    receipt_path = directory / f'ours_fold{fold}_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    path = directory / 'predictions' / f'ours_fold{fold}.npy'
    assert sha(path) == receipt['prediction_sha256'] and sha(selection) == receipt['selection_sha256']
    assert receipt['dev_replay_256_maxdiff'] == 0 and not receipt['new_fit']
    values.append(np.load(path))
    receipts[str(fold)] = sha(receipt_path)
prediction = np.mean(values, axis=0)
assert prediction.shape == (3146, 12, 2) and np.isfinite(prediction).all()
path = directory / 'predictions/ours_selected_ensemble.npy'
assert not path.exists()
np.save(path, prediction)
submission = write_submission('ours_selected_ensemble', prediction)
result = {'samples': 3146, 'fold_receipts': receipts, 'selection_lock': sha(ROOT / 'all_fold_selections_frozen.json'),
    'prediction_sha256': sha(path), 'submission': submission, 'new_fit': False,
    'ensemble_rule': 'arithmetic mean of three dev-selected frozen fold models',
    'hidden_GT_score': None, 'API_calls': 0, 'source_sha256': sha(__file__)}
(directory / 'ours_receipt.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
