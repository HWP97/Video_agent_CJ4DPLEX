"""Sol 없는 full1258 localpool 추론과 영열 제거의 동등성을 검증한다."""
import json
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ours/b2'))
spec = importlib.util.spec_from_file_location("legacy_affect_inference", ROOT/'ours/b2/inference.py')
assert spec is not None and spec.loader is not None
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)
from bundle_paths import RESULTS

class B2OursTests(unittest.TestCase):
    def test_full1258_head_without_Sol_or_GT_matches_original_shared908(self):
        reference=json.loads((RESULTS/'b2/reference/localpool_ridge.json').read_text())
        expected={r['sample_id']:r['prediction'] for r in reference}
        active = True
        def block_teacher_and_gt(event,args):
            if active and event=='open' and isinstance(args[0],str) and any(part in args[0] for part in ['/ground_truth/','/labels_','/b2/reference/','/sol/','/ridge/','/api_based/']):
                raise PermissionError('test_Sol_and_future_GT_forbidden')
        sys.addaudithook(block_teacher_and_gt)
        try:
            actual=inference.predictions()
        finally:
            active = False
        self.assertEqual(len(actual),1258)
        self.assertEqual(sum(row['sample_id'] in expected for row in actual),908)
        for row in actual:
            if row['sample_id'] in expected:
                np.testing.assert_allclose(row['prediction'],expected[row['sample_id']],rtol=0,atol=1e-6)

    def test_removed_ten_zero_columns_preserve_frozen_model(self):
        with np.load(inference.DATA/'original_pool.npz') as original, np.load(inference.DATA/'pool.npz') as stripped:
            np.testing.assert_array_equal(original['mean'][64:74],np.zeros(10))
            np.testing.assert_array_equal(original['coef'][64:74],np.zeros((10,8)))
            keep=np.r_[np.arange(64),np.arange(74,80)]
            np.testing.assert_array_equal(stripped['mean'],original['mean'][keep])
            np.testing.assert_array_equal(stripped['coef'],original['coef'][keep])
            np.testing.assert_array_equal(stripped['center'],original['center'])

if __name__=='__main__':
    unittest.main()
