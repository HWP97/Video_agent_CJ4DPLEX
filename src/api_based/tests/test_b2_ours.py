"""selection236으로 선정한 Ridge가 원 예측을 GT 없이 재현하는지 검사한다."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ours/b2'))
import inference
from bundle_paths import RESULTS

class B2OursTests(unittest.TestCase):
    def test_selected_ridge_matches_original_908_without_GT(self):
        reference=[json.loads(line) for line in (ROOT/'data/b2/ridge/predictions.jsonl').read_text().splitlines()]
        expected={r['sample_id']:r['prediction'] for r in reference if r['method']=='observation_residual_lambda1'}
        self.assertEqual(inference.METHOD,'observation_residual_lambda1')
        def block_gt(event,args):
            if event=='open' and isinstance(args[0],str) and any(part in args[0] for part in ['/ground_truth/','/labels_','/b2/reference/']):
                raise PermissionError('test_future_GT_forbidden')
        sys.addaudithook(block_gt)
        actual=inference.predictions()
        self.assertEqual(len(actual),908)
        for row in actual:
            np.testing.assert_array_equal(row['prediction'],expected[row['sample_id']])

if __name__=='__main__':
    unittest.main()
