"""GT와 API 없이 full1258 hold 및 현재 VA 입력 동작을 검사한다."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baseline/b2'))
import runtime
from bundle_paths import RESULTS

class B2LocalTests(unittest.TestCase):
    def test_hold_from_current_va_clips_and_repeats(self):
        np.testing.assert_array_equal(runtime.hold([1.2,-0.3]), [[1.,-0.3]] * 4)

    def test_nonfinite_current_va_rejected(self):
        with self.assertRaises(runtime.InvalidCache):
            runtime.hold([float('nan'),0.])

    def test_full1258_replay_matches_existing_hold_without_gt(self):
        reference = json.loads((RESULTS / 'b2/reference/local_forecast.json').read_text())
        expected = {(r['sample_id'],r['horizon_s'],r['axis']):r['prediction'] for r in reference if r['method']=='recognition_last_state'}
        active = True
        def block_gt(event, args):
            if active and event == 'open' and isinstance(args[0],str) and ('/ground_truth/' in args[0] or '/b2/reference/' in args[0]):
                raise PermissionError('test_future_GT_block')
        sys.addaudithook(block_gt)
        try:
            actual = runtime.predictions('final')
        finally:
            active = False
        self.assertEqual(len(actual),1258)
        for row in actual:
            target = [[expected[(row['sample_id'],h,a)] for a in ['valence','arousal']] for h in [.5,1.,2.,3.]]
            np.testing.assert_array_equal(row['prediction'],target)

if __name__ == '__main__':
    unittest.main()
