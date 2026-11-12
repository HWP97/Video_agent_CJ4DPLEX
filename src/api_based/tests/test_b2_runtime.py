"""실제 캐시의 오류 경계와 원본 예측 동일성을 검증한다."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ours/b2'))
sys.path.insert(0, str(ROOT / 'baseline/b2'))
import runtime

class B2CacheTests(unittest.TestCase):
    def test_cached_replay_matches_existing_raw_and_ridge(self):
        reference = [json.loads(line) for line in (ROOT / 'data/b2/ridge/predictions.jsonl').read_text().splitlines()]
        def block_gt(event, args):
            if event == 'open' and isinstance(args[0],str) and ('/ground_truth/' in args[0] or '/labels_' in args[0]):
                raise PermissionError('test_future_GT_block')
        sys.addaudithook(block_gt)
        for stage, method in [('baseline','sol_future'),('final','observation_residual_lambda1')]:
            expected = {row['sample_id']: row['prediction'] for row in reference if row['method'] == method}
            actual = runtime.predictions(stage)
            self.assertEqual(len(actual), 908)
            self.assertEqual(len(expected), 908)
            for row in actual:
                np.testing.assert_array_equal(row['prediction'], expected[row['sample_id']])

    def test_missing_response_fails_without_fallback(self):
        signature = runtime.read(runtime.DATA / 'ridge/inputs.json')[0]['source_signature']
        path = runtime.DATA / 'sol/cache' / signature / 'response.json'
        temporary = path.with_suffix('.temporarily_hidden')
        path.rename(temporary)
        try:
            with self.assertRaises(FileNotFoundError):
                runtime.cached(signature)
        finally:
            temporary.rename(path)

    def test_wrong_endpoint_rejected(self):
        signature = runtime.read(runtime.DATA / 'ridge/inputs.json')[0]['source_signature']
        path = runtime.DATA / 'sol/cache' / signature / 'request.json'
        original = path.read_bytes()
        changed = json.loads(original)
        changed['endpoint'] = 'https://example.invalid/v1/responses'
        try:
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(runtime.InvalidCache, 'endpoint_model_signature_mismatch'):
                runtime.cached(signature)
        finally:
            path.write_bytes(original)

if __name__ == '__main__':
    unittest.main()
