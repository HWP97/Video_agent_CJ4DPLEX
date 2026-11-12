"""실제 모델의 숨은 미래 차단 및 원본 연산 보존 검사."""
from pathlib import Path
import ast
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import adapter_fast as runtime
from features import Batch


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261110)
        torch.set_num_threads(2)
        self.fast = runtime.model_module('fast')
        self.masked = runtime.model_module('masked')

    def test_original_model_ast(self):
        pairs = [('/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/b3_tokens_recursive/run.py', ROOT / 'adapters/fast/model.py', {'Codec', 'fit_codec', 'Predictor', 'predict'}), ('/mnt/RG2/metric_local_followup_20261110/b3/network.py', ROOT / 'adapters/masked/model.py', {'Predictor'})]
        for original, current, names in pairs:
            trees = []
            for path in (Path(original), current):
                trees.append({node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names})
            self.assertEqual(trees[0], trees[1])

    def test_masked_hidden_future_invariance(self):
        net = self.masked.Predictor(125, 'masked').eval()
        x, y = torch.randn(8, 125), torch.randn(8, 12, 2)
        visible = torch.rand(8, 12) > .5
        mutated = torch.where(visible[..., None], y, y + 10000)
        self.assertTrue(torch.equal(net.restore(x, y, visible), net.restore(x, mutated, visible)))
        self.assertTrue(torch.equal(net.restore(x, y, torch.zeros_like(visible)), net.restore(x, mutated, torch.zeros_like(visible))))
        self.assertTrue(torch.equal(net.predict(x), net.predict(x)))

    def test_ar_causal_prefix_and_mtp_gradient(self):
        x, y = torch.randn(8, 125), torch.randn(8, 12, 2)
        net = self.masked.Predictor(125, 'mtp')
        changed = y.clone()
        changed[:, 7:] += 1000
        self.assertTrue(torch.equal(net.teacher(x, y)[0][:, :8], net.teacher(x, changed)[0][:, :8]))
        control = self.masked.Predictor(125, 'ar')
        control.load_state_dict(net.state_dict())
        net.objective(x, y, torch.ones(8)).backward()
        control.objective(x, y, torch.ones(8)).backward()
        self.assertFalse(torch.equal(net.encoder[0].weight.grad, control.encoder[0].weight.grad))
        self.assertTrue(torch.equal(net.predict(x), control.predict(x)))

    def test_codec_train_only_and_grammar(self):
        values = np.random.default_rng(9).normal(size=(32, 12, 2))
        for frequency in (False, True):
            codec = self.fast.fit_codec(values, 20., frequency)
            tokens = codec.encode(values)
            restored = codec.decode(tokens)
            self.assertEqual(restored.shape, values.shape)
            self.assertTrue(np.isfinite(restored).all())
            self.assertTrue(all(sum(len(codec.tokenizer.id_to_token(t)) for t in seq) == 24 for seq in tokens))
            self.assertTrue(np.array_equal(codec.low, np.quantile(values, .01, axis=(0, 1))))

    def test_physical_coordinate_and_anchor_transform(self):
        batch = Batch(ego=np.zeros((1, 9, 5)), neighbors=np.zeros((1, 8, 9)), mask=np.zeros((1, 8)), rotation=np.array([[[0., -1.], [1., 0.]]]), scale=np.array([2.]), origin=np.array([[10., 20.]]), kalman=np.zeros((1, 12, 2)), cv=np.zeros((1, 12, 2)))
        target = np.tile([12., 24.], (1, 12, 1))
        base = np.tile([11., 22.], (1, 12, 1))
        expected = np.tile([2., -1.], (1, 12, 1))
        self.assertTrue(np.array_equal(runtime.normalized(batch, target), expected))
        self.assertTrue(np.array_equal(runtime.world(batch, expected, base, False), target))
        self.assertTrue(np.array_equal(runtime.world(batch, expected - runtime.normalized(batch, base), base, True), target))
        self.assertTrue(np.array_equal(runtime.world(batch, np.zeros_like(expected), base, True), base))
        self.assertEqual(runtime.features(batch, base, False).shape, (1, 125))
        self.assertEqual(runtime.features(batch, base, True).shape, (1, 149))

if __name__ == '__main__':
    unittest.main(verbosity=2)
