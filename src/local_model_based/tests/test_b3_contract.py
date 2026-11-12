# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest", "numpy"]
# ///
# How to run: existing environment python -m pytest tests/test_b3_contract.py
"""추론 GT/네트워크 경계와 실제 관측 토큰화 계약."""
from pathlib import Path
import subprocess
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline/b3"))
from features import tokenize, load
from data import Track, Observation
from features import Batch
sys.path.insert(0, str(ROOT / "ours/b3"))
from ours_inference import apply_static

def test_future_truth_and_network_blocked_without_key() -> None:
    code = """
import sys, pathlib, socket, os
sys.path.insert(0, sys.argv[1])
from common import guard, DATA, BoundaryError
os.environ['SOL_API_KEY']='test-only'
guard(True)
assert 'SOL_API_KEY' not in os.environ
for fn in (lambda: (DATA/'ground_truth/futures.jsonl').open(), lambda: socket.getaddrinfo('localhost',80)):
    try: fn()
    except BoundaryError: pass
    else: raise AssertionError('boundary allowed forbidden operation')
print('both_blocked')
"""
    run = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "baseline/b3")],
        capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "both_blocked"

def test_raw_observations_reproduce_stored_features() -> None:
    rows = list(map(json.loads, (ROOT / "data/b3/observations.jsonl").read_text().splitlines()))
    cache = load(ROOT / "data/b3/validation.npz")
    for i in (0, 19, 1942):
        row = rows[i]
        tracks = tuple(Track(t["pedestrian_id"], np.asarray(t["points"])[:, 1],
                            np.asarray(t["points"])[:, 2:]) for t in row["tracks"])
        primary = next(t for t in tracks if t.pedestrian == row["primary_id"])
        tokens = tokenize(Observation(row["sample_id"], row["dataset"], row["primary_id"],
                            (), primary.xy, tracks))
        for field in tokens.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(tokens, field), getattr(cache, field)[i])

def test_static_residual_uses_observation_rotation_and_scale() -> None:
    ego = np.zeros((1, 9, 5))
    ego[0, 0, 0] = 3.
    batch = Batch(ego=ego, neighbors=np.zeros((1, 8, 9)), mask=np.zeros((1, 8)),
        rotation=np.array([[[0., -1.], [1., 0.]]]), scale=np.array([2.]),
        origin=np.zeros((1, 2)), kalman=np.zeros((1, 12, 2)), cv=np.zeros((1, 12, 2)))
    coefficients = np.zeros((19, 24))
    coefficients[0, ::2] = 2.
    coefficients[-1, 1::2] = 4.
    anchor = np.tile([10., 20.], (1, 12, 1))
    actual = apply_static(batch, anchor, coefficients)
    np.testing.assert_array_equal(actual, np.tile([2., 32.], (1, 12, 1)))

