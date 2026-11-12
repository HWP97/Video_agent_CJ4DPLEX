# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: main.py b3 ours predict
"""기존 비학습 Sol/CV 결합. 신규 API와 신규 선택·학습은 없다."""
from pathlib import Path
import numpy as np
from common import DATA, save, sha
from inference import infer
from postprocess import Inputs, Candidate, transform

def run(output: Path, device: str) -> None:
    raw = output / "raw_sol"
    raw.mkdir()
    infer(raw, False)
    with np.load(raw / "predictions.npz") as values:
        sol = values["positions"]
    with np.load(DATA / "inputs.npz") as values:
        inputs = Inputs(*(values[k] for k in Inputs.__dataclass_fields__))
    np.testing.assert_array_equal(sol, inputs.sol)
    positions = transform(inputs, Candidate("cv_blend_0.75", "cv", .75))
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, positions=positions)
    save(output / "inference.json", {"method": "cv_blend_0.75",
        "selection": "posthoc_previously_seen_validation", "requested": 1943,
        "provided": int(np.isfinite(positions).all((1, 2)).sum()), "missing": 4,
        "api_calls": 0, "new_training": 0, "collision_tradeoff": True,
        "prediction_sha256": sha(output / "predictions.npz")})

