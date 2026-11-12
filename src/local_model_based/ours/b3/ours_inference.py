# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: main.py b3 ours predict --device cpu
"""기존 attention 및 동결된 train-only static residual의 실제 추론."""
from pathlib import Path
import json
import numpy as np
from common import DATA, save, sha
from features import Array, Batch, load
from inference import infer

def apply_static(batch: Batch, anchor: Array, coefficients: Array) -> Array:
    """기존 local_static_r1.0의 관측19특징과 좌표 역변환을 보존한다."""
    positions = batch.ego[:, :, :2]
    x = np.column_stack((positions.reshape(len(positions), -1), np.ones(len(positions))))
    normalized = (x @ coefficients).reshape(-1, 12, 2)
    correction = np.matmul(normalized, batch.rotation.transpose(0, 2, 1)) * batch.scale[:, None, None]
    return anchor + correction

def run(output: Path, device: str) -> None:
    anchor_dir = output / "attention"
    anchor_dir.mkdir()
    infer(anchor_dir, device)
    with np.load(anchor_dir / "predictions.npz") as values:
        positions = values["positions"]
    batch = load(DATA / "validation.npz")
    index = json.loads((DATA / "index.json").read_text())
    manifest = json.loads((DATA / "ours/manifest.json").read_text())
    for target in ("hotel", "students", "zara"):
        weight = DATA / "ours" / f"{target}_static_r1.npy"
        record = next(r for r in manifest if r["destination"] == f"data/b3/ours/{weight.name}")
        assert sha(weight) == record["installed_sha256"]
        coefficients = np.load(weight)
        assert coefficients.shape == (19, 24) and np.isfinite(coefficients).all()
        ids = [i for i, row in enumerate(index) if row["group"] == target]
        positions[ids] = apply_static(batch.take(ids), positions[ids], coefficients)
    assert positions.shape == (1943, 12, 2) and np.isfinite(positions).all()
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, positions=positions)
    save(output / "inference.json", {"method": "local_static_r1.0",
        "selection": "posthoc_previously_seen_validation", "requested": 1943, "provided": 1943,
        "missing": 0, "api_calls": 0, "new_training": 0, "device": device,
        "prediction_sha256": sha(output / "predictions.npz")})

