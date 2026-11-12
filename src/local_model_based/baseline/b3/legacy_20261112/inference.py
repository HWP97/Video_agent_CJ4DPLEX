# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: entry.py predict --output results/b3/new-run
"""고정 attention 전수 추론. Sol 및 GT 수집은 이 모듈에 없다."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import DATA, BoundaryError, sha, save
from features import load
from model import Config, Recurrent, predict

def infer(output: Path, device: str) -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    batch = load(DATA / "validation.npz")
    index = json.loads((DATA / "index.json").read_text())
    positions = np.full((len(index), 12, 2), np.nan)
    receipts = []
    for target in ("hotel", "students", "zara"):
        ids = [i for i, row in enumerate(index) if row["group"] == target]
        weight = DATA / "models" / f"{target}_direct_unshared4.pt"
        receipt = json.loads(weight.with_suffix(".json").read_text())
        assert sha(weight) == receipt["checkpoint_sha256"]
        cfg = Config(**receipt["config"])
        checkpoint = torch.load(weight, map_location=device, weights_only=False)
        net = Recurrent(cfg).to(device)
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        positions[ids] = predict(net, batch.take(ids), 4, device=device)
        receipts.append({"fold": target, "n": len(ids), "sha256": sha(weight)})
    if positions.shape != (1943, 12, 2) or not np.isfinite(positions).all():
        raise BoundaryError("invalid_predictions")
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, positions=positions)
    save(output / "inference.json", {"seconds": time.monotonic()-started, "device": device,
        "requested": 1943, "provided": 1943, "api_calls": 0, "fits": 0,
        "input_sha256": sha(DATA / "validation.npz"), "weights": receipts,
        "prediction_sha256": sha(output / "predictions.npz")})

