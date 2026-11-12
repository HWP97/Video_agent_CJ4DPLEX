# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "pydantic", "tokenizers", "scipy"]
# ///
# How to run: main.py b3 ours predict --device cuda
"""동결 D04 또는 최신 attention의 관측 전용 실제 추론."""
from pathlib import Path
import os
import sys
import time
from typing import Literal
import numpy as np
from numpy.typing import NDArray
import torch
from common import BoundaryError, save, sha
from model import Config, Recurrent, predict
from latest_data import LATEST, checked, manifest, observations


def run(output: Path, stage: Literal["baseline", "ours"], device: str) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def observation_only(event: str, args: tuple) -> None:
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(LATEST / "reference") or path.is_relative_to(LATEST / "ground_truth"):
                raise BoundaryError("prediction_reference_or_GT_forbidden", str(path))

    sys.addaudithook(observation_only)
    forbidden_checked = []
    for folder in ("reference", "ground_truth"):
        try:
            (LATEST / folder / "negative_probe").open("rb")
        except BoundaryError as error:
            forbidden_checked.append(error.reason)
    assert len(forbidden_checked) == 2
    assets = manifest()
    positions, all_ids, receipts = [], [], []
    for fold in assets.folds:
        batch, ids = observations(fold, assets)
        weight = checked(f"models/{stage}_fold{fold.fold}.pt", assets)
        saved = torch.load(weight, map_location=device, weights_only=False)
        match stage:
            case "baseline":
                net = Recurrent(Config(**saved["config"])).to(device)
                net.load_state_dict(saved["state_dict"], strict=True)
                values = predict(net, batch, 4, device=device)
            case "ours":
                checked("source/fast_model.py", assets)
                sys.path.insert(0, str(LATEST / "source"))
                from fast_model import Predictor
                net = Predictor(125, "two_state_deep").to(device).eval()
                net.load_state_dict({key.removeprefix("base."): value for key, value in saved["state"].items()
                                     if key.startswith("base.")}, strict=True)
                pieces: list[NDArray[np.float64]] = []
                with torch.inference_mode():
                    for indices in np.array_split(np.arange(len(ids)), max(1, (len(ids) + 511) // 512)):
                        ego = torch.as_tensor(batch.ego[indices], dtype=torch.float32, device=device)
                        neighbors = torch.as_tensor(batch.neighbors[indices], dtype=torch.float32, device=device)
                        mask = torch.as_tensor(batch.mask[indices], dtype=torch.float32, device=device)
                        x = torch.cat((ego.flatten(1), (neighbors * mask[:, :, None]).flatten(1), mask), -1)
                        x = ((x - saved["center"]) / saved["spread"]).clamp(-10, 10)
                        normalized = net.continuous(x)[-1].reshape(-1, 12, 2).cpu().numpy().astype(np.float64)
                        pieces.append(batch.origin[indices, None] + normalized @ batch.rotation[indices].transpose(0, 2, 1)
                                      * batch.scale[indices, None, None])
                values = np.concatenate(pieces)
            case unexpected:
                from typing_extensions import assert_never
                assert_never(unexpected)
        assert values.shape == (len(ids), 12, 2) and np.isfinite(values).all()
        positions.append(values)
        all_ids.extend(ids)
        receipts.append({"fold": fold.fold, "n": len(ids), "checkpoint_sha256": sha(weight), "epoch": fold.epoch if stage == "ours" else None})
        print(f"fold {fold.fold}: {len(ids)} predictions", flush=True)
    assert len(all_ids) == len(set(all_ids)) == 240890
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, positions=np.concatenate(positions), sample_ids=np.array(all_ids))
    save(output / "inference.json", {"method": assets.method if stage == "ours" else "baseline_attention",
        "n": len(all_ids), "device": device, "seconds": time.monotonic() - started, "folds": receipts,
        "GT_read": False, "reference_read": False, "negative_open_checks": len(forbidden_checked),
        "api_calls": 0, "training": False, "prediction_sha256": sha(output / "predictions.npz")})
