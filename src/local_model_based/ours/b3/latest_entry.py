# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "pydantic", "tokenizers"]
# ///
# How to run: main.py b3 baseline|ours status|replay|predict|evaluate|verify
"""최신 보행 실험의 같은 평가 범위를 연결하는 공통 진입점."""
from pathlib import Path
import argparse
import json
import time
from enum import Enum
from typing import Literal
import numpy as np
from typing_extensions import assert_never
from common import guard, save, sha
from bundle_paths import RESULTS, artifact_path
from latest_data import LATEST, checked, manifest


class Action(str, Enum):
    STATUS = "status"
    REPLAY = "replay"
    PREDICT = "predict"
    EVALUATE = "evaluate"
    VERIFY = "verify"


def main(stage: Literal["baseline", "ours"], entry: Path) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[action.value for action in Action])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    action = Action(args.action)
    guard(action in (Action.PREDICT, Action.REPLAY))
    metadata = json.loads(entry.with_name("branch.json").read_text())
    if action == Action.STATUS:
        print(json.dumps(metadata, ensure_ascii=False))
        return
    output = artifact_path(args.output) if args.output else RESULTS / "b3/latest" / f"{stage}-{action.value}-{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    assets = manifest()
    match action:
        case Action.PREDICT:
            from latest_inference import run
            run(output, stage, args.device)
        case Action.REPLAY:
            arrays, ids = [], []
            for fold in assets.folds:
                arrays.append(np.load(checked(f"reference/{stage}_fold{fold.fold}.npy", assets)))
                for stem in fold.stems:
                    ids.extend(row["sample_id"] for row in json.loads(checked(f"indices/{stem}_index.json", assets).read_text()))
            assert len(ids) == len(set(ids)) == 240890
            with (output / "predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, positions=np.concatenate(arrays), sample_ids=np.array(ids))
            save(output / "replay.json", {"n": len(ids), "new_forward": False, "prediction_sha256": sha(output / "predictions.npz")})
        case Action.EVALUATE:
            from latest_scoring import evaluate
            if args.prediction is None:
                parser.error("evaluate에는 --prediction이 필요합니다")
            evaluate(artifact_path(args.prediction), output, stage)
        case Action.VERIFY:
            for record in assets.files:
                checked(record.path, assets)
            save(output / "verification.json", {"files": len(assets.files), "hash_match": True, "symlinks": 0})
        case Action.STATUS:
            raise AssertionError("status already returned")
        case unexpected:
            assert_never(unexpected)
    print(json.dumps({"action": action.value, "output": str(output), "method": metadata["method"]}))
