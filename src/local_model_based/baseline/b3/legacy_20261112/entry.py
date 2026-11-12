# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
# How to run: family root main.py b3 baseline|final status|replay|predict|evaluate|verify
"""동결 b3 분리본 명령행."""
from pathlib import Path
from enum import Enum
from typing_extensions import assert_never
import argparse
import json
import time
import numpy as np
from common import ROOT, DATA, guard, save, sha
from bundle_paths import RESULTS, REPORTS, artifact_path

class Action(str, Enum):
    STATUS = "status"
    REPLAY = "replay"
    PREDICT = "predict"
    EVALUATE = "evaluate"
    VERIFY = "verify"

def main(entry_path: Path = Path(__file__)) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[a.value for a in Action])
    parser.add_argument("--output", type=Path, help="존재하지 않는 새 결과 디렉터리")
    parser.add_argument("--prediction", type=Path, help="evaluate가 읽을 positions NPZ")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--research-option", action="store_true", help="API만: 미채택 0.75Sol+0.25CV")
    args = parser.parse_args()
    action = Action(args.action)
    guard(action in (Action.PREDICT, Action.REPLAY))
    meta = json.loads((entry_path.parent / "branch.json").read_text())
    if action == Action.STATUS:
        print(json.dumps(meta, ensure_ascii=False))
        return
    output = artifact_path(args.output) if args.output else RESULTS / "b3" / f"{action.value}-{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    match action:
        case Action.PREDICT | Action.REPLAY:
            from inference import infer
            if meta["family"] == "api":
                infer(output, args.research_option)
            else:
                if args.research_option:
                    parser.error("local에는 research-option이 없습니다")
                if action == Action.REPLAY:
                    with np.load(RESULTS / "b3/original_predictions.npz") as values:
                        positions = values["direct_unshared4_d4"]
                    with (output / "predictions.npz").open("xb") as stream:
                        np.savez_compressed(stream, positions=positions)
                else:
                    infer(output, args.device)
        case Action.EVALUATE:
            from scoring import evaluate
            prediction = args.prediction
            if prediction is None:
                parser.error("evaluate에는 --prediction이 필요합니다")
            prediction = artifact_path(prediction)
            evaluate(prediction, output)
        case Action.VERIFY:
            manifest = json.loads((REPORTS / "b3/copy_manifest.json").read_text())
            checked = []
            for row in manifest:
                path = artifact_path(row["destination"])
                expected = row.get("installed_sha256", row["sha256"])
                assert sha(path) == expected, str(path)
                assert not path.is_symlink(), str(path)
                checked.append(row["destination"])
            save(output / "verification.json", {"files": len(checked), "hash_match": True, "symlinks": 0})
        case Action.STATUS:
            raise AssertionError("status already returned")
        case unexpected:
            assert_never(unexpected)
    print(json.dumps({"output": str(output), "action": action.value}, ensure_ascii=False))

if __name__ == "__main__":
    main()

