# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "torch"]
# ///
# How to run: main.py b3 ours status|replay|predict|evaluate|verify
"""기존 실험에서 사후 선정한 b3 방법의 독립 진입점."""
from pathlib import Path
import sys
import json
import argparse
import time
from enum import Enum
from typing_extensions import assert_never
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baseline/b3"))
from common import DATA, guard, save, sha
from bundle_paths import RESULTS, REPORTS, artifact_path

class Action(str, Enum):
    STATUS = "status"
    REPLAY = "replay"
    PREDICT = "predict"
    EVALUATE = "evaluate"
    VERIFY = "verify"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[a.value for a in Action])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    action = Action(args.action)
    metadata = json.loads(Path(__file__).with_name("branch.json").read_text())
    guard(action in (Action.PREDICT, Action.REPLAY))
    if action == Action.STATUS:
        print(json.dumps(metadata, ensure_ascii=False))
        return
    output = artifact_path(args.output) if args.output else RESULTS / "b3" / f"ours-{action.value}-{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    match action:
        case Action.PREDICT | Action.REPLAY:
            from ours_inference import run
            run(output, args.device)
        case Action.EVALUATE:
            if args.prediction is None:
                parser.error("evaluate에는 --prediction이 필요합니다")
            from scoring import evaluate
            evaluate(artifact_path(args.prediction), output)
        case Action.VERIFY:
            rows = json.loads((REPORTS / "b3/copy_manifest.json").read_text())
            rows += json.loads((DATA / "ours/manifest.json").read_text())
            for row in rows:
                path = artifact_path(row["destination"])
                assert not path.is_symlink()
                assert sha(path) == row.get("installed_sha256", row["sha256"]), str(path)
            save(output / "verification.json", {"files": len(rows), "hash_match": True})
        case Action.STATUS:
            raise AssertionError("status already returned")
        case unexpected:
            assert_never(unexpected)
    print(json.dumps({"action": action.value, "output": str(output), "method": metadata["method"]}))

if __name__ == "__main__":
    main()

