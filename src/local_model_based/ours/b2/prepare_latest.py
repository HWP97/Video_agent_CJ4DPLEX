# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: 기존 Python으로 prepare_latest.py (bundle 실행 경계 밖)
"""완료된 실험 자산을 읽기 전용 원본에서 독립 복사한다."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[2]
SOURCE = WORKSPACE / "HWP_CJ4DPLEX_AI_support/official_local_train_20261111/b2/attempt_fp32"
EXPERIMENT = WORKSPACE / "HWP_CJ4DPLEX_AI_support/iterative_research_20261111/b2/diagnosis_targeted_experiments"
DESTINATION = ROOT / "data/b2/latest"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def main() -> None:
    files = [(SOURCE / name, name) for name in ("inputs.npz", "manifest.json")]
    files += [(SOURCE / f"labels_{name}.json", f"ground_truth/labels_{name}.json") for name in ("train", "dev", "eval")]
    files += [(EXPERIMENT / "full" / name, f"model/{name}") for name in ("transform.npz", "B2-D02.npz", "B2-D04.npz")]
    files += [(EXPERIMENT / name, f"reference/{name}") for name in ("B2-D04_prediction.npz", "scores.json")]
    files += [(EXPERIMENT / name, name) for name in ("selection_lock.json", "training_receipt.json", "inference_receipt.json", "RESULTS.md")]
    files += [(EXPERIMENT / name, f"training_source/{name}") for name in (
        "train.py", "head.py", "prepare.py", "infer.py", "score.py", "DECISIONS.md",
        "registration.json", "split.json", "training_protocol_lock.json")]
    files.append((EXPERIMENT.parents[1] / "followup_common.py", "training_source/followup_common.py"))
    records = []
    for source, relative in files:
        before = digest(source)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == before, target
        else:
            shutil.copy2(source, target)
        assert digest(source) == before == digest(target)
        records.append({"source": str(source), "destination": relative, "sha256": before, "bytes": target.stat().st_size})
    (DESTINATION / "assets.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps({"copied_files": len(records), "bytes": sum(r["bytes"] for r in records)}))


if __name__ == "__main__":
    main()
