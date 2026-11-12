# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: 기존 Python -B prepare_latest.py (배포 자산을 최초 복사할 때만)
"""원 실험을 읽기 전용으로 보존하며 최신 보행 배포 자산을 복사한다."""
from pathlib import Path
import hashlib
import json
import shutil
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
WORK: Final = ROOT.parents[2]
SOURCE: Final = WORK / "HWP_CJ4DPLEX_AI_support/iterative_research_20261111/b3/diagnosis_targeted_experiments"
ORIGINAL: Final = WORK / "HWP_CJ4DPLEX_AI_support/official_local_train_20261111/b3"
DEST: Final = ROOT / "data/b3/latest"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
        if value is not None:
            return value.hexdigest()
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
        return result.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    records = []

    def copy(source: Path, relative: str) -> None:
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        if target.exists():
            assert digest(target) == before, str(target)
        else:
            shutil.copy2(source, target)
        assert digest(source) == digest(target) == before
        records.append({"source": str(source), "path": relative, "sha256": before})

    for stage in ("baseline", "ours"):
        folder = ROOT / stage / "b3"
        archive = folder / "legacy_20261112"
        if not archive.exists():
            archive.mkdir()
            for path in folder.iterdir():
                if path.is_file() and path.name != "prepare_latest.py":
                    shutil.copy2(path, archive / path.name)
    copy(ORIGINAL / "runplan.json", "runplan.json")
    copy(SOURCE / "aggregate.json", "reference/aggregate.json")
    plan = json.loads((ORIGINAL / "runplan.json").read_text())
    folds = []
    for fold in (1, 2, 3):
        request = json.loads((SOURCE / f"score_request_fold{fold}.json").read_text())
        lock = json.loads((SOURCE / f"selection_fold{fold}.json").read_text())
        stems = [Path(row["file"]).stem for row in plan["groups"][plan["folds"][fold - 1]["evaluation"]]]
        folds.append({"fold": fold, "stems": stems, "epoch": lock["additional_epochs"]["B3-D04"]})
        copy(SOURCE / f"selection_fold{fold}.json", f"selection/fold{fold}.json")
        copy(SOURCE / f"full/fold{fold}/B3-D04/best.pt", f"models/ours_fold{fold}.pt")
        copy(ORIGINAL / f"attention/fold{fold}/direct_unshared4.pt", f"models/baseline_fold{fold}.pt")
        for method, key in (("ours", "B3-D04"), ("baseline", "baseline_attention")):
            copy(Path(request["predictions"][key]), f"reference/{method}_fold{fold}.npy")
        for stem in stems:
            for folder, suffix in (("inputs", "_inputs.npz"), ("indices", "_index.json"), ("targets", "_labels.npz")):
                target_folder = "ground_truth" if folder == "targets" else folder
                copy(ORIGINAL / "data" / folder / (stem + suffix), f"{target_folder}/{stem}{suffix}")
            copy(ORIGINAL / "evaluation_only" / (stem + ".npz"), f"ground_truth/raw/{stem}.npz")
            copy(SOURCE / f"scores/fold{fold}/{stem}.npz", f"ground_truth/scores/{stem}.npz")
    copy(ORIGINAL / "adapters/fast/model.py", "source/fast_model.py")
    copy(ORIGINAL / "code/collision_metric.py", "source/collision_metric.py")
    manifest = {"method": "B3-D04", "selection": "latest_condition_deployment_choice_not_global_experiment_winner",
                "folds": folds, "files": records, "source_unchanged": True}
    with (DEST / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"files": len(records), "destination": str(DEST)}))


if __name__ == "__main__":
    main()
