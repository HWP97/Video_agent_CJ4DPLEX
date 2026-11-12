# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: 기존 Python으로 prepare_c01.py (bundle 실행 경계 밖)
"""C01 동결 자산을 복사하고 이전 hold 비교 Demo를 보존한다."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
WORKSPACE = ROOT.parents[2]
SOURCE = WORKSPACE / "HWP_CJ4DPLEX_AI_support/iterative_research_20261111/b2/followup_reasoned_combinations"
DATA = ROOT / "data/b2/latest"
TRAINING = ROOT / "training/latest_affect_20261112/c01"
EVIDENCE = WORKSPACE / ".omo/evidence/local_affect_c01_20261112"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    before = EVIDENCE / "before.json"
    if not before.exists():
        protected = [DATA / "model/B2-D02.npz", DATA / "model/B2-D04.npz", DATA / "model/transform.npz",
                     DATA / "reference/B2-D04_prediction.npz", PROJECT / "results/local_model_based/b2/latest_d04_predictions_20261112/result.json"]
        before.write_text(json.dumps({str(p): sha(p) for p in protected}, indent=2))
    pairs = []
    for name in ("checkpoint.pt", "transform.npz", "receipt.json"):
        pairs.append((SOURCE / "full/20261111/A/expert" / name, DATA / "model/c01" / name))
    pairs += [(SOURCE / "full/20261111/B2-C01/prediction.npz", DATA / "reference/B2-C01_prediction.npz"),
              (SOURCE / "full/20261111/B2-C01/receipt.json", DATA / "model/c01/combiner_receipt.json"),
              (SOURCE / "full/20261111/B2-C01/checkpoint.pt", DATA / "model/c01/combiner_checkpoint.pt"),
              (SOURCE / "scores.json", DATA / "reference/c01_scores.json")]
    for name in ("models.py", "engine.py", "run.py", "registration.json", "selection_lock.json", "DECISIONS.md", "RESULTS.md"):
        pairs.append((SOURCE / name, TRAINING / name))
    pairs.append((SOURCE.parents[1] / "followup_common.py", TRAINING / "followup_common.py"))
    records = []
    for source, destination in pairs:
        digest = sha(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha(destination) == digest, destination
        else:
            shutil.copy2(source, destination)
        assert sha(source) == sha(destination) == digest
        records.append({"source": str(source), "destination": str(destination.relative_to(ROOT)), "sha256": digest, "bytes": destination.stat().st_size})
    (DATA / "assets_c01.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    public = PROJECT / "demo/local_model_based/affect_forecasting"
    archive = WORKSPACE / "HWP_CJ4DPLEX_AI_support/submission_preparation_20261111/local_affect_hold_archive"
    if not archive.exists():
        hashes = {str(p.relative_to(public)): sha(p) for p in public.rglob("*") if p.is_file()}
        shutil.move(str(public), str(archive))
        assert all(sha(archive / p) == value for p, value in hashes.items())
        (EVIDENCE / "hold_archive_sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps({"copied_assets": len(records), "hold_archive": str(archive)}))


if __name__ == "__main__":
    main()
