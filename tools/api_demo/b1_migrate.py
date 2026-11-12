# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: python tools/api_demo/b1_migrate.py
"""사용자가 요청한 관계 예측 demo 명칭과 20개 범위로 생성물만 이동한다."""
import json
from pathlib import Path
import shutil


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    old = project / "demo/api_based/b1"
    destination = project / "demo/api_based/scene_graph_anticipation"
    destination.mkdir(parents=True, exist_ok=True)
    selected = json.loads((old / "selection.json").read_text())
    selected = [{**row, "category": "demo"} for row in selected if row["category"] == "improvement"]
    assert len(selected) == 20
    moved, removed = [], []
    for source in list(old.iterdir()):
        if source.name.startswith("failure_") or ".rendering." in source.name:
            source.unlink()
            removed.append(source.name)
            continue
        target = destination / source.name.replace("improvement_", "demo_", 1)
        if target.exists():
            raise FileExistsError(target)
        shutil.move(str(source), str(target))
        moved.append(target.name)
    old.rmdir()
    (destination / "selection.json").write_text(json.dumps(selected, indent=2))
    evidence = destination / ".omo/evidence"
    for name in ("CATALOG.md", "catalog.json", "verification.json"):
        current = destination / name
        if current.exists():
            current.rename(evidence / ("before_rename_" + name))
    receipt = {"moved": moved, "removed_user_cancelled_outputs": removed,
               "destination": str(destination), "selected_count": len(selected)}
    (evidence / "migration.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
