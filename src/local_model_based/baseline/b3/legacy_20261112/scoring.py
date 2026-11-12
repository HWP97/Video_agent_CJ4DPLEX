# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
# How to run: entry.py evaluate
"""공식 평가 함수의 전체 및 공통 분모 집계."""
import json
from pathlib import Path
import numpy as np
from metrics import score
from common import DATA, save

def evaluate(prediction: Path, output: Path) -> None:
    with np.load(prediction) as values:
        positions = values["positions"]
    index = json.loads((DATA / "index.json").read_text())
    common_ids = set(json.loads((DATA / "common_ids.json").read_text()))
    truths = {r["sample_id"]: r for r in map(json.loads, (DATA / "ground_truth/futures.jsonl").read_text().splitlines())}
    assert positions.shape == (len(index), 12, 2)
    rows = []
    for row, path in zip(index, positions):
        obs = {"primary_id": row["primary_id"], "future_frames": row["future_frames"],
               "tracks": [{"pedestrian_id": p} for p in row["observed_ids"]]}
        truth = truths[row["sample_id"]]
        values = path.tolist() if np.isfinite(path).all() else None
        result = score({"dataset": "trajnetplusplus", "obs": obs, "diagonal": None},
                       {"xy": [[p[2], p[3]] for p in truth["paths"][0]], "raw": truth, "cohort": "primary"}, values)
        rows.append({"sample_id": row["sample_id"], "dataset": row["dataset"], **result})
    summaries = []
    for cohort, subset in (("full1943", rows), ("common1939", [r for r in rows if r["sample_id"] in common_ids])):
        for dataset in ("all", *sorted({r["dataset"] for r in subset})):
            selected = [r for r in subset if dataset == "all" or r["dataset"] == dataset]
            valid = [r for r in selected if r["ade"] is not None]
            summaries.append({"cohort": cohort, "dataset": dataset, "requested": len(selected),
                "provided": len(valid), "missing": len(selected)-len(valid),
                "ADE": float(np.mean([r["ade"] for r in valid])),
                "FDE": float(np.mean([r["fde"] for r in valid])),
                "ColII_percent": 100*float(np.mean([r["collision_gt"] for r in valid]))})
    save(output / "scores.json", rows)
    save(output / "summary.json", summaries)

