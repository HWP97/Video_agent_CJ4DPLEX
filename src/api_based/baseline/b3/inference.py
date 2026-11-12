# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: entry.py replay
"""원시 응답에서 좌표를 다시 추출하는 cache-only Sol 실행."""
import json
from pathlib import Path
import numpy as np
from common import DATA, ROOT, BoundaryError, save, sha

def infer(output: Path, research: bool) -> None:
    index = json.loads((DATA / "index.json").read_text())
    parsed = {row["sample_id"]: row for row in map(json.loads, (DATA / "parsed_sol.jsonl").read_text().splitlines())}
    positions = np.full((len(index), 12, 2), np.nan)
    for i, row in enumerate(index):
        if row["sample_id"] not in parsed:
            continue
        cached = parsed[row["sample_id"]]
        response = json.loads((DATA / "sol_cache" / cached["signature"] / "response.json").read_text())
        texts = [part["text"] for item in response["output"] for part in item.get("content", []) if part.get("type") == "output_text"]
        prediction = json.loads("".join(texts))["positions"]
        value = np.asarray(prediction, dtype=np.float64)
        if value.shape != (12, 2) or not np.isfinite(value).all():
            raise BoundaryError("invalid_cached_response", row["sample_id"])
        np.testing.assert_array_equal(value, cached["positions"])
        positions[i] = value
    assert int(np.isfinite(positions).all((1, 2)).sum()) == 1939
    if research:
        import sys
        sys.path.insert(0, str(ROOT / "ours/b3"))
        from postprocess import Inputs, Candidate, transform
        with np.load(DATA / "inputs.npz") as values:
            inputs = Inputs(*(values[k] for k in Inputs.__dataclass_fields__))
        np.testing.assert_array_equal(positions, inputs.sol)
        positions = transform(inputs, Candidate("cv_blend_0.75", "cv", .75))
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, positions=positions)
    save(output / "inference.json", {"requested": 1943, "provided": 1939, "missing": 4,
        "method": "research_cv_blend_0.75" if research else "raw_sol",
        "api_calls": 0, "fits": 0, "research_option": research, "default_changed": False,
        "prediction_sha256": sha(output / "predictions.npz")})

