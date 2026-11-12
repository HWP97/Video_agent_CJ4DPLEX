# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pillow", "pydantic"]
# ///
# How to run: existing b1 model/env/bin/python -B tools/api_demo/b3_verify.py
"""보행 궤적 데모의 디코딩·선정 순위·공식 충돌값을 독립 확인한다."""
from __future__ import annotations

from collections import namedtuple
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from b3_data import ROOT, load_cases, select_cases
from b3_projection import OUTPUT, load_sources


def main() -> None:
    output = OUTPUT
    evidence = output / ".omo/evidence"
    cases = load_cases()
    by_id = {case.sample_id: case for case in cases}
    sources = load_sources()
    chosen = select_cases(cases, frozenset(s.dataset for s in sources))
    catalog = json.loads((output / "catalog.json").read_text())
    assert [row["sample_id"] for row in catalog] == [case.sample_id for case in chosen]
    assert len(catalog) == 20 and all(row["category"] == "demo" for row in catalog)
    metric_path = ROOT / "src/api_based/baseline/b3/official_metrics.py"
    spec = importlib.util.spec_from_file_location("b3_demo_official_metrics", metric_path)
    assert spec is not None and spec.loader is not None
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    track_type = namedtuple("Track", "frame pedestrian x y")
    collisions = []
    for row in catalog:
        case = by_id[row["sample_id"]]
        visible = {track.pedestrian_id for track in case.observation.tracks}
        neighbors = [[track_type(*point) for point in path] for path in case.truth.paths[1:]
                     if path and path[0][1] in visible]
        for prediction, score in ((case.baseline, case.baseline_score), (case.ours, case.ours_score)):
            predicted = [track_type(frame, case.observation.primary_id, *point)
                         for frame, point in zip(case.observation.future_frames, prediction)]
            actual = int(any(official.collision(predicted, neighbor) for neighbor in neighbors))
            assert actual == score.collision_gt
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-threads", "2", "-i", str(ROOT / row["video"]),
                        "-f", "null", "-"], check=True, capture_output=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(ROOT / row["video"])],
                               check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        assert (stream["width"], stream["height"], stream["nb_frames"], stream["r_frame_rate"]) == (1600, 900, "321", "25/1")
        assert abs(float(stream["duration"])-12.84) < 1e-6
        assert (ROOT / row["poster"]).stat().st_size > 0
        collisions.append({"sample_id": case.sample_id, "baseline": case.baseline_score.collision_gt,
                           "ours": case.ours_score.collision_gt, "decoded": True, "probe": stream})
    test = subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", str(Path(__file__).with_name("b3_test.py"))],
                          check=True, capture_output=True, text=True)
    (evidence / "tests.txt").write_text(test.stdout, encoding="utf-8")
    result = {"verified": True, "catalog_rows": len(catalog), "demos": len(chosen),
              "full_decode_videos": len(collisions), "official_collision_replay": collisions,
              "test_stdout": test.stdout, "lsp": "basedpyright unavailable; installation previously declined"}
    (evidence / "independent_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"verified": True, "videos": len(collisions), "cases_recomputed": len(cases)}))


if __name__ == "__main__":
    main()
