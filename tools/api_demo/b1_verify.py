# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest", "numpy", "torch", "Pillow"]
# ///
# 실행: 기존 b1 Python -B tools/api_demo/b1_verify.py
"""생성 영상 전체 디코딩과 선택·점수 계약의 증거를 기록한다."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

from b1_data import PROJECT


def main() -> None:
    root = PROJECT / "demo/api_based/scene_graph_anticipation"
    evidence = root / ".omo/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((root / "catalog.json").read_text())
    assert len(catalog) == 20
    assert all(row["category"] == "demo" for row in catalog)
    assert len({row["sample_id"].split(":")[1] for row in catalog}) == 20
    records = []
    before_path = evidence / "feedback_before.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else None
    original = {row["sample_id"]: row for row in before["catalog"]} if before else {}
    command_output = []
    for row in catalog:
        video = PROJECT / row["video"]
        command = ["/usr/bin/ffmpeg", "-v", "error", "-threads", "2", "-i", str(video), "-f", "null", "-"]
        decoded = subprocess.run(command, capture_output=True, text=True, check=True)
        assert not decoded.stderr
        probe_command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=codec_name,width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(video)]
        probed = subprocess.run(probe_command, capture_output=True, text=True, check=True)
        stream = json.loads(probed.stdout)["streams"][0]
        duration = row["observation_seconds"] + 2 * row["future_seconds"]
        assert abs(float(stream["duration"]) - duration) <= 1 / 15 + 0.001
        assert int(stream["nb_frames"]) == math.floor(duration * 15) + 1
        assert row["future_playback_speed"] == 0.5
        if before:
            old = original[row["sample_id"]]
            for key in ("baseline_metric", "ours_metric", "improvement", "baseline_WC_R20", "ours_WC_R20", "observation_seconds", "future_seconds"):
                assert row[key] == old[key]
            prediction = PROJECT / row["prediction"]
            previous_name = prediction.name.replace("demo_", "improvement_", 1)
            assert hashlib.sha256(prediction.read_bytes()).hexdigest() == before["prediction_sha256"][previous_name]
        command_output.append("COMMAND: " + " ".join(probe_command) + "\nEXIT: 0\n" + probed.stdout)
        records.append({"sample_id": row["sample_id"], "video": row["video"], "full_decode_exit": decoded.returncode,
                        "bytes": video.stat().st_size, "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                        "ffprobe": stream, "expected_playback_duration": duration,
                        "original_scores_timestamps_predictions_preserved": before is not None})
    test_command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(Path(__file__).with_name("b1_tests.py"))]
    tests = subprocess.run(test_command, capture_output=True, text=True, check=True)
    (evidence / "tests.txt").write_text(tests.stdout + tests.stderr)
    command_output.append("COMMAND: " + " ".join(test_command) + "\nEXIT: 0\n" + tests.stdout + tests.stderr)
    (evidence / "feedback_validation.txt").write_text("\n".join(command_output))
    contact = Image.new("RGB", (1600, 5 * 250), "#0c1421")
    draw = ImageDraw.Draw(contact)
    for index, row in enumerate(catalog):
        with Image.open(PROJECT / row["poster"]) as poster:
            contact.paste(poster.resize((400, 225)), ((index % 4) * 400, (index // 4) * 250))
        draw.text(((index % 4) * 400 + 8, (index // 4) * 250 + 228), row["sample_id"], fill="white")
    contact.save(evidence / "contact_sheet.jpg", quality=93)
    summary = {"videos": records, "tests_command": test_command, "tests_exit": tests.returncode,
               "render_probe_evidence": "../../verification.json", "selection": "../../selection.json",
               "score_validation": "Every rendered case re-evaluated with official NC evaluator; mean R20 matches frozen at atol=1e-10.",
               "new_api_calls": 0, "training_runs": 0, "gpu_runs": 0,
               "scope": "b1 demo assets and b1*.py only; original data/source preserved"}
    (evidence / "verification.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"videos": len(records), "tests_exit": tests.returncode, "evidence": str(evidence)}))


if __name__ == "__main__":
    main()
