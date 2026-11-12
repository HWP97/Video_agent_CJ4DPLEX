# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "pydantic", "Pillow"]
# ///
# 실행: 기존 model/env/bin/python -B tools/local_demo/affect.py --preview 또는 --render
"""C01/D04의 official_test에서 실제 특징 관측1.8초와 미래3초를 보여준다."""
from __future__ import annotations
import argparse
from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import cv2
import numpy as np

ROOT: Final = Path(__file__).resolve().parents[2]
OUT: Final = ROOT / "demo/local_model_based/affect_forecasting"
DATA: Final = ROOT / "src/local_model_based/data/b2/latest"
sys.path.insert(0, str(ROOT / "tools/api_demo"))
from b2_data import Case, HORIZONS, VIDEO_ROOT
from b2_tracking import TargetTracker
from errors import DemoError
from video import BLUE, GRAY, RED, Frame, encode, label

FPS: Final = 25
OBSERVATION: Final = 1.8


def select_cases() -> list[Case]:
    """같은 official_test 사례의 MAE 차이로 원본 그룹당 한 구간을 고른다."""
    rows = json.loads((DATA / "manifest.json").read_text())
    truth = {r["sample_id"]: r for r in json.loads((DATA / "ground_truth/labels_eval.json").read_text())}
    ours = json.loads((ROOT / "results/local_model_based/b2/latest_d04_predictions_20261112/result.json").read_text())["predictions"]
    baseline_rows = json.loads((ROOT / "results/local_model_based/b2/latest_c01_predictions_20261112/result.json").read_text())["predictions"]
    descriptions = {r["video_id"]: r["target_description"] for r in json.loads((ROOT / "src/api_based/data/b2/inputs/identity_review/descriptions.json").read_text())}
    candidates = []
    for row, a, b in zip(rows, ours, baseline_rows, strict=True):
        assert row["sample_id"] == a["sample_id"] == b["sample_id"]
        if row["split"] != "official_test":
            continue
        assert abs(row["cutoff_s"] - min(row["frame_pts_s"]) - OBSERVATION) < 1e-8
        y = truth[row["sample_id"]]
        p, baseline = np.clip(a["prediction"], -1, 1), np.clip(b["prediction"], -1, 1)
        ma, mb = float(abs(p - y["future"]).mean()), float(abs(baseline - y["future"]).mean())
        candidates.append(Case(sample_id=row["sample_id"], video_id=row["video_id"], group=row["group"], rank=0,
            cutoff_s=row["cutoff_s"], baseline=baseline.tolist(), ours=p.tolist(), truth=y["future"], gt_pts=y["target_pts_s"],
            baseline_metric=mb, ours_metric=ma, metric_delta=mb-ma, target_description=descriptions.get(row["video_id"]) or "Annotation target not specified"))
    chosen, used = [], set()
    for case in sorted(candidates, key=lambda c: (-c.metric_delta, c.sample_id)):
        if case.group in used:
            continue
        used.add(case.group)
        chosen.append(case.model_copy(update={"rank": len(chosen)+1}))
        if len(chosen) == 20:
            break
    assert len(chosen) == 20
    return chosen


def frames(case: Case) -> Iterator[Frame]:
    """관측 특징 시작 시각부터 실제 원영상과 확정 예측을 렌더링한다."""
    capture = cv2.VideoCapture(str(VIDEO_ROOT / f"{case.video_id}.mp4"))
    assert abs(capture.get(cv2.CAP_PROP_FPS) - FPS) < .01
    capture.set(cv2.CAP_PROP_POS_FRAMES, round((case.cutoff_s - OBSERVATION) * FPS))
    seed = json.loads((OUT / "target_boxes.json").read_text())["boxes"][case.sample_id]
    tracker = None
    try:
        for index in range(round((OBSERVATION + 3) * FPS) + 1):
            ok, source = capture.read()
            if not ok:
                raise DemoError(f"원영상 부족: {case.sample_id}/{index}")
            elapsed = index / FPS - OBSERVATION
            frame = np.full((720, 1280, 3), (24, 20, 16), dtype=np.uint8)
            scale = min(768 / source.shape[1], 470 / source.shape[0])
            image = cv2.resize(source, (round(source.shape[1]*scale), round(source.shape[0]*scale)))
            frame[100:100+image.shape[0], 24:24+image.shape[1]] = image
            label(frame, f"Affect Forecasting / {case.sample_id}", (24, 35))
            phase = "OBSERVED FRAMES / 1x" if elapsed <= 1e-8 else "FUTURE GT VIDEO / 0.5x playback"
            label(frame, f"{phase} | source t={elapsed:+.2f}s", (24, 68), GRAY)
            if index == round(OBSERVATION*FPS) and seed is not None:
                tracker = TargetTracker(image, seed)
            box = tracker.update(image) if tracker is not None else None
            slots = [i for i, h in enumerate(HORIZONS) if h <= elapsed+1e-8]
            if box is not None:
                x,y,w,h = box
                cv2.rectangle(frame, (24+x,100+y), (24+x+w,100+y+h), BLUE, 2)
                label(frame, "Target ROI (display only)", (min(24+x,500),max(93,94+y)), BLUE)
                if slots:
                    v,a = case.ours[slots[-1]]
                    label(frame, f"V {v:+.2f} A {a:+.2f}", (min(24+x,590),min(565,100+y+h+22)), BLUE)
            elif elapsed > 0:
                label(frame, "Target ROI unavailable / occluded / cut", (24,584), GRAY)
            label(frame, "Valence: negative < 0 < positive", (824,120))
            label(frame, "Arousal: calm < 0 < activated", (824,148))
            label(frame, "Baseline: C01 direct GRU forecast", (824,194), GRAY)
            label(frame, "Ours: D04 current + future delta", (824,225), BLUE)
            if slots:
                slot = slots[-1]
                label(frame, f"Horizon +{HORIZONS[slot]:g}s / PTS {case.gt_pts[slot]:.2f}s", (824,268))
                for row,(name,values,color) in enumerate((("Baseline",case.baseline[slot],GRAY),("Ours",case.ours[slot],BLUE),("GT",case.truth[slot],RED))):
                    label(frame,f"{name:8} V {values[0]:+.3f} A {values[1]:+.3f}",(824,315+row*36),color)
                error = abs(np.array(case.ours[slot])-case.truth[slot])
                label(frame,f"Abs error V {error[0]:.3f} A {error[1]:.3f}",(824,455),BLUE)
            else:
                label(frame,"Future values hidden until horizon",(824,300),GRAY)
            label(frame,"Model input: 15 observed feature frames across 1.8s; full-scene recognition",(24,620),GRAY)
            label(frame,"Values clipped to [-1, 1], matching evaluation; source time unchanged",(24,650),GRAY)
            if elapsed > 0:
                label(frame,f"Case MAE: baseline {case.baseline_metric:.4f} / ours {case.ours_metric:.4f}",(24,684))
            yield frame
            if elapsed > 1e-8:
                yield frame
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = select_cases()
    (OUT / "cases.json").write_text(json.dumps([c.model_dump() for c in cases], ensure_ascii=False, indent=2))
    if args.preview:
        import b2_preview
        b2_preview.OUT = OUT
        b2_preview.main()
    if not args.render:
        print(json.dumps({"selected": len(cases), "first": cases[0].sample_id})); return
    catalog = []
    for case in cases:
        name = f"demo_{case.rank:02d}_{case.sample_id}"
        path, poster = OUT / f"{name}.mp4", OUT / f"{name}.jpg"
        if not path.exists():
            encode(path, frames(case), FPS)
        cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, 7000)
        ok, frame = cap.read(); cap.release()
        if not ok or not cv2.imwrite(str(poster), frame):
            raise DemoError(f"포스터 실패: {path}")
        catalog.append({"task":"affect_forecasting","sample_id":case.sample_id,"rank":case.rank,"category":"demo",
            "baseline_method":"B2-C01","ours_method":"B2-D04",
            "metric":"MAE","baseline_metric":case.baseline_metric,"ours_metric":case.ours_metric,"metric_delta":case.metric_delta,
            "video":str(path.relative_to(ROOT)),"poster":str(poster.relative_to(ROOT)),"observation_seconds":OBSERVATION,
            "future_seconds":3,"future_playback_speed":.5,"note":"최신 official_test438 중 원본 그룹당1개. 실제 특징 관측1.8초·미래3초, 미래0.5x. C01 직접GRU / D04, 평가지표와 같은 범위 제한."})
        print(json.dumps({"rendered":case.sample_id}),flush=True)
    (OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    sources = [DATA/"manifest.json",DATA/"model/B2-D02.npz",DATA/"model/B2-D04.npz",DATA/"model/c01/checkpoint.pt",
        DATA/"model/c01/transform.npz",DATA/"ground_truth/labels_eval.json",OUT/"cases.json",
        ROOT/"results/local_model_based/b2/latest_c01_predictions_20261112/result.json",
        ROOT/"results/local_model_based/b2/latest_d04_predictions_20261112/result.json"]
    (OUT / "provenance.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))


if __name__ == "__main__":
    main()
