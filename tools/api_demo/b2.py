# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pydantic", "opencv-python-headless"]
# ///
# 실행: 기존 b1 환경 Python -B tools/api_demo/b2.py [--select-only]
"""실제 VEATIC 영상 위에 확정 V/A 예측과 평가 오차를 표시한다."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
from pathlib import Path
import textwrap
from typing import Final

import cv2
import numpy as np

from b2_data import Case, HORIZONS, OUT, ROOT, VIDEO_ROOT, select_cases
from video import BLUE, GRAY, INK, RED, Frame, encode, label
from errors import DemoError
from b2_tracking import TargetTracker

FPS: Final = 25


def frames(case: Case) -> Iterator[Frame]:
    capture = cv2.VideoCapture(str(VIDEO_ROOT / f"{case.video_id}.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, round((case.cutoff_s - 8) * FPS))
    seed = json.loads((OUT / "target_boxes.json").read_text())["boxes"][case.sample_id]
    tracker = None
    try:
        for index in range(11 * FPS + 1):
            success, source = capture.read()
            if not success:
                raise DemoError(f"영상 프레임 부족: {case.sample_id}/{index}")
            elapsed = index / FPS - 8
            frame = np.full((720, 1280, 3), (24, 20, 16), dtype=np.uint8)
            scale = min(768 / source.shape[1], 470 / source.shape[0])
            width, height = round(source.shape[1] * scale), round(source.shape[0] * scale)
            image = cv2.resize(source, (width, height))
            frame[100:100 + height, 24:24 + width] = image
            phase = "OBSERVED FRAMES" if elapsed <= 0 else "FUTURE GT VIDEO / FROZEN FORECAST"
            label(frame, f"Affect Forecasting / {case.sample_id}", (24, 35))
            speed = "1x" if elapsed <= 0 else "0.5x playback"
            label(frame, f"{phase} | source t={elapsed:+.2f}s | {speed}", (24, 68), GRAY)
            if index == 8*FPS and seed is not None:
                tracker = TargetTracker(image, seed)
            face_box = tracker.update(image) if tracker is not None else None
            if face_box is not None:
                x, y, w, h = face_box
                cv2.rectangle(frame, (24+x, 100+y), (24+x+w, 100+y+h), BLUE, 2)
                label(frame, "Target ROI (display only)", (min(24+x, 500), max(93, 94+y)), BLUE)
                slots = [i for i,horizon in enumerate(HORIZONS) if horizon <= elapsed+1e-8]
                if slots:
                    va = case.ours[slots[-1]]
                    label(frame, f"V {va[0]:+.2f} A {va[1]:+.2f}", (min(24+x, 590), min(565,100+y+h+22)), BLUE)
            elif elapsed > 0:
                label(frame, "Target ROI unavailable / occluded / cut", (24, 584), GRAY)
            for offset, line in enumerate(textwrap.wrap("Target: " + case.target_description, 82)):
                label(frame, line, (24, 600 + offset*23), GRAY)
            label(frame, "Valence: negative < 0 < positive", (824, 120))
            label(frame, "Arousal: calm < 0 < activated", (824, 148))
            label(frame, "Baseline / Ours / GT", (824, 194))
            if elapsed > 0:
                visible = [i for i, h in enumerate(HORIZONS) if h <= elapsed + 1e-8]
                if visible:
                    slot = visible[-1]
                    base, ours, gt = case.baseline[slot], case.ours[slot], case.truth[slot]
                    label(frame, f"Forecast horizon +{HORIZONS[slot]:g}s", (824, 231))
                    label(frame, f"GT source PTS {case.gt_pts[slot]:.2f}s", (824, 258), GRAY)
                    for row, (name, values, color) in enumerate((("Baseline", base, GRAY), ("Ours", ours, BLUE), ("GT", gt, RED))):
                        label(frame, f"{name:8} V {values[0]:+.3f}  A {values[1]:+.3f}", (824, 305 + row*36), color)
                    error = np.abs(np.array(ours) - gt)
                    label(frame, f"Ours abs error V {error[0]:.3f} A {error[1]:.3f}", (824, 439), BLUE)
                    label(frame, "Values held until next scored horizon", (824, 474), GRAY)
                else:
                    label(frame, "First scored forecast: +0.5s", (824, 245), GRAY)
            else:
                label(frame, "Future GT hidden during observation", (824, 245), GRAY)
            if elapsed > 0:
                label(frame, f"Case MAE: baseline {case.baseline_metric:.4f} / ours {case.ours_metric:.4f}", (24, 684))
            yield frame
            if elapsed > 0:
                yield frame
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="생성한 데모 영상을 새 코드로 교체")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = select_cases()
    (OUT / "cases.json").write_text(json.dumps([c.model_dump() for c in cases], indent=2, ensure_ascii=False))
    if args.select_only:
        print(json.dumps({"selected": len(cases), "first": cases[0].sample_id}))
        return
    catalog = []
    for case in cases:
        name = f"demo_{case.rank:02d}_{case.sample_id}"
        path = OUT / f"{name}.mp4"
        if args.refresh or not path.exists():
            encode(path, frames(case), FPS)
        poster = OUT / f"{name}.jpg"
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_MSEC, 12400)
        success, image = capture.read()
        capture.release()
        if not success or not cv2.imwrite(str(poster), image):
            raise DemoError(f"포스터 생성 실패: {path}")
        catalog.append({"task": "affect_forecasting", "sample_id": case.sample_id, "rank": case.rank,
            "category": "demo", "metric": "MAE", "baseline_metric": case.baseline_metric,
            "ours_metric": case.ours_metric, "metric_delta": case.metric_delta,
            "video": str(path.relative_to(ROOT)), "poster": str(poster.relative_to(ROOT)),
            "observation_seconds": 8, "future_seconds": 3,
            "future_playback_speed": 0.5,
            "note": "관측 1x · 미래 0.5x. 시간과 지표는 원본 기준. 지정 대상 ROI는 표시 전용으로 추적하며 가림·컷에서는 중단."})
        print(json.dumps({"rendered": case.sample_id}), flush=True)
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    lines = ["# 감정 상태 예측 demo", "", "final421에서 baseline과의 MAE 차이를 기준으로 원본 그룹별 한 구간씩20개를 선택했다. 전체 성능을 대표하는 무작위 표본이 아니다.", "", "관측8초는1x, 이후3초는0.5x로 재생한다. 표시 시간과 지표는 원본 기준이다. 수치는 +0.5/+1/+2/+3초 평가점에서만 바뀌고 보간하지 않는다. 지정한 얼굴/머리 ROI는 표시 전용이며 컷·가림·추적 실패에서는 중단한다.", "", "| demo | 사례 | baseline MAE | ours MAE | 차이 | 영상 |", "|---:|---|---:|---:|---:|---|"]
    for item in catalog:
        lines.append(f"|{item['rank']}|{item['sample_id']}|{item['baseline_metric']:.6f}|{item['ours_metric']:.6f}|{item['metric_delta']:.6f}|[재생]({Path(item['video']).name})|")
    (OUT / "CATALOG.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
