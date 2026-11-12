# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "pillow", "pydantic"]
# ///
# 실행: 기존 환경 Python으로 tools/api_demo/b3_alignment.py
"""실제 원영상과 GT 투영의 검토용 접촉표를 만든다."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from b3_data import ROOT, load_cases, select_cases
from b3_projection import OUTPUT, VideoSource, project
from errors import DemoError


def main() -> None:
    folder = Path(os.environ.get("TRAJECTORY_VIDEO_ROOT", str(ROOT.parent / "HWP_CJ4DPLEX_AI_support/submission_preparation_20261111/trajectory_videos")))
    evidence = OUTPUT / ".omo/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    sources = []
    for dataset, name in (("crowds_students003", "students003.avi"), ("crowds_zara01", "crowds_zara01.avi")):
        path = folder / name
        capture = cv2.VideoCapture(str(path))
        try:
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (width, height) != (720, 576) or abs(capture.get(cv2.CAP_PROP_FPS)-25.) > .001:
                raise DemoError(f"원영상 metadata가 다릅니다: {path}")
        finally:
            capture.release()
        sources.append(VideoSource(dataset=dataset, path=path, frame_offset=0, width=width, height=height,
                                   alignment_evidence=evidence / "alignment_cutoff.jpg"))
    cases = select_cases(load_cases(), frozenset(s.dataset for s in sources))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    records = []
    for phase, future_index in (("cutoff", -1), ("future", 11)):
        sheet = Image.new("RGB", (1440, 1540), "white")
        draw = ImageDraw.Draw(sheet)
        for i, case in enumerate(cases):
            source = next(s for s in sources if s.dataset == case.dataset)
            native_frame = case.observation.cutoff_frame
            world = case.primary_observed[-1]
            if future_index >= 0:
                native_frame = case.observation.future_frames[future_index]
                world = case.primary_truth[future_index]
            capture = cv2.VideoCapture(str(source.path))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, native_frame+source.frame_offset)
                ok, frame = capture.read()
                if not ok:
                    raise DemoError(f"원영상 프레임을 읽지 못했습니다: {case.sample_id}")
            finally:
                capture.release()
            pixel = project(world, (source.width, source.height))
            xy = tuple(np.rint(pixel).astype(int))
            cv2.circle(frame, xy, 13, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, xy, 9, (0, 0, 240), 2, cv2.LINE_AA)
            cv2.drawMarker(frame, xy, (0, 0, 240), cv2.MARKER_CROSS, 12, 1)
            x, y = (i % 4)*360, (i // 4)*308
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((360, 288))
            sheet.paste(image, (x, y+20))
            draw.text((x+5, y+3), f"{i+1:02d} {case.sample_id} frame={native_frame}", font=font, fill="black")
            records.append({"sample_id": case.sample_id, "phase": phase, "source_frame": native_frame,
                            "source_video": str(source.path), "world_xy": world.tolist(), "pixel_uv": pixel.tolist()})
        sheet.save(evidence / f"alignment_{phase}.jpg", quality=94)
    (evidence / "alignment_points.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUTPUT / "source_videos.json").write_text(json.dumps([s.model_dump(mode="json") for s in sources], indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "frames_projected": len(records), "evidence": str(evidence)}))


if __name__ == "__main__":
    main()
