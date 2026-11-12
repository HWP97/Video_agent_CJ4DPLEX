# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pillow", "opencv-python-headless", "pydantic"]
# ///
# How to run: 기존 환경 Python -B tools/local_demo/trajectories.py
"""최신 로컬 보행 모델의 실제 영상 Demo20개와 재생 목록을 만든다."""
from pathlib import Path
import hashlib
import json
import subprocess
import cv2
from PIL import Image, ImageDraw
from trajectory_data import ROOT, OUT, prepare
from trajectory_render import write
from b3_projection import load_sources
from b3_render import font


def main() -> None:
    cases = prepare()
    sources = {source.dataset: source for source in load_sources()}
    catalog, verification = [], []
    sheets = {"cutoff": Image.new("RGB", (1440, 1125), "white"), "future": Image.new("RGB", (1440, 1125), "white")}
    for rank, case in enumerate(cases, 1):
        source = sources[case.dataset]
        name = f"demo_{rank:02d}_{case.sample_id}"
        target, poster = OUT / (name + ".mp4"), OUT / (name + ".jpg")
        if not target.exists():
            write(case, source, target)
        capture = cv2.VideoCapture(str(target))
        try:
            for phase, frame_index in (("cutoff", 80), ("future", 320)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                assert ok
                if phase == "future":
                    assert cv2.imwrite(str(poster), frame)
                thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((360, 203))
                x, y = ((rank - 1) % 4) * 360, ((rank - 1) // 4) * 225
                sheets[phase].paste(thumb, (x, y + 22))
                ImageDraw.Draw(sheets[phase]).text((x + 4, y + 3), f"Demo {rank:02d}: {case.sample_id}", font=font(11), fill="black")
        finally:
            capture.release()
        webm = target.with_suffix(".webm")
        if not webm.exists():
            subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-i", str(target), "-an", "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-threads", "2", "-row-mt", "1", "-crf", "33", "-b:v", "0", str(webm)], check=True)
        decoded = subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], capture_output=True, check=True)
        assert not decoded.stderr
        probe = json.loads(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(target)], text=True))["streams"][0]
        assert (probe["width"], probe["height"], probe["nb_frames"], probe["r_frame_rate"]) == (1600, 900, "321", "25/1")
        catalog.append({"branch": "trajectory_forecasting", "family": "local_model_based", "task": "궤적 예측", "rank": rank,
            "category": "demo", "sample_id": case.sample_id, "metric": "ADE", "baseline_metric": case.baseline_score.ade,
            "ours_metric": case.ours_score.ade, "metric_difference": case.improvement,
            "video": str(target.relative_to(ROOT)), "poster": str(poster.relative_to(ROOT)),
            "source_video": str(source.path), "source_dataset": case.dataset,
            "source_frame_start": case.observation.cutoff_frame - 80, "source_frame_end": case.observation.future_frames[-1],
            "observation_seconds": 3.2, "future_seconds": 4.8, "observed_playback_speed": 1., "future_playback_speed": .5,
            "display_duration_seconds": 12.84, "fde_baseline": case.baseline_score.fde, "fde_ours": case.ours_score.fde,
            "collision_baseline": case.baseline_score.collision_gt, "collision_ours": case.ours_score.collision_gt,
            "note": "원영상별10개·인물과8초 구간 중복 없음. 평균/끝/최대 오차12/20/25px 이하로 선정. 로컬 attention과 D04의 실제 영상 투영."})
        verification.append({"path": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "full_decode": True, "probe": probe})
        print(f"Demo {rank:02d}/20 rendered and decoded", flush=True)
    for phase, sheet in sheets.items():
        sheet.save(OUT / f"alignment_{phase}.jpg", quality=95)
    (OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    (OUT / "verification.json").write_text(json.dumps({"demos": 20, "full_decode": 20, "videos": verification}, indent=2))
    lines = ["# 로컬 궤적 예측 Demo", "", "최신 동일3fold 평가 중 실제 UCY 원영상2개에서10사례씩 선정했다. 원720×576에서 ours 평균/끝/최대 오차12/20/25px 이하, GT 이동길이와 시작→끝 변위30px 이상, GT와 ours 미래점 모두 영상 안 조건을 먼저 적용했다. 같은 인물은 한 번만, 같은 원영상의8초 구간은 비중복으로 MILP 동시 선택했다. 두 원영상을 번갈아 배치한다. 전체 평가의 무작위 표본이 아니며 전체 성능 개선을 뜻하지 않는다. 실제 기준값은 selection_manifest.json을 따른다.", "", "관측9점·3.2초는1배속, 미래12점·4.8초는0.5배속으로 표시한다. 원영상25fps·프레임 offset0·VSP420점 최대축오차0.005 이내를 확인했다. 거리 지표는 원 좌표 단위이며 충돌은 관측 이웃 GT 기준이다.", "", "|Demo|사례|Baseline ADE|D04 ADE|영상|", "|---|---|---:|---:|---|"]
    for row in catalog:
        lines.append(f"|{row['rank']:02d}|{row['sample_id']}|{row['baseline_metric']:.6f}|{row['ours_metric']:.6f}|[MP4]({Path(row['video']).name})|")
    lines += ["", "원영상 출처와 라이선스는 기존 UCY 원영상 기록을 따른다. 예측·정답 전체는 cases.json, 투영 원주석 대조는 projection_verification.json, 영상 검증은 verification.json에 있다.", ""]
    (OUT / "CATALOG.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
