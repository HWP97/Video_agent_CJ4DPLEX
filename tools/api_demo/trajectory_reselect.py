# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "opencv-python-headless", "pillow", "pydantic"]
# ///
# How to run: existing model/env Python -B tools/api_demo/trajectory_reselect.py
"""기존1번을 보존하고2~20번을 제외한 API 비교 Demo19개를 새 폴더에 만든다."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from typing import Final

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d

from b3_data import ROOT, Case, load_cases
from b3_projection import OUTPUT, load_sources
from b3_render import write_video, font
from b3_source_check import splines
from trajectory_selection import passes, quality
from errors import DemoError

DEST: Final = ROOT / "demo/api_based/trajectory_forecasting_reselected"


def separation(case: Case) -> np.ndarray:
    return np.linalg.norm((case.baseline-case.ours)/[.0210, .0239], axis=1)


def select(previous: list) -> list[Case]:
    excluded = {row["sample_id"] for row in previous[1:]}
    fixed_id = previous[0]["sample_id"]
    assert fixed_id == "crowds_zara01_576" and len(excluded) == 19
    cases = load_cases()
    fixed = next(case for case in cases if case.sample_id == fixed_id)
    eligible = []
    sources = {source.dataset for source in load_sources()}
    for case in cases:
        if case.dataset not in sources or case.sample_id in excluded or case.sample_id == fixed_id:
            continue
        value = quality(case)
        pixels = case.baseline*[1/.0210, -1/.0239]+[360, 288]
        difference = separation(case)
        if (passes(value) and value.baseline_mean_px-value.mean_px > 1
                and difference.mean() >= 4 and difference.max() >= 8
                and np.all((pixels >= [0, 0]) & (pixels <= [719, 575]))):
            eligible.append(case)
    eligible.sort(key=lambda case: (-(quality(case).baseline_mean_px-quality(case).mean_px), -separation(case).mean(), case.sample_id))
    # 동일 인물의 최선 구간을 먼저 놓고, 부족한 수량만 나머지 다른 구간으로 채운다.
    selected, seen, repeated = [], {(fixed.dataset, fixed.observation.primary_id)}, []
    for case in eligible:
        key = (case.dataset, case.observation.primary_id)
        if key in seen:
            repeated.append(case)
        else:
            seen.add(key)
            selected.append(case)
    selected = (selected + repeated)[:19]
    if len(selected) != 19:
        raise DemoError(f"엄격한 GT·가시 차이 조건의 새 사례는{len(selected)}개입니다. 기준을 자동 완화하지 않습니다.")
    return [fixed, *selected]


def main() -> None:
    if DEST.exists():
        raise DemoError(f"기존 결과를 덮어쓰지 않습니다: {DEST}")
    previous = json.loads((OUTPUT / "catalog.json").read_text())
    cases = select(previous)
    DEST.mkdir(parents=True)
    prior_hashes = {str(p.relative_to(OUTPUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.rglob("*") if p.is_file()}
    (DEST / "previous_catalog.json").write_text(json.dumps(previous, ensure_ascii=False, indent=2))
    sources = {source.dataset: source for source in load_sources()}
    raw = ROOT.parent / "my_works/experiments/benchmarks/2026-11-06/001_frozen_full/b3/data/author_conversion/data/raw/crowds"
    curves = {name: splines(raw / ("students003.vsp" if name == "crowds_students003" else name+".vsp")) for name in sources}
    records, catalog, decoded = [], [], []
    sheet = Image.new("RGB", (1440, 1125), "white")
    for rank, case in enumerate(cases, 1):
        value, difference = quality(case), separation(case)
        frames = [point[0] for point in next(t for t in case.observation.tracks if t.pedestrian_id == case.observation.primary_id).points]+case.observation.future_frames
        curve = curves[case.dataset][case.observation.primary_id]
        kind = "cubic" if len(curve)>5 else "linear"
        original = np.column_stack([interp1d(curve[:, 2], curve[:, axis], kind=kind)(frames) for axis in range(2)])*[.0210, .0239]
        projection_error = float(np.max(np.abs(original-np.vstack((case.primary_observed, case.primary_truth)))))
        assert projection_error <= .005000001
        source = sources[case.dataset]
        stem = f"demo_{rank:02d}_{case.sample_id}"
        video, poster, webm = [DEST / (stem+suffix) for suffix in (".mp4", ".jpg", ".webm")]
        if rank == 1:
            for suffix in (".mp4", ".jpg", ".webm"):
                shutil.copy2(OUTPUT / (stem+suffix), DEST / (stem+suffix))
                assert hashlib.sha256((DEST/(stem+suffix)).read_bytes()).hexdigest() == prior_hashes[stem+suffix]
        else:
            write_video(case, source, video)
            subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-i", str(video), "-an", "-c:v", "libvpx-vp9",
                "-deadline", "realtime", "-cpu-used", "8", "-threads", "2", "-row-mt", "1", "-crf", "33", "-b:v", "0", str(webm)], check=True)
        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 320)
            ok, image = capture.read()
            assert ok
            if rank != 1:
                assert cv2.imwrite(str(poster), image)
            x, y = ((rank-1)%4)*360, ((rank-1)//4)*225
            sheet.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((360, 203)), (x, y+22))
            ImageDraw.Draw(sheet).text((x+4, y+3), f"Demo {rank:02d} {case.sample_id}", fill="black", font=font(11))
        finally:
            capture.release()
        check = subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], capture_output=True, check=True)
        assert not check.stderr
        probe = json.loads(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(video)], text=True))["streams"][0]
        assert (probe["width"], probe["height"], probe["nb_frames"], probe["r_frame_rate"]) == (1600, 900, "321", "25/1")
        records.append({"rank": rank, "sample_id": case.sample_id, "source": case.dataset, "primary_id": case.observation.primary_id,
            "ours_mean_error_px": value.mean_px, "ours_final_error_px": value.final_px, "ours_max_error_px": value.max_px,
            "baseline_mean_error_px": value.baseline_mean_px, "error_gain_px": value.baseline_mean_px-value.mean_px,
            "prediction_separation_mean_px": float(difference.mean()), "prediction_separation_final_px": float(difference[-1]),
            "prediction_separation_max_px": float(difference.max()), "projection_max_axis_error": projection_error,
            "ours_prediction_sha256": hashlib.sha256(case.ours.tobytes()).hexdigest(), "baseline_prediction_sha256": hashlib.sha256(case.baseline.tobytes()).hexdigest()})
        catalog.append({"branch": "trajectory_forecasting", "task": "궤적 예측", "category": "demo", "rank": rank,
            "sample_id": case.sample_id, "metric": "ADE", "baseline_metric": case.baseline_score.ade, "ours_metric": case.ours_score.ade,
            "metric_difference": case.improvement, "video": str(video.relative_to(ROOT)), "poster": str(poster.relative_to(ROOT)),
            "source_video": str(source.path), "source_dataset": case.dataset, "source_frame_start": case.observation.cutoff_frame-80,
            "source_frame_end": case.observation.future_frames[-1], "observation_seconds": 3.2, "future_seconds": 4.8,
            "observed_playback_speed": 1., "future_playback_speed": .5, "display_duration_seconds": 12.84,
            "fde_baseline": case.baseline_score.fde, "fde_ours": case.ours_score.fde,
            "collision_baseline": case.baseline_score.collision_gt, "collision_ours": case.ours_score.collision_gt,
            "note": "기존1번 유지·기존2~20번 제외. GT근접과 baseline 대비 가시 차이로 재선정. scene·시간 구간 중복 허용."})
        decoded.append({"video": str(video.relative_to(ROOT)), "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "full_decode": True, "probe": probe})
        print(f"Demo {rank:02d}/20 {'preserved' if rank==1 else 'rendered'} and decoded: {case.sample_id}", flush=True)
    assert all(hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()==digest for name,digest in prior_hashes.items())
    (DEST / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    (DEST / "selection_manifest.json").write_text(json.dumps({"fixed_first": previous[0]["sample_id"], "excluded_previous_2_to_20": [r["sample_id"] for r in previous[1:]],
        "thresholds": {"ours_mean_px_max":12,"ours_final_px_max":20,"ours_max_px_max":25,"new_error_gain_px_min_exclusive":1,"new_separation_mean_px_min":4,"new_separation_max_px_min":8},
        "scene_overlap_allowed":True,"time_overlap_allowed":True,"new_unique_people_prioritized":True,
        "unique_source_person_pairs":len({(c.dataset,c.observation.primary_id) for c in cases}), "records":records}, ensure_ascii=False, indent=2))
    (DEST / "verification.json").write_text(json.dumps({"videos":decoded,"full_decode":20,"new_videos":19,"first_preserved":True,
        "previous_directory_unchanged":True,"previous_file_sha256":prior_hashes,"projection_points_checked":420},indent=2))
    sheet.save(DEST / "alignment_future.jpg", quality=95)
    lines=["# API 궤적 예측 Demo 재선정", "", "기존1번은 파일 그대로 유지하고 기존2~20번19개 sample ID를 제외했다. GT 평균/끝/최대 오차12/20/25px 기준을 유지하면서, 새19개는 baseline보다 평균오차1px 초과 감소·두 예측의 평균거리4px 이상·최대거리8px 이상을 만족한다. 모든 값은 원720×576 기준이다.", "", "Scene과 시간 구간 중복은 허용한다. 가능한 다른 인물을 먼저 선정한 후 수량을 채웠으며20개 독립 원영상이 아니다. 관측1배속·미래0.5배속, 원 예측과 전체 평가 점수는 변경하지 않았다. 기존20개 폴더는 그대로 보존했다.", "", "|번호|사례|MP4|WebM|GT 평균오차(px)|평균 예측 간 거리(px)|", "|---|---|---|---|---:|---:|"]
    for row in records:
        stem=f"demo_{row['rank']:02d}_{row['sample_id']}"
        lines.append(f"|{row['rank']:02d}|{row['sample_id']}|[영상]({stem}.mp4)|[브라우저용]({stem}.webm)|{row['ours_mean_error_px']:.2f}|{row['prediction_separation_mean_px']:.2f}|")
    (DEST / "README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
