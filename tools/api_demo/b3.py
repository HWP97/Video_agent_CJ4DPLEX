# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "pydantic", "opencv-python-headless"]
# ///
# 실행: 기존 환경 Python으로 tools/api_demo/b3.py
"""실제 영상 위의 궤적 예측 데모20개를 생성한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import cv2

from b3_data import DATA, RESULTS, ROOT, load_cases, select_cases
from b3_projection import OUTPUT, load_sources
from b3_render import write_video


def main(start_rank: int = 1) -> None:
    sources = load_sources()
    cases = load_cases()
    datasets = frozenset(source.dataset for source in sources)
    chosen = select_cases(cases, datasets, OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, case in enumerate(chosen, 1):
        source = next(s for s in sources if s.dataset == case.dataset)
        stem = f"demo_{rank:02d}_{case.sample_id}"
        video, poster = OUTPUT / f"{stem}.mp4", OUTPUT / f"{stem}.jpg"
        if rank >= start_rank:
            write_video(case, source, video)
        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 320)
            ok, frame = capture.read()
            assert ok and cv2.imwrite(str(poster), frame)
        finally:
            capture.release()
        rows.append({"branch": "trajectory_forecasting", "task": "궤적 예측",
                     "sample_id": case.sample_id, "rank": rank, "category": "demo", "metric": "ADE",
                     "baseline_metric": case.baseline_score.ade, "ours_metric": case.ours_score.ade,
                     "metric_difference": case.improvement, "video": str(video.relative_to(ROOT)),
                     "poster": str(poster.relative_to(ROOT)), "observation_seconds": 3.2, "future_seconds": 4.8,
                     "observed_playback_speed": 1., "future_playback_speed": .5, "display_duration_seconds": 12.84,
                     "source_dataset": case.dataset, "source_video": str(source.path),
                     "source_frame_start": case.observation.cutoff_frame-80+source.frame_offset,
                     "source_frame_end": case.observation.future_frames[-1]+source.frame_offset,
                     "note": "원영상별10개·인물 중복 없음. 평균/끝/최대 오차12/20/25px 이하로 선정. API 영상 구간은 일부 겹치며 보간은 표시 전용.",
                     "collision_baseline": case.baseline_score.collision_gt, "collision_ours": case.ours_score.collision_gt,
                     "fde_baseline": case.baseline_score.fde, "fde_ours": case.ours_score.fde})
        print(f"rendered demo {rank:02d} {case.sample_id}", flush=True)
    (OUTPUT / "catalog.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    eligible = sum(case.dataset in datasets for case in cases)
    lines = ["# 궤적 예측 데모", "",
             f"원영상이 확보된 students003·zara01의 동일 API 예측 {eligible}사례에서 절대 오차를 먼저 제한해 원영상별10개씩 선정했다. 원720×576에서 ours 평균/끝/최대 오차는12/20/25px 이하, GT 이동길이와 시작→끝 변위는30px 이상이며 GT와 ours 미래점은 모두 영상 안에 있다. 같은 인물은 한 번만 선택하고 두 원영상을 번갈아 배치한다. 전체 유효1,939사례의 성능은 바꾸지 않았다. 선정용 GT는 Demo 선정에만 사용한다.", "",
             "기존 API 평가의 영상 시작 범위가 students003130.0~164.8초·zara01215.2~276.0초로 제한된다.8초 영상20개를 모두 시간 비중복으로 만들 수 없어 students 시작 간격3.2초 이상, zara4.4초 이상을 적용했다. 영상 구간은 일부 겹치며20개 독립 원영상을 뜻하지 않는다. 절대 오차 기준은 완화하지 않았다. MILP 동시 선택과 모든 기준의 실제 값은 selection_manifest.json에 기록했다.", "",
             "실제720×576 원영상에 흰색 관측 경로, 주황색 baseline, 파란색 ours, 빨간색 GT 점과 궤적을 겹친다. 검출 bbox를 새로 만든 것이 아니다. 관측9점은0~3.2초, 미래12점은3.6~8초이며, 관측은1배속·미래는0.5배속이다. 마지막 프레임을 포함한 표시 파일은25fps·321프레임·12.84초다. 원 예측 배열과 평가 점수는 재생 속도에 따라 바뀌지 않는다.", "",
             "원영상은 OpenTraj 연구 배포의 UCY students03/video.avi와 zara01/video.avi다. 저자 변환의 x×0.0210·y×0.0239를 역산하고 영상 중심 원점과 위쪽이 양수인y축을 이미지 좌표로 변환한다. 영상·주석 프레임 번호는0기반으로 연결했다. cutoff와 마지막 미래 시점의20개 사례를 원영상 위에 표시해 정합을 검토했다. 표시용 선형 보간은 원래12시점 평가에 사용하지 않는다.", "",
             "Col-II는 관측에서 보였던 이웃의 미래 GT에 대해 기존 공식 평가기로 계산한 전체 horizon 충돌값이다.0은 충돌 없음,1은 충돌 있음이다. 원영상이 없는 students001을 다른 장면으로 대체하지 않았다.", "",
             "|순번|사례|baseline ADE|ours ADE|차이|FDE baseline→ours|충돌 baseline→ours|영상|",
             "|---|---|---:|---:|---:|---|---|---|"]
    for row in rows:
        lines.append(f"|{row['rank']}|{row['sample_id']}|{row['baseline_metric']:.6f}|{row['ours_metric']:.6f}|{row['metric_difference']:+.6f}|{row['fde_baseline']:.6f}→{row['fde_ours']:.6f}|{row['collision_baseline']}→{row['collision_ours']}|[MP4]({Path(row['video']).name})|")
    lines += ["", "## 재생성", "", "프로젝트 루트에서 기존 환경 Python으로 tools/api_demo의 보행 궤적 실행기를 사용한다. NumPy·Pillow·Pydantic·OpenCV, libx264 지원 FFmpeg가 필요하다. 원영상 위치와 투영 검토 자료는 source_videos.json에 있으며 PEDESTRIAN_VIDEO_MANIFEST로 다른 manifest를 지정할 수 있다. TRAJECTORY_VIDEO_ROOT로 원영상 폴더만 변경할 수 있다. 원영상 재배포 권리는 원출처 조건을 따른다.", "",
              "## 원영상 출처", "", "[students03 원영상](https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/UCY/students03/video.avi)의 SHA256은 ba91590a92ecfd7cfaea15cbace7df34edade733038508d23fdf67bb2d8f77ee, [zara01 원영상](https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/UCY/zara01/video.avi)은3bf3e856fc7bb902977b378d9f42a2d11f7f33e9cf6f679a54501fe2274c8417이다. 둘다720×576·25fps이며 각각5,405·9,031프레임이다."]
    (OUTPUT / "CATALOG.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    evidence = OUTPUT / ".omo/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    checks = []
    for row in rows:
        video = ROOT / row["video"]
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,nb_frames,r_frame_rate,codec_name,duration", "-of", "json", str(video)],
                               check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        assert (stream["width"], stream["height"], stream["nb_frames"], stream["r_frame_rate"], stream["codec_name"]) == (1600, 900, "321", "25/1", "h264")
        checks.append({"video": row["video"], "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "probe": stream})
    result = {"complete": True, "demos": len(chosen), "eligible_cases": eligible, "metric_replay_cases": len(cases),
              "new_api_calls": 0, "new_training": 0, "gpu_used": False, "videos": checks,
              "source_data": str(DATA), "source_results": str(RESULTS)}
    (evidence / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(checks), "catalog": str(OUTPUT / "catalog.json")}), flush=True)


if __name__ == "__main__":
    main()
