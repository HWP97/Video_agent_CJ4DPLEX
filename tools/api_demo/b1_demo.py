# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "opencv-python-headless", "Pillow"]
# ///
# 실행: 기존 b1 환경 Python -B tools/api_demo/b1_demo.py
"""관계 예측 demo 20개의 실제 영상을 생성한다."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess

from b1_data import PROJECT, Candidate, candidates, load_scene, unique_ranked
from b1_render import FPS, render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "demo/api_based/scene_graph_anticipation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--metadata-only", action="store_true", help="영상 인코딩 없이 예측·평가 자료만 준비")
    parser.add_argument("--replace", action="store_true", help="이 도구가 만든 MP4와 poster를 새 렌더로 교체")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    if selection_path.exists():
        saved = json.loads(selection_path.read_text())
        selected = [(Candidate(**row["candidate"]), "demo", row["rank"]) for row in saved]
    else:
        pool = candidates()
        chosen = unique_ranked([c for c in pool if c.improvement > 0], 20, True)
        selected = [(c, "demo", i + 1) for i, c in enumerate(chosen)]
        assert len(selected) == 20 and len({c.video_id for c, _, _ in selected}) == 20
        selection_path.write_text(json.dumps([{"candidate": asdict(c), "category": group, "rank": rank}
                                             for c, group, rank in selected], indent=2))
    catalog = []
    verified = []
    for candidate, category, rank in selected[:args.limit]:
        print(f"Rendering demo {rank}: {candidate.sample_id}", flush=True)
        scene = load_scene(candidate)
        stem = f"{category}_{rank:02d}_{candidate.mode}_{Path(candidate.video_id).stem}"
        video = output / f"{stem}.mp4"
        prediction = {
            "sample_id": candidate.sample_id, "video_path": str(scene.source_video),
            "observed_start_s": 0.0, "cutoff_s": scene.cutoff, "future_end_s": scene.end,
            "source": "frozen macro_direct decoder; baseline/ours labels are prediction-only",
            "baseline_relations": [list(snapshot.baseline) for snapshot in scene.snapshots],
            "ours_relations": [list(snapshot.ours) for snapshot in scene.snapshots],
            "forecast_times_s": [snapshot.time for snapshot in scene.snapshots],
            "single_cosmos_chunk_covers_future": scene.end - scene.cutoff <= 5.5,
        }
        (output / f"{stem}.prediction.json").write_text(json.dumps(prediction, indent=2))
        evidence = [{"pts_s": snap.time, "baseline_R20": snap.baseline_recall,
                     "ours_R20": snap.ours_recall, "GT_relations": list(snap.truth),
                     "baseline_top20_matched_GT_indices": snap.baseline_matches,
                     "ours_top20_matched_GT_indices": snap.ours_matches} for snap in scene.snapshots]
        (output / f"{stem}.evaluation.json").write_text(json.dumps(evidence, indent=2))
        if args.metadata_only:
            continue
        if args.replace or not video.exists():
            render(scene, video)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name,width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(video)],
                               check=True, text=True, capture_output=True)
        stream = json.loads(probe.stdout)["streams"][0]
        assert (stream["width"], stream["height"], stream["codec_name"]) == (1600, 900, "h264")
        expected_duration = scene.cutoff + 2 * (scene.end - scene.cutoff)
        assert abs(float(stream["duration"]) - expected_duration) <= 1 / FPS + 0.001
        assert int(stream["nb_frames"]) == math.floor(expected_duration * FPS) + 1
        assert video.with_suffix(".jpg").is_file()
        verified.append({"sample_id": candidate.sample_id, "probe": stream,
                         "score_match_atol": 1e-10, "future_frames": len(scene.snapshots),
                         "source_cutoff_s": scene.cutoff, "source_end_s": scene.end,
                         "expected_playback_duration_s": expected_duration,
                         "observation_speed": 1.0, "future_speed": 0.5,
                         "matching": "unchanged official NC evaluator; IoU=0"})
        catalog.append({"branch": "b1", "sample_id": candidate.sample_id, "rank": rank,
            "category": category, "metric": "NC R@20 (%)", "baseline_metric": candidate.baseline * 100,
            "ours_metric": candidate.ours * 100, "improvement": candidate.improvement * 100,
            "baseline_WC_R20": candidate.baseline_wc * 100, "ours_WC_R20": candidate.ours_wc * 100,
            "video": str(video.relative_to(PROJECT)), "poster": str(video.with_suffix(".jpg").relative_to(PROJECT)),
            "observation_seconds": scene.cutoff, "future_seconds": scene.end - scene.cutoff,
            "playback_seconds": float(stream["duration"]), "future_playback_speed": 0.5,
            "note": "관측 1x·미래 0.5x 재생. 연두색 관계는 공식 NC 평가에서 GT와 일치하며 IoU=0 조건이다. GT 박스는 별도 주석 스냅샷의 평가용 위치이며 미래 위치 예측이 아님.",
            "prediction": str((output / f"{stem}.prediction.json").relative_to(PROJECT))})
        (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
        (output / "verification.json").write_text(json.dumps(verified, indent=2))
    if args.metadata_only:
        return
    lines = ["# 관계 예측 demo", "", "관측 50%에서 같은 사례의 NC R@20 차이를 기준으로 원영상 ID 중복 없이 demo 20개를 골랐다. 값은 %이며 차이는 %p다. 이미 본 평가 결과로 선별한 시연 사례이며 전체 성능의 무작위 표본이 아니다.", "", "연속 원영상과 정확한 주석 프레임의 GT 박스를 별도로 보여 준다. 관계 목록은 해당 주석 시점의 공식 NC 상위 20개 순서다. GT 관계는 공간 제약에 따라 화면 일부만 표시하고 전체는 사례 evaluation JSON에 보존한다. 파란색은 ours, 빨간색은 GT다. 박스는 평가용 실제 위치이며 미래 박스 예측으로 해석하지 않는다.", "", "|구분|순위|사례|NC baseline|NC ours|차이|WC baseline|WC ours|영상|", "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for item in catalog:
        lines.append(f"|{item['category']}|{item['rank']}|{item['sample_id']}|{item['baseline_metric']:.3f}|{item['ours_metric']:.3f}|{item['improvement']:+.3f}|{item['baseline_WC_R20']:.3f}|{item['ours_WC_R20']:.3f}|[{Path(item['video']).name}]({Path(item['video']).name})|")
    lines += ["", "## 색과 재생 속도", "", "baseline/ours 목록에서 공식 NC 평가기가 GT와 일치한다고 판정한 관계만 연두색이다. 객체 클래스·관계 방향·predicate와 기존 IoU=0 조건을 그대로 사용한다. 문자열이 같다는 이유만으로 색칠하지 않는다. 각 상위20 예측이 일치한 GT 행 번호는 evaluation JSON에 보존한다.", "", "관측은 원래 속도(1x), cutoff 이후 미래 구간만 절반 속도(0.5x)로 재생한다. 화면 Source t와 cutoff는 원본 시간이며 지표·예측·GT·Cosmos 입력은 바뀌지 않는다. 영상 길이는 cutoff + 2 × (미래 종료 − cutoff)이고 출력15fps의 한 프레임 범위 오차만 허용한다. 기존 파일을 새 렌더로 교체하려면 `--replace`를 명시한다."]
    lines += ["", "## 재현과 출처", "", "프로젝트 루트에서 기존 관계 예측 환경 Python으로 `python -B tools/api_demo/b1_demo.py`를 실행한다. 기존 영상은 덮어쓰지 않고 ffprobe로 검사한다. 새 출력을 만들려면 `--output`에 프로젝트 안의 새 디렉터리를 지정한다.", "", "선정 원본은 `results/api_based/b1/frozen/results_full_*/per_case.jsonl`의 sol/macro_direct다. 로컬 원영상·frame_records는 `/mnt/RG2/my_works/b1/benchmarks/action_genome_20261106/data/`에 있다. 기존 API 번들의 prepared 자료에서 예측 entry와 관측 feature만 decoder에 전달하고, 미래 GT는 채점·렌더링 단계에 사용한다. 20개 사례의 baseline/ours 공식 재채점 평균이 frozen 값과 절대오차 1e-10 이내인지 생성 시 검사한다.", "", "원영상 권리와 배포 허가는 별도 확인 대상이다. 로컬 시각화 제작이 공개 재배포 허가를 의미하지 않는다. [Action Genome 원 논문](https://arxiv.org/abs/1912.06992)의 사람·객체 관계 표현과 [SceneSayer](https://github.com/rohithpeddi/SceneSayer)의 미래 관계 비교를 참고했다."]
    (output / "CATALOG.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
