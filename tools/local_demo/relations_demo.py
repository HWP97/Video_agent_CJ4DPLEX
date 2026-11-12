# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "opencv-python-headless", "Pillow"]
# ///
# 실행: 기존 AG Python -B tools/local_demo/relations_demo.py
"""최신 로컬 관계 예측 Demo20개를 실제 영상으로 렌더링한다."""
import argparse
import json
from pathlib import Path
import subprocess

from relations_data import PROJECT, DATA, Candidate, load_scene
from relations_render import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only',action='store_true')
    parser.add_argument('--limit',type=int,default=20)
    args = parser.parse_args()
    out = PROJECT/'demo/local_model_based/scene_graph_anticipation'
    out.mkdir(parents=True,exist_ok=True)
    rows = json.loads((DATA/'latest/demo_selection.json').read_text())
    catalog = []
    for rank,row in enumerate(rows[:args.limit],1):
        candidate = Candidate(**row)
        scene = load_scene(candidate)
        stem = f'demo_{rank:02d}_{candidate.mode}_{Path(candidate.video_id).stem}'
        video = out/f'{stem}.mp4'
        prediction = {'sample_id':candidate.sample_id,'video_path':str(scene.source_video),'observed_start_s':0.,'cutoff_s':scene.cutoff,'future_end_s':scene.end,'source':'latest local memory gate; prediction-only relation labels','baseline_relations':[list(s.baseline) for s in scene.snapshots],'ours_relations':[list(s.ours) for s in scene.snapshots],'forecast_times_s':[s.time for s in scene.snapshots],'single_cosmos_chunk_covers_future':scene.end-scene.cutoff<=5.5}
        (out/f'{stem}.prediction.json').write_text(json.dumps(prediction,indent=2))
        (out/f'{stem}.evaluation.json').write_text(json.dumps([{'pts_s':s.time,'GT_relations':s.truth,'baseline_R20':s.baseline_recall,'ours_R20':s.ours_recall,'baseline_top20_matched_GT_indices':s.baseline_matches,'ours_top20_matched_GT_indices':s.ours_matches} for s in scene.snapshots],indent=2))
        print(json.dumps({'rank':rank,'sample_id':candidate.sample_id,'cutoff_s':scene.cutoff,'future_seconds':scene.end-scene.cutoff}),flush=True)
        if args.metadata_only:
            continue
        if not video.exists():
            render(scene,video)
        webm = video.with_suffix('.webm')
        if not webm.exists():
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-y','-i',str(video),'-an','-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','8','-threads','2','-row-mt','1','-crf','33','-b:v','0',str(webm)],check=True)
        catalog.append({'branch':'scene_graph_anticipation','family':'local_model_based','sample_id':candidate.sample_id,'rank':rank,'category':'demo','metric':'NC R@20 (%)','baseline_metric':candidate.baseline*100,'ours_metric':candidate.ours*100,'metric_delta':(candidate.ours-candidate.baseline)*100,'video':str(video.relative_to(PROJECT)),'poster':str(video.with_suffix('.jpg').relative_to(PROJECT)),'prediction':str((out/f'{stem}.prediction.json').relative_to(PROJECT)),'note':'최신 로컬 ImparTail+메모리 gate. 관측1x·미래0.5x. 연두색은 공식 GT 일치, IoU=0. GT 박스는 평가용 위치다.'})
        (out/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
    if catalog:
        lines=['# 로컬 관계 예측 Demo','','동일 사례의 NC R@20 차이로 원영상 중복 없이20개를 선택했다. 모집단의 무작위 표본이 아니다. 실제 ImparTail 관측 prior에서 최신 선택 head를 재계산한 뒤 공식 평가기로 GT와 대조했다. 미래 GT는 렌더링·평가에서만 읽는다.','','|Demo|사례|Baseline|Ours|영상|','|---:|---|---:|---:|---|']
        lines.extend(f"|{r['rank']}|{r['sample_id']}|{r['baseline_metric']:.3f}|{r['ours_metric']:.3f}|[{Path(r['video']).name}]({Path(r['video']).name})|" for r in catalog)
        lines += ['','관측50%는 주석 프레임 기준이다. 원 PTS를 유지하고 미래만0.5배속으로 표시한다. 연두색 GT 일치는 공식 NC 평가기의 클래스·방향·관계·IoU=0 판정이며 GT 위치는 예측 박스가 아니다. 원영상 전달 권리는 제출 전에 별도로 확인한다.']
        (out/'CATALOG.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
