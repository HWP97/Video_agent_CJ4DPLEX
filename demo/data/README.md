# 최종 Demo 입력 자료

확정된 API Demo 15개의 실제 원영상과 사례별 관측 정보·GT·동결 예측을 모았다. 바로가기나 심볼릭 링크가 아닌 실제 파일이다. 기존 모델을 다시 학습하거나 API를 호출하지 않았다.

| 폴더 | Demo 번호 | 자료 |
|---|---|---|
| `affect_forecasting/` | 2, 4, 6, 12, 18 | 원영상5개, cutoff·horizon, 대상 설명, 미래 GT·baseline·ours VA |
| `scene_graph_anticipation/` | 4, 6, 7, 9, 10 | 원영상5개, 추출 프레임, 프레임 시각, baseline·ours 관계와 미래 GT 평가 |
| `trajectory_forecasting/` | 1, 2, 5, 7, 13 | 원영상2개, 관측 target/neighbor tracks, 미래 GT·baseline·ours 좌표, 투영식 |

각 폴더의 `cases.json`에서 `source_video`와 사례 번호를 확인한다. 파일 경로는 이 `demo/data/`를 기준으로 한다. 각 폴더 이름은 공개 과제명이고 내부 b1/b2/b3 번호를 사용하지 않는다. Trajectory 번호는 최종 재선정 폴더 기준이다.

`manifest.json`에는 선정 목록과 원자료 용량을, `FILES_SHA256.jsonl`에는 실제 복사한 원영상·프레임의 원본 경로와 SHA를 기록한다. 원영상에는 관측 이후의 GT 화면도 있으므로 모델 추론에서는 `cases.json`의 cutoff 이후를 입력하지 않는다.

이 폴더는 데모용 원자료이며 가중치·전체 API 캐시는 support의 `runtime_bundle_20261112/` 묶음에 있다. 임의 새 원영상에 캐시가 자동 생성되는 서비스는 아니다. 이미 확정된 사례를 재현하는 자료다. 원영상 이용 조건은 그대로 적용된다.
