# API Demo

API 경로의 세 head(Scene graph · Affection · Trajectory)에 대해 Baseline · Ours · GT를 함께 시각화한 Demo 사례 catalog다. 관측은 1x, cutoff 이후 미래 구간은 0.5x로 재생한다. 시간 표시와 평가 지표는 원본 기준이다.

| head | 전체 catalog | 표시 내용 |
|---|---|---|
| Scene Graph Anticipation | [CATALOG](scene_graph_anticipation/CATALOG.md) | 관측 종료 이후의 사람·물체 관계. 공식 평가기에서 GT와 일치한 관계는 연두색 |
| Affection Forecasting | [CATALOG](affect_forecasting/CATALOG.md) | 대상 ROI 추적·Valence/Arousal 예측·GT·절대오차 |
| Trajectory Forecasting | [CATALOG](trajectory_forecasting_reselected/CATALOG.md) | 실제 원영상 위 예측(파랑)·GT(빨강)·Baseline(주황) 점과 궤적 |

최종 15개 선정 사례(head별 5개)는 [`assets/final_demo_manifest.json`](../../assets/final_demo_manifest.json)에 기록했다. README의 미리보기 GIF 6개로 Repository만으로 사례를 확인할 수 있다.

Repository에는 각 사례의 `catalog.json`·`CATALOG.md`·평가·예측 JSON만 포함한다. 원본 MP4·포스터 JPG는 용량 때문에 외부 저장소에 보관한다. 평가자 접근 방법은 [`docs/RESOURCES.md`](../../docs/RESOURCES.md)에 안내한다.

현재 실행은 저장 API 응답과 동결 후처리의 재현이며 새 유료 API 호출은 수행하지 않는다. 관계 예측의 미래 bbox는 GT 위치이며 모델의 미래 추적 결과가 아니다. Affection의 ROI는 표시용 광학 흐름 추적이며 대상 가림·컷에서 중단한다. Trajectory는 예측 대상이 좌표이므로 실제 영상 위에 점을 투영한다.
