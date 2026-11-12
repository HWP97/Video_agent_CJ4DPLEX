# Local Model Demo

로컬 경로의 최신 동결 모델을 실제 원영상과 함께 보여준다. API Demo와는 다른 모델·평가집합이며 같은 계열 안의 Baseline / Ours만 비교한다.

| head | Baseline → Ours | catalog |
|---|---|---|
| Scene Graph Anticipation | ImparTail ODE → 메모리 gate · 보정 head | [CATALOG](scene_graph_anticipation/CATALOG.md) |
| Affect Forecasting | C01 GRU → D02 현재 보정 + D04 미래 변화량 | [CATALOG](affect_forecasting/CATALOG.md) |
| Trajectory Forecasting | Attention → F 두 상태 반복 D04 | [CATALOG](trajectory_forecasting/CATALOG.md) |

공개 분류는 Demo만 사용한다. 지표 차이와 실제 원영상 가용성을 기준으로 선정한 사례이며 무작위 평가 표본이 아니다. 성능 회복에 실패한 실험 결과는 Technical Report에 그대로 기록한다.

관측은 1x, 미래는 0.5x이며 화면의 시간·지표는 원본 기준이다. 관계의 공식 GT 매칭은 연두색, 보행 예측은 파랑·GT는 빨강으로 표시한다. Affect은 실제 입력 특징의 마지막 1.8초를 관측으로 표시하며 미래 +0.5/+1/+2/+3초의 값을 비교한다. 대상 가림·불확실성은 없는 얼굴 상자를 꾸며서 채우지 않는다.

Repository에는 catalog·평가 metadata만 포함하며 원본 MP4는 외부 저장소에 보관한다. 상세 모델 구성은 [로컬 Agent 안내](../../src/local_model_based/README.md)와 [Technical Report 9장](../../docs/TECHNICAL_REPORT.md)을 따른다.
