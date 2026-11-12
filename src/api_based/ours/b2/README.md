# b2 API ours: selection236 선정 Ridge

최종 방법은 `observation_residual_lambda1`이다. train251로 과거에 학습한 동결 모델을 사용하며, selection236 MAE로 선택한 뒤 final421을 별도로 보고한다. 이전 meanpool은 final 성적을 보고 골랐던 참고 방법으로 내리고 가중치·결과·코드를 보존했다. 이번에 새 학습이나 API 요청은 하지 않았다.

| 같은 집합 | raw Sol baseline MAE / CCC | 현재 ours MAE / CCC |
|---|---:|---:|
| selection236 |0.255143 / 0.536874|0.197490 / 0.559004|
| final421 |0.222684 / 0.533652|0.191646 / 0.485849|

final MAE는13.94% 낮지만 CCC는 낮다. 기존 Ridge와 같은 모델이므로 별도 새 개선 모델을 만들었다고 세지 않는다. 평가 풀을 train251/selection236/final421로 분할한 내부 실험이며 공식 VEATIC train/test 재현이 아니다. final도 이미 여러 번 열람했으므로 새 독립 test라고 주장하지 않는다.

관측 입력은 저장 Sol 현재·미래 예측, 관측 로컬 VA와 horizon이다. 미래 GT는 채점에만 사용한다. 학습 정답은 기존 `baseline_residual_20261110/b2/labels_train.json`의251개 공개 미래 VA이며, 이 중52개가 공식 인식 test 영역이다. 고정 모델은 `data/b2/ridge/frozen_model.json`, 선택 기록은 이 폴더의 `selection236.json`이다.

계열 루트에서:
```bash
python3 main.py b2 baseline evaluate
python3 main.py b2 ours status
python3 main.py b2 ours predict --sample-id v000_t0050.00
python3 main.py b2 ours evaluate
python3 main.py b2 ours verify
```

실제 결과는 프로젝트 최상위 `results/api_based/b2/selection236_final_20261111/result.json`, baseline은 `selection236_baseline_20261111/result.json`에 있다. 결과는908개 성공/1258요청이며 기권287·대상 미확정63을 보존한다. 점수표는 같은 final421로 비교한다.

이전 meanpool 코드·설명·메타데이터·검사는 외부 `HWP_CJ4DPLEX_AI_support/reports/api_based/b2/previous_meanpool_before_selection236/`에 보존했다. 현재 `final` 명령도 `ours`의 호환 별칭이다. 로컬 b2와 b1/b3 방법은 이번 변경 대상이 아니다.
