# 로컬 관계 예측: 최신 메모리 gate와 제한 잔차

기준 모델은 ImparTail ODE다. 최신 내부 선택은 SGCLS `c04_gate_positive_cap3`, PredCls `c06_gate_positive_cap15`이며 평가 전 고정한 `data/b1/latest/combined_selection_lock.json`을 따른다. 동일한 개발 자료에서 적응적으로 탐색했으므로 새 독립 holdout 성공으로 표현하지 않는다.

관측 history4×71과 미래 간격1, ImparTail prior26의311차원에서26개 logit 잔차를 만든다. tanh와 cap3/1.5로 제한하고 prior에 더한다. 같은 영상은 제외한 학습 메모리 top8의 관계 profile을 조회하며, 간격·entropy·유사도·prior와 메모리 차이로26개 gate를 계산한다. 양성 희귀도 가중0.25는 학습 손실이며 추론 입력이 아니다.

| 관측50%·NC | Baseline R20/mR20 | Ours R20/mR20 |
|---|---:|---:|
| SGCLS |0.480268 / 0.336508|0.491097 / 0.418568|
| PredCls |0.633108 / 0.423031|0.666159 / 0.490949|

직전 유망 후보와 비교하면 R20은 SGCLS6/8·PredCls7/8 조건에서 하락하고 mR20은 두 모드8/8 조건에서 상승했다. 위 표의 baseline 대비와 직전 후보 대비를 혼동하지 않는다. 전체13,635사례의 원 평가 요약은 `data/b1/latest/summary.json`, 출처와 SHA는 `provenance.json`에 있다.

## 실행

`src/local_model_based`에서 기존 환경을 사용한다.

```bash
python3 main.py b1 ours status
python3 main.py b1 ours predict --sample predcls:GF7DV.mp4_0.5
python3 main.py b1 ours verify
python3 main.py b1 ours evaluate --sample predcls:GF7DV.mp4_0.5
```

결과는 `results/local_model_based/b1/latest/<action>_<시각>/`이다. `predict`와 `replay`는 기존 관측 compact graph·ImparTail prior를 읽어 최신 head를 실제 재계산한다. 전체 영상 backbone을 매번 새로 실행하는 명령은 아니다. sample을 지정하지 않은 predict는 첫 유효 사례이고 replay/evaluate는 전체 유효 사례다. `--limit`으로 명시적 제한이 가능하다. evaluate만 미래 GT를 읽는다. verify는 복사 자산 SHA와20개 Demo의 최신 확률을 원 실험 출력과 비교한다.

현재 관측 GT bbox/class와 미래 annotation 시각 schedule을 사용한다. 객체 기하를 관측 위치에 고정한 관계 예측이며 미래 bbox 예측이 아니다. IoU=0의 기존 공식 평가 조건을 유지한다. head forward 구간의 파일 열기를 차단하고 기존 runtime_guard를 유지한다.

## 이전 연결과 보존

교체 전 entry·branch·설명은 `legacy_20261112/`에 보존했다. 이전 ImparTail 유지 결과는 baseline과 동일하므로 `python3 main.py b1 baseline ...`로 재현한다. 기존 baseline·원본 실험·checkpoint·평가 결과는 변경하지 않았다. 최신 모델과 학습 메모리는 출처 SHA를 확인해 별도 복사했다. 신규 학습·API·모델 다운로드는 없다.

## 실제 영상 Demo

프로젝트 루트에서 기존 AG Python으로 `tools/local_demo/relations_demo.py`를 실행한다. `demo/local_model_based/scene_graph_anticipation/`에20개를 생성한다. 같은 사례의 baseline/ours NC R20 차이로 원영상 중복을 제거했다. 공식 GT 매칭은 연두색, 미래는0.5배속이다. 사례 선별은 시연 목적이며 전체 성적을 대표하는 무작위 표본이 아니다. 원영상 권리 확인은 별도 제출 조건이다.
