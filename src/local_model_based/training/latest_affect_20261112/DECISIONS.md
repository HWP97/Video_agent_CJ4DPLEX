# b2 진단 기반 실험 사전 결정

2026-11-11 실행 지시의 b2 범위만 수행한다. Python은 기존 b1 model/env 환경을 그대로 사용한다. 입력 소표본 forward만 GPU1, 보정 head는 CPU 2 threads를 사용한다. 신규 API·다운로드·패키지 변경은 없다.

seed는 20261111이다. 원래 train/dev/official_test/additional은 보존한다. 기존 inner group 검증 256개를 유지하고, 나머지 train 그룹마다 cutoff 순서 마지막 20%(올림)를 시간 검증으로 고정한다. 학습 사례는 관측 시작부터 cutoff+최대 horizon까지의 구간이 시간 검증의 관측 시작 이후와 겹치면 제외한다. 관측 시작은 manifest의 보수적인 8초 관측 시작을 쓴다. 분리와 제외 ID를 정답을 읽기 전에 등록한다. 두 검증은 같은 공통 학습 부분에서 fit한 head를 평가한다. 최종 모델은 선택 lock 후 원래 train 전체만으로 다시 fit한다.

| 조건 | 고정 입력과 구조 | 학습 target | 손실/정규화 | 학습 길이 |
|---|---|---|---|---|
| B2-D01 | 현재 인식 VA 2개+절편 선형 보정 | current GT−인식 VA | 사례 평균 제곱오차+ridge 0.01 | 해석적 1회 |
| B2-D02 | 고정 영상 특징의 마지막/시간 평균을 train-only 표준화·PCA16으로 압축하고 numeric6 결합한 선형 잔차 head | current GT−인식 VA | 사례 평균 제곱오차+ridge 0.01 | 해석적 1회 |
| B2-D03 | D02와 동일 구조·규모 | D02와 동일 | 그룹별 평균 제곱오차 균등+ridge 0.01 | 해석적 1회 |
| B2-D04 | 두 검증의 current MAE·CCC가 모두 hold 이상인 current-only 후보 중 최악 상대 current MAE가 최소인 모델을 고정하고 같은 관측 특징에 8출력 선형 delta head 추가 | future GT−current GT | 사례 평균 제곱오차+ridge 0.01 | 조건 충족 때만 해석적 1회 |

절편은 규제하지 않는다. D01~03의 보정값은 네 미래 시점에 그대로 유지한다. D02/03는 동일 PCA와 feature 표준화를 공유하므로 그룹 손실 가중만 다르다. PCA는 torch.pca_lowrank(q=20,niter=3), 첫16축, seed 고정이다. 입력/target은 float64이며 PCA 계산만 float32다. MAE/CCC 채점 전 출력은 기존과 동일하게 [-1,1]로 제한한다. epoch/규제/규모 추가 탐색은 없다.

최종 선택은 두 검증 모두 hold 대비 미래 MAE가 비증가하고 CCC가 비감소하는 후보 중 최악 상대 MAE가 최소인 후보, 없으면 hold이다. 동점은 조건 ID 오름차순이다. test 점수는 선택 lock 뒤 별도 프로세스에서만 읽고 선택을 바꾸지 않는다. test는 반복 열람 자료이며 새 독립 holdout이 아니다. backbone의 과거 개발 영역 노출은 남는다.

실제 입력 점검은 train 학습 부분·기존 inner group 검증·official_test 각각 sample ID에 seed를 붙인 SHA 순서 첫3개를 선정한다. 정답이나 오차는 보지 않는다. 각 표본의 마지막 recognition sequence 5프레임을 순차 디코딩하고 기존 JPEG→PIL RGB→/128−1→640×480 경로와 원본 VEATIC 전체 forward를 다시 실행한다. 원본은 별도 사람 crop 없이 전체 프레임을 쓰고 human은 layer4 이름이다. 출력 순서는 valence/arousal이다. cache와 atol2e-5,rtol1e-5로 대조한다. 일치하면 전처리 수정 조건을 만들지 않는다.

기존 E04는 미래 손실에 current 보조 손실을 섞었다. D02는 current-only 목표와 보정값 유지 대조를 분리하는 검증이며 새로운 원리라고 주장하지 않는다. 기존 hold·평활화·GRU의 저장 예측은 SHA/순서를 검증해 재사용한다. GRU에는 current head가 없으므로 current 지표는 해당 없음으로 표시한다. 진단용 현재값 oracle은 실제 후보 표에 넣지 않는다.
