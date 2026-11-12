# 감정 상태 예측: 최신 D04 실행

사용자 지시에 따라 ours는 최신 실험의 사전 선택 D04, 기본 baseline은 **C01 소형 GRU 직접 미래 예측**을 사용한다. hold는 추가 비교군으로만 남긴다. D04는 관측 영상 요약 PCA16과 관측 수치6개를 입력으로 사용한다. D02의 현재 감정 잔차를 인식 VA에 더하고, D04의 네 미래 시점 변화량을 추가한다. 새 학습이나 API 호출은 없다.

| 동일 official_test438 | 미래 MAE↓ | 미래 CCC↑ |
|---|---:|---:|
| Baseline C01 직접 GRU |0.2216956917|0.5658295809|
| D04 |0.2188385454|0.5773870911|
| 추가 비교: 로컬 인식 hold |0.2201507492|0.6130040170|

C01 대비 D04는 MAE가 낮고 CCC가 높지만, hold 대비로는 MAE만 낮아지고 CCC는 악화했다. 최신 진단 실험의 hold 기준 공동 개선 실패와 기존 GRU·평활화 성적 미회복을 그대로 보고한다. D04 사전 선택은 두 내부 검증을 사용했다. C01을 기본 비교군으로 사용한 것은 이번 사용자 결정이며 D04와 같은 실험에서 새로 사전 선택됐다고 주장하지 않는다. 평가 영역은 반복 열람했으므로 새로운 독립 holdout으로 부르지 않는다.

## C01 직접 미래 예측

기존 동결 PCA32에 상대시각1개를 붙인15×33 입력을 GRU(33,8)에 넣는다. 마지막 state8과 관측 통계6개를 Linear(14,16)로 투영한 뒤 네 horizon의 학습 embedding을 더하고 tanh→Linear(16,2)로 네 미래 VA를 직접 출력한다. 현재 VA head는 없으므로 current 지표는 null/해당 없음이다. hold 현재값의 점수를 C01 출력으로 사용하지 않는다.

추론 경로는1,370개 파라미터다. 원 체크포인트 전체1,978개 중608개는 C01에서 사용하지 않는 MTP용 보조 모듈이다. C01은 mtp=False, seed20261111, 원train1169에서20epoch 학습한 동결 모델이다. fixed_gate1·a1·b0으로 expert 출력에 범위 제한만 적용하며 학습 gate나 현재값과의 가중 혼합은 없다. 원 실험의 float32 덧셈·뺄셈 순서는 수치 재현을 위해 유지했다.

## 입력과 실행 범위

동결 관측 특징2,740개를 실제 행렬 계산으로 다시 추론한다. train1169/dev265/official_test438/additional868을 구분하며 전체를 독립 test로 합산하지 않는다. 실제 특징은 마지막1.8초의15프레임, 로컬 VA는 각각5프레임을 사용하는 세 관측 시퀀스다. manifest의8초 보수적 관측 범위와 특징 프레임 범위는 다르다. 임의 새 영상에서 backbone을 다시 실행하는 명령은 아니다.

`predict`와 `replay` 모두 동결 관측 특징에서 재계산하며 저장 예측을 답으로 읽지 않는다. 추론 중 GT와 reference 열기를 차단한다. `evaluate`만 추론 완료 후 GT를 읽어 [-1,1] 제한을 적용한 기존 MAE·8채널 CCC를 계산한다. `verify`는 자산 SHA와 선택한 method의2,740개 원예측 일치를 확인한다.

```bash
cd /mnt/RG2/HWP_CJ4DPLEX_AI/src/local_model_based
python3 main.py b2 baseline predict
python3 main.py b2 ours predict
python3 main.py b2 ours predict --sample-id v086_official_test_t00066.3200
python3 main.py b2 baseline evaluate
python3 main.py b2 ours evaluate
python3 main.py b2 baseline verify
python3 main.py b2 ours verify
python3 main.py b2 baseline evaluate --method baseline_hold
```

기존 실험과 같은 호환 model/env Python을 branch.json에 지정했다. `data/b2/latest/assets.json`과 `assets_c01.json`에 복사 자산의 원본 경로·SHA·크기를 기록했다. API 원본 자료를 추론에서 읽지 않으며 공통 runtime_guard를 완화하지 않았다.

학습·선택의 원문 소스와 분할·등록 기록은 `data/b2/latest/training_source`에 SHA와 함께 보존했다. 이 사본은 원래 경로 구조를 사용하는 연구 코드이며 배포 진입점이 아니다. 제출 준비에서 재학습은 실행하지 않았다. D04 배포 추론은 이 사본을 import하지 않고 동결 변환과 두 head만 읽는다.

C01 학습 소스·등록·선택 기록은 `training/latest_affect_20261112/c01`에 원문 그대로 있다. C01 추론은 이 보존본의 `models.py`에서 원 Expert 구조만 가져오고 학습 실행기는 호출하지 않는다. 이전 hold 비교 Demo는 `HWP_CJ4DPLEX_AI_support/submission_preparation_20261111/local_affect_hold_archive`로 SHA 보존 이동했다. 공개 Demo20개는 C01/D04 비교로 다시 선정했다.

## 이전 모델 보존

`inference.py`와 기존 `data/b2/ours`는 그대로 보존했다. 교체 전 entry·branch.json·README는 `legacy_localpool`에 보존했다. 기존 결과 폴더도 수정하지 않았다. 이전 모델은 다음과 같이 직접 재현한다.

```bash
/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B runtime_guard.py ours/b2/legacy_entry.py evaluate
/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B runtime_guard.py baseline/b2/legacy_entry.py evaluate
```

이전 baseline의 `predict --recognize` 이미지 인식도 legacy_entry 경로에 남아 있다. 원영상 픽셀 경로의 환경 차이는 최신 실험 RESULTS.md에 기록되어 있고 이번 연결 작업에서 다시 해결했다고 주장하지 않는다.
