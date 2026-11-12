# 감정 상태 예측: 최신 D04

현재 기본 실행은 최신 실험의 사전 선택 **D04**다. [최신 모델의 실행·성적·범위](LATEST.md)를 참조한다. 아래는 교체 이전 localpool Ridge의 역사적 기록이며 현재 기본 모델 설명이 아니다. 기존 소스와 자산, 결과는 보존했다.

## 교체 이전 localpool Ridge 기록

**로컬 방법끼리 같은 final421에서 비교한 기존 모델이다. 이미 본 final MAE 기준 사후 선택이며, 당시 selection236은 hold보다 나빴다. 독립 검증 성공이나 개발 기준 통과를 주장하지 않는다.**

| 완전 로컬 방법 | final421 MAE↓ | final421 CCC↑ |
|---|---:|---:|
| VEATIC 인식→hold baseline | 0.260499 | 0.273737 |
| 기존 direct visual linear rank2 | 0.221843 | 0.234855 |
| 기존 direct numeric tiny MLP | 0.235255 | -0.013110 |
| **ours: 기존 localpool Ridge** | **0.213456** | **0.279669** |

final은 hold보다 두 지표가 좋지만 selection236은 ours0.286671/0.171460, hold0.266225/0.309384로 둘 다 나쁘다. 전체1258 재채점은 ours0.223368/0.320824, hold0.280927/0.208860이다. 전체에는 과거 train251도 들어 있으므로 일반화 점수가 아니다. 공통908/final421은 과거 API 성공 사례에 맞춘 비교 부분집합이지만 현재 로컬 추론은 전체1258의 독립 관측 manifest를 사용하며 Sol 파일을 열지 않는다.

입력은 관측15프레임의 기존2048차원 특징과 관측 VA3개의 마지막값·기울기·표준편차6개다. PCA32의 평균32+마지막32+로컬수치6을 동결 Ridge에 넣는다. 원래80열 중 Sol 자리10열은 학습·추론 모두0이었다. 그 평균과 계수가 정확히0임을 확인하고 제거하여70열 런타임으로 분리했다. Sol 수치·Sol 교사·API 잔차 anchor는 입력에 없다. PCA와 로컬 정규화 통계만 복사했고 Sol 통계는 제외했다.

학습 자산은 `/mnt/RG2/llm_transfer_20261110/b2/localpool_model.npz`다. 목표는 사람의 공개 미래 VA GT이며 Sol 응답을 정답으로 학습하지 않았다. 다만 학습251개/20원본 그룹은 별도 제공 train이 아니라 **기존 평가 풀 개발분할을 train으로 사용한 것**이다. 정답은 `/mnt/RG2/baseline_residual_20261110/b2/labels_train.json`, 구성은 공식 인식 test 구간52개와 additional26의199개다. 과거 준비 과정의 API 성공 공통908 선별 편향도 남는다. 공식 VEATIC의 첫98영상 앞70% 사전학습과 이 head 학습은 다르다. 기존 `wave2.py`에서 만들어진 모델을 가져왔고 이번에는 학습하지 않았다.

`HWP_CJ4DPLEX_AI/src/local_model_based`에서 실행한다. 결과는 프로젝트의 `results/local_model_based/b2`에 새로 저장한다.

```bash
python3 main.py b2 baseline evaluate
python3 main.py b2 ours status
python3 main.py b2 ours predict --sample-id v000_t0050.00
python3 main.py b2 ours replay
python3 main.py b2 ours evaluate
python3 main.py b2 ours verify
```

`final`은 `ours` 별칭이다. ours predict는 저장 관측 특징→동결 head 계산이며 이미지 임베딩 재추론이 아니다. 현재 VA만으로는 visual 입력이 없어 이 head를 실행할 수 없다. 기존 `--current-va` 및 실제5JPEG 인식은 `baseline predict`에서 유지한다. full1258 결과는 새 모델 탐색·학습 없이 기존 head를 기존 관측 특징 전체에 적용한 결과다.

미래 GT·reference·Sol·API 계열 읽기 차단 상태에서1258개 추론과 기존 공통908 예측 재현을 검사한다. verify는 추가 head·PCA·특징 해시까지 확인한다. 출처·서명은 외부 `HWP_CJ4DPLEX_AI_support/reports/local_model_based/b2/ours_assets.json`에 있다. 새 API·학습·다운로드는 없다. 실제 관측은 마지막1.8초의 full-scene 특징이므로 특정인물 서비스 검증은 아니다.
