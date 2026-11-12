# 감정 상태 예측의 이전 실험 비교와 학습 정답 출처

이 문서는 **이전251개 개발 학습 자료와 localpool 등 과거 실험의 역사적 기록**이다. 현재 로컬 기본 모델은 사용자 지정 C01 직접 미래 GRU baseline과 사전 선택 D04 ours이며, 두 모델의 최종 학습 자료는 원train1169다. 현재 모델·평가집합·한계는 [LATEST.md](LATEST.md)를 따른다. 아래 meanpool의 ours와251개 학습 설명을 현재 C01/D04의 설명으로 읽지 않는다. 원실험 수치와 본문은 보존한다.

API 최종 ours는2026-11-11 사용자 지시에 따라 selection236으로 선정한 observation residual Ridge로 변경했다. 최신 수치는 `docs/TECHNICAL_REPORT.md` 4장을 따른다.

**사용자가 별도 train set을 주지 않았는데도 과거 b2 소형 모델은 학습되어 있었다. 기존 평가 풀의 개발 분할251개를 train으로 사용한 것이다. 공식 VEATIC train 분할을 새로 확보해 학습한 결과가 아니다. 이번 ours 연결 작업에서는 그 기존 가중치만 읽었고 새 학습은0회다.**

## API와 로컬에서 실제 실행한 방법

| 실험 묶음 | API 계열 실제 실행 | 완전 로컬 실제 실행 | 구분 |
|---|---|---|---|
| 기존 공식 인식·raw Sol | 저장 Sol 현재/미래 | 공식 VEATIC→hold/linear | 로컬 인식은 이미 공개된497MB checkpoint; 우리 신규 인식기 학습 아님 |
| previous_exp/B | Sol 변화량 alpha 후보; 선택alpha1은raw와동일 | hold/linear 비교군 | 별도 학습 head 없음 |
| baseline_residual | current_bias/sol_residual/observation_residual×λ3,9개 Ridge 개발 | hold/linear 비교군만 | Ridge 학습은 API 계열에서만 실행 |
| fixed_baseline_bottleneck | 잔차축소·직교화·offset·directRidge | hold와train_mean 비교 | API direct head1개, 기존Ridge 재적합 검산1개; 모든14방법이 새학습이라는 뜻 아님 |
| llm_transfer | Sol잔차 GRU6+oldridge잔차 GRU2+meanpool Ridge1 | direct GRU6+localpool Ridge1 | 총 신경 head14개와 집계 head2개; 양 계열 실제 실행 |
| llm_transfer_expansion | 검색10+mean1+summary MLP6+linear4 | 검색10+mean1+summary MLP6+linear4 | 각21설정, 합42; 실제 학습 head각10개/총20. 검색·평균22설정은 신경학습 아님 |
| paper_experiments | raw Sol 비교 | Hi-EF 특징 분류·관측 VA departure 선택위험 | VA회귀421과 과제·분모가 달라 같은 성능 순위에 넣지 않음 |

확장 준비 코드에는 별도로 OOF Ridge5fold와 기존 전체train Ridge 재현1회의 적합도 있다. 위20개 후보 head 수와 준비용6회 적합을 혼동하지 않는다. `baseline_residual/development.json`에는9개 개발 결과가 있지만 최종 보존모델은 선택된1개다. 이번 조사는 기존 결과를 읽었고 그 적합 명령을 다시 실행하지 않았다.

## 로컬은 로컬끼리, API는 API끼리 비교

같은 final421, 미래0.5/1/2/3초×VA2축, CCC는8개 평균이다.

| 계열 | 기존 방법 | MAE↓ | CCC↑ |
|---|---|---:|---:|
| 로컬 | 인식→hold baseline | 0.260499 | 0.273737 |
| 로컬 | direct visual linear rank2 | 0.221843 | 0.234855 |
| 로컬 | direct numeric tiny MLP | 0.235255 | -0.013110 |
| 로컬 | **ours localpool Ridge** | **0.213456** | **0.279669** |
| API | raw Sol baseline | 0.222684 | 0.533652 |
| API | observation residual Ridge | 0.191646 | 0.485849 |
| API | residual_scale0.75 | 0.190787 | 0.521512 |
| API | **ours meanpool Ridge** | **0.188207** | **0.484942** |

이번 연결은 MAE를 우선 기준으로 둔 **이미 본 final의 사후 선택**이다. API meanpool은 CCC 손실이 있다. selection236에서는 API ours0.234353/0.432584 대 기존Ridge0.197490/0.559004, 로컬 ours0.286671/0.171460 대 hold0.266225/0.309384로 새로 연결한 두 방법 모두 나빴다. 기존 개발 채택 성공·새 독립 검증으로 표현하면 안 된다. 기존4개 실험의95방법 저장 예측을 이번에 동일421/236 GT로 직접 재채점했다.

현재 로컬 full1258은0.223368/0.320824, hold는0.280927/0.208860이다. full에는 과거 train251이 포함된다. API 제공은908/1258(72.18%), 제외350=기권287+대상 미확정63이며 로컬은1258/1258이다. 공통908의 사전 선별은 API 성공에 의존했던 과거 데이터 편향이며, 현재 로컬 runtime은 API 파일 없이1258개를 실행한다.

## head의 train 정답은 어디서 왔나

- 직접 확인한 파일: `/mnt/RG2/baseline_residual_20261110/b2/labels_train.json`,251개 미래VA8좌표,20원본 그룹.
- 같은 ID와 모든 수치가 `/mnt/RG2/my_works/experiments/benchmarks/2026-11-06/001_frozen_full/b2/ground_truth/forecast.json`의 공개 미래 GT와 정확히 일치한다.
- train251 구성은 `official_first98_test`52개와 `additional_unseen26`199개다. 공식 인식 테스트 구간을 우리 forecasting head의 개발학습에 쓴 사례가 포함된다.
- 분할의 앞선 출처는 `previous_exp/B/protocol/split.json`과 `paper_experiments_20261109/b2/results/split.json`이다. 기존평가풀을 development251/selection236/final421로 나눈 것을 이후 head 학습에 재사용했다.
- 공개 원 GT는 VEATIC `rating_averaged/{video_id}_{valence,arousal}.csv`의 사람 평정이다. 이 head의 목표값은 Sol 출력이 아니다. API meanpool은 Sol 수치를 **입력**으로 사용하며, localpool은 그10열을 학습·추론 모두0으로 치환했다.
- localpool의 기존 평균·계수10열이 정확히0이고, 해당 열을 제거한70열 head의 forward가 원80열 head와 동일함을 검증했다. 로컬 정규화 통계에도 Sol 값을 남기지 않았다. 학습 표본이 과거 API성공908에서 선택된 편향까지 없어지는 것은 아니다.

공식 VEATIC 현재 VA 인식기는 원래 첫98영상 앞70%로 사전학습된 별도 공개 모델이다. 우리 hold는 추가학습이 없고, meanpool/localpool 등 head는 위 평가 개발분할을 이용해 과거에 새로 학습한 모델이다. 기존 특징 캐시의 공식 앞70% train 특징은0개였으므로 이 둘을 공식 train 학습으로 설명할 수 없다.

## 과거 실제 학습 호출과 시각 근거

아래 명령은 과거 기록이며 **이번에 실행하지 않았다**. 환경은 기존 `/mnt/RG2/my_works/b1/benchmarks/action_genome_20261106/model/env/bin/python`이다.

| 자산 | 과거 실행 | 실제 기록 | 파일 수정시각 KST |
|---|---|---|---|
| meanpool_model.npz | `experiment.py develop` | `llm_transfer_20261110/b2/.omo/evidence/develop.log`, 개발8개 신경trace 및 meanpool모델 | 2026-11-10 10:09:54 |
| localpool_model.npz | `wave2.py` | 같은폴더 `.omo/evidence/wave2.log`, 추가6개 신경trace 및 localpool모델 | 2026-11-10 10:11:46 |
| PCA projection.npz | `experiment.py develop`의 train-only fit_projection | 원본모델·frozen.json·sourcehash | 2026-11-10 10:08:36 |
| 확장30설정 | `run.py develop` | `llm_transfer_expansion_20261110/b2/.omo/evidence/wave1.log` | 2026-11-10 11:12:19 |
| 확장12설정 | `run.py wave2` | 같은폴더 `.omo/evidence/wave2.log` | 2026-11-10 11:13:12 |

시각은 이번에 읽은 filesystem mtime이며 인증된 학습 시작시각이나 당시 사용자 승인 시각을 뜻하지 않는다. 코드·로그·가중치 SHA와 정확한 UTC 시각은 `/mnt/RG2/.omo/evidence/b2_ours/training_lineage.json`에 있다. 이 자료만으로 과거 사용자 승인 여부를 단정하지 않는다.

## 요청된 폴더 확인 목록

| 폴더 | 확인 상태와 b2 관련 판단 |
|---|---|
| research_runs | NEW_METHODS.md와 branch_comparison의 SCOPE_CORRECTION.md 확인. b2 새head/세계모델은 실행조건 미충족 후보이며 완료 실행은 주로b3. 과거 B.reproduce는 equality 실패·중단 기록이므로 새 성공 결과로 세지 않음 |
| previous_exp | B/REPORT.md와 BCD/START_HERE.md의 alpha 결과 확인. alpha1=rawSol, hold/linear 기준 및 split 출처. BCD는 이 결과의 통합이며 별도 b2 우승모델 아님 |
| C | README 확인. 현재 운동예측·웹 통합이며 새 b2학습·API 경로 아님 |
| my_works | b2/reports 초기표정질적실험, VEATIC 초기14/65평가, Sol common14, frozen_full1258 및 실제캐시 확인. 초기14/65는421과 분모가 달라 순위 분리 |
| paper_experiments_20261109 | b2 코드·START_HERE 확인. 공통split/GT 출처, Hi-EF와선택위험은별도과제 |
| baseline_residual_20261110 | b2 model/prepare/run/development/저장예측 직접 확인·재채점 |
| fixed_baseline_bottleneck_20261110 | b2 experiment/frozen/14방법 저장예측 직접 확인·재채점 |
| metric_local_followup_20261110 | 루트·하위목록 확인. b1/b3만 있고 b2실험 없음 |
| llm_transfer_20261110 | b2 code/projection/16학습head/33평가방법 확인·재채점; meanpool/localpool 자산 출처 |
| llm_transfer_expansion_20261110 | b2 prepare/run/42설정/46평가방법 확인·재채점 |
| sol_postprocess_b3_20261110 | 루트목록 확인. b3 후처리 묶음이며 b2실험 없음 |

실행 가능한 ours 명령과 세부 제약은 각 계열 `ours/b2/README.md`에 있다. 새 API·학습·다운로드 없이 기존 가중치 추론과 재채점만 수행했다.
