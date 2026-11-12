# 완전 로컬 예측 — 최신 실험 연결

외부 추론 API 없이 기존 동결 모델을 실행한다. 사용자 승인으로 최신 실험 모델을 연결했으며 이전 배포 코드와 결과는 보존했다. 이번 제출 준비에서 모델을 다시 학습하지 않았다.

| 과제 | Baseline | 최신 Ours | 평가와 한계 |
|---|---|---|---|
| 관계 예측 | ImparTail ODE | 메모리 gate·logit 보정, SGCLS cap3 / PredCls cap1.5 | 관측 GT 객체·미래 주석 시각 사용. 내부 선택 조합 |
| 감정 상태 예측 | C01 GRU 직접 미래 예측 | D02 현재 잔차 + D04 미래 변화량 | 전체2740, official_test438. C01 대비 MAE·CCC 개선 |
| 보행 궤적 예측 | attention | F 두 상태 반복 모델 D04 | 동일3fold240890장면. 성능 회복 실패 |

보행 D04는 최신 추가학습 조건의 명시적 배포 선택이며 전역 사전 선택 winner가 아니다. API와 로컬은 평가집합이 달라 같은 표의 숫자로 직접 우열을 비교하지 않는다.

감정 C01은 기존 후속 조합의 소형 GRU 기준 모델이다. 이번 제출 준비에서 사용자가 주 Baseline으로 지정한 비교 기준 변경이며 결과 열람 전 사전 지정이라고 표현하지 않는다. hold와 이전 GRU는 추가 비교군이다. D04가 C01보다 두 지표가 좋아도 기존 GRU보다 우수하다는 뜻은 아니다.

보행240,890장면은 TrajNet++ v4.0 실사 train의 사용자 정의 그룹 교차검증이며 공식 challenge test가 아니다. 관측9점·미래12점은 같지만 WildTrack은0.5초, 나머지는0.4초 간격이다.

## 실행

새 환경의 설치 목록은 `requirements/local.txt`다. 동결 모델 실행은 기존 Python3.10·Torch2.5.1+cu118·NumPy1.23.5·pydantic2.11.10·tokenizers0.21.4에서 확인했다. 새 장비 전체 설치는 별도 검증이 필요하다.

프로젝트 메인 폴더에서 실행한다. 실제 지원 옵션은 각 과제 명령의 `--help`와 상세 README를 따른다.

```bash
python3 main.py local relations ours status
python3 main.py local affect baseline evaluate
python3 main.py local affect ours evaluate
python3 main.py local trajectories ours status
python3 main.py local trajectories ours verify
```

루트 실행기는 의미 있는 과제 이름을 내부 과제 식별자로 변환한다. 계열 내부 실행기 `src/local_model_based/main.py`의 기존 식별자와 `final=ours` 호환 별칭도 유지한다. Python 환경을 바꾸려면 루트의 전역 `--python /환경/bin/python` 옵션을 사용한다.

결과는 `results/local_model_based/` 아래의 과제별 최신 경로에 저장한다. predict는 지원되는 동결 특징·좌표·관계 입력에서 실제 모델 계산을 실행하며, replay·evaluate·verify와 구분한다. 감정 최신 predict는 저장된 관측 특징 계산이며 원영상 임베딩 재추론이 아니다. 이전 인식기의 `--recognize` 경로는 기본 최신 명령으로 안내하지 않는다.

## 입력과 재현 경계

- 관계: 관측 graph와 ImparTail prior 및 학습 memory가 필요하다. 미래 GT는 채점 단계에만 연다.
- 감정: 마지막1.8초의15개 영상 특징과 관측 VA 통계를 사용한다. 학습 분리의 보수적8초 관측창과 실제 특징 범위를 구분한다.
- 보행: 대상·이웃의9개 관측을125차원 특징으로 구성한다. 출력은12개 미래2D 좌표다. 원영상→좌표 검출의 범용 서비스는 아니다.

runtime_guard는 네트워크·하위 프로세스·원본 작업 폴더 접근과 외부 쓰기를 제한한다. 최신 모델·관측·GT·필요 코드는 bundle에 독립 복사해 제한을 유지한다. 이는 악성 native 코드까지 격리하는 OS sandbox는 아니다.

## 상세 설명과 이전 자료

- [관계 최신 연결](ours/b1/README.md)
- [감정 최신 연결](ours/b2/LATEST.md)
- [보행 최신 연결](ours/b3/README.md)
- [통합 Technical Report](../../docs/TECHNICAL_REPORT.md)

이전 entry·branch 설정은 과제별 legacy 경로에 남긴다. 이전 결과의 성적·데이터와 최신 결과를 섞지 않는다. 학습 원문 snapshot은 출처 추적용으로 포함했고 제출 준비에서 재학습 CLI나 새 장비 전체 설치를 검증한 것은 아니다.

자료와 가중치의 원 이용 조건을 유지한다. 모델·대용량 데이터는 Repository 외부에서 평가자가 접근할 위치를 별도로 확정해야 한다. 이번 작업은 GitHub push나 외부 업로드를 수행하지 않는다.
