# Executive Summary

## 문제

영상 콘텐츠에서 "곧 무엇이 일어날지"를 미리 알아야 하는 요구가 늘고 있다. 과제원문 3.1이 예시한 Motion Prediction 시나리오는 관계 변화, 감정 변화, 보행자 위치 변화 등 서로 성격이 다른 예측을 포함한다. 이 모두를 한 번의 end-to-end 모델로 처리하면 각 과제의 독립 평가와 실패 분석이 어려워져, 본 Agent는 범위를 세 개의 forecasting head로 분해해 각각을 benchmark 조건에서 검증한다.

## 접근 방식

- **과제 분해.** Scene graph head(관계), Affect head(Valence/Arousal), Trajectory head(좌표) 세 모듈로 나눠 설계·평가·실패 분석의 모듈성을 확보했다.
- **초기 비교와 API 중심 설계.** 각 head에서 최신 상용 Sol API와 공개 베이스라인 기반 로컬 모델을 같은 사례로 비교했다. Sol 응답이 전 head에서 우위여서 Sol 응답에 작은 보정(관계 head · 잔차 Ridge · 고정비율 혼합)을 얹어 API 경로를 1차 Ours로 채택했다.
- **Affect head의 자원 제약 대응.** Ridge 학습을 위한 추가 Sol 호출 비용이 과다하여, 과거 확보한 Sol 응답 908건을 Train 251 / Selection 236 / Test 421로 분할해 재사용했다. 범위와 한계는 Technical Report 3장에 공개한다.
- **API 비용이 부담스러운 운영 시나리오 대응 — 로컬 경로 추가.** API 비용·호출 승인이 제약이 되는 환경에서도 Agent를 운영할 수 있어야 한다는 판단으로, 외부 추론 API 없이 동결 체크포인트만으로 동작하는 로컬 head 세 개를 뒤이어 개발했다. 설계·학습 루프는 Codex 기반 반복 실험을 활용했다.
- **프로토콜 분리와 비교 범위.** 로컬 모델 개발 중 공개 데이터셋 가용 범위에 맞춰 평가 프로토콜을 조정했다(VEATIC Affect 70/30 영상 분할, Trajectory TrajNet++ v4 train 기반 custom 3-fold 등). 그 결과 API 경로(final 421·공통 1,939)와 로컬 경로(official_test 438·240,890 장면)의 cutoff·사례 모집단이 달라 **두 경로의 숫자를 직접 비교하지 않는다**.

## 주요 결과

| 경로·head | 같은 집합에서 확인한 지표 | 함께 보고하는 한계 |
|---|---|---|
| API Scene graph | GAGS NC R@20: 47.37% → 61.77% | WC R@20: 43.81% → 42.17% |
| API Affect | final 421 MAE: 0.2227 → 0.1916 | CCC: 0.5337 → 0.4858 |
| API Trajectory | 공통 1,939 ADE: 0.5883 → 0.5715 | Col-II 18.88% → 19.13% |
| Local Scene graph | SGCLS R20/mR20: 0.4803/0.3365 → 0.4911/0.4186 | 직전 유망 후보 대비 R20 하락 |
| Local Affect | official_test 438 MAE: 0.2217 → 0.2188, CCC: 0.5658 → 0.5774 | 과거 GRU 수준 미회복 |
| Local Trajectory | 240,890 장면 FDE: 2.1705 → 2.1575 | ADE 1.0522 → 1.0590, 충돌률 22.05% → 22.15% |

보정이 모든 지표를 동시에 올리지는 않는다. Affect의 MAE(오차)와 CCC(변화 일치도), Trajectory의 거리 오차와 충돌률은 자주 상충한다. 감정 값은 관측자가 평정한 표시 상태이며 임상적 내면 감정이 아니다.

## 시연과 검증

각 head의 baseline · Ours · GT를 함께 표시한 Demo 15개(head별 5개, API 경로)를 지표 차이와 원영상 이용 가능성을 고려해 선정했다. Demo는 모집단 대표 표본이 아니다. 재현성은 동결 체크포인트 SHA 대조와 저장 예측과의 수치 일치(최대 차이 0)로 검증했다.

## 리스크

- **API와 Local의 평가집합·cutoff가 달라 직접 성능 비교가 성립하지 않는다.** 통합 지표로 환산하지 않는다.
- **평가 자료의 반복 열람·선택 편향**이 있어 독립 hidden test 성적이 아니다. Affect Ridge 분할은 과거 응답 재사용에 의존한다.
- **로컬 head의 지표 상충.** Trajectory의 ADE↑/FDE↓, Affect의 CCC 미회복 등 특정 지표의 악화가 공존한다.
- **외부 자산 제공 범위.** 학습 가중치와 전체 API Sol 캐시는 OneDrive 로 제공된다(README 3-2 링크). 그러나 로컬 모델 전수 재추론에 필요한 입력 특징(`inputs.npz`·`manifest.json`) 과 평가 GT 는 외부 묶음에 포함되지 않아, Repository + OneDrive 만으로는 240k 전수 forward 재채점을 보장하지 않는다.

## 향후 개선 방향

- API cutoff와 Local cutoff를 통일한 공정 비교 체계 구축. 기존 Ridge를 동일 cutoff에서 재채점.
- Trajectory의 ADE·충돌률 악화 해소를 위한 상호작용 손실·가중치 재탐색, Affect CCC 회복을 위한 시계열 상관 손실 도입.
- 임의 영상 입력의 검출·추적·대상 식별 파이프라인 연결로 benchmark 재현 범위를 영상 서비스까지 확장.
- 로컬 모델 전수 재추론 입력(`inputs.npz`·`manifest.json`·GT) 외부 제공, 원영상 재배포 조건 확정 후 README·Resources 에 추가 반영.

이 과제의 평가는 최고 점수 제시가 아니라 **문제 분해·비용/데이터 제약 공개·재현 가능한 결과와 명시된 한계를 함께 보고**하는 데 중점을 두었다.
