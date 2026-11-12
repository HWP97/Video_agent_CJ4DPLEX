# 사용 내역 및 구현 범위

API와 최신 로컬 모델 제출본의 기존 기술·데이터·직접 구현 범위를 구분한다. 버전은 확인된 commit·파일 해시·저장 요청 식별자를 사용한다. 확인되지 않은 이용 권한은 확보했다고 표시하지 않는다.

## 데이터와 권리

| 자산 | 출처·버전 | 이용 조건과 제출 범위 | 가공 |
|---|---|---|---|
| Action Genome / Charades | [공식 안내](https://github.com/JingweiJ/ActionGenome), 2016 원본 묶음의 `Charades/license.txt` | **License for Non-Commercial Use**. 연구·평가 이용 범위가 명시돼 있으나 제3자 배포와 수정 데이터 공개 권한은 부여하지 않음. 제출·공개 범위 확인 필요 | annotation 시점 선택, 관측 객체 구조화, 관계 평가 |
| VEATIC | [프로젝트](https://veatic.github.io/), 124 대상 영상; 원본 ZIP SHA `7c314f398cc37767e4313c0d96ead1db32744a2f03fd73d7d2b1132c7c684aca` | 공식 사이트는 연구 목적 배포, 원영상 저작권은 원권리자. 채용 제출·공개 영상 허가 별도 확인 필요 | 관측 프레임 선택, 대상 설명, PTS 정렬, 자체 미래 V/A 구간 |
| TrajNet++ 변환 좌표 | [데이터 저장소](https://github.com/vita-epfl/trajnetplusplusdata), 로컬 index·manifest | 원천별 조건 유지 | 대상·이웃 관측9점, 미래12점, nominal meter |
| UCY 원영상 | [OpenTraj 연구 배포](https://github.com/crowdbotp/OpenTraj/tree/master/datasets/UCY), students03·zara01의 video.avi | [UCY 안내](https://github.com/crowdbotp/OpenTraj/blob/master/datasets/UCY/README.md)는 원 배포자 출처 표시 조건의 자유 이용을 명시. Lerner·Chrysanthou·Lischinski, Crowds by Example(2007) 인용 | 중심 원점·축 변환으로 예측과 GT 점을 원영상에 투영. 표시 보간 및 미래0.5x 재생 |

VEATIC 사이트의 웹디자인 라이선스를 원영상 허가로 적용하지 않는다. 공식 저장소 [LICENSE](https://github.com/AlbusPeter/VEATIC/blob/master/LICENSE)는 CC BY-NC 4.0이다. 현재 확인만으로 비상업 연구 외의 재배포 권리를 확정하지 않는다. 원문 2.1·6.3에 따라 권리 미확인 원본·파생 영상을 그대로 공개 Repository에 넣는 방식으로 제출 준비를 완료할 수 없다.

Charades의 보존된 라이선스는 일반적인 CC 라이선스로 치환하지 않는다. 제3자 데이터 배포와 가공본 공개 권한을 부여하지 않으며, 학술 출판의 필요한 짧은 구간·정지화면 예시에 대한 별도 범위를 명시한다. 로컬 검토용 demo 제작과 회사 전달·공개 배포는 구분한다. 원문은 `licenses/CHARADES_LICENSE.txt`에 보존한다. UCY 다운로드 파일의 원 URL·SHA256은 보행 궤적 demo의 `source_videos.json`에 기록했다.

## 모델과 API

| 기술 | 확인 버전 | 사용 기능 | 기존 사용과 직접 구현 경계 |
|---|---|---|---|
| OpenAI Responses API / Sol | 저장 요청 `gpt-6.1-sol`, endpoint `https://api.openai.com/v1/responses` | 관계·현재/미래 VA·좌표 예측의 기존 raw 응답 | 모델 자체는 외부 기술. 입력 구성·대상 처리·응답 파싱·후처리·평가 연결은 프로젝트 구현 |
| 관계 예측 macro_direct | `data/b1/ours` 및 selection manifest의 동결 파일 | 94차원 특징→26관계 보정 | 기존 자체 실험 head를 재사용. 이번 제출 준비에서 새 학습하지 않음 |
| 감정 상태 예측 observation residual Ridge λ1 | `data/b2/ridge/frozen_model.json`, `ours/b2/selection236.json` | Sol 미래 VA 보정 | 기존 자체 특징·Ridge 및 selection236 선정 재사용 |
| VEATIC 현재 인식 모델 | 코드 commit `2a9d5f1b3b4af3c8746c9a508ceb44f4b4891f95`, checkpoint SHA `8c5fedc813fcc4b96c4d541ea7dc859cf154f9d19a5fe3d47ffb6c43cb2ee006` | 감정 상태 예측 잔차 모델의 저장된 관측 VA 특징 | ResNet50+시간 ViT는 저자 모델. API 제출 실행은 특징 캐시 재사용. 가중치 별도 조건 확인 필요 |
| 보행 궤적 예측 CV 혼합 | 고정 `cv_blend_0.75` | 관측 등속도와 Sol 혼합 | 자체 후처리; 모델 학습 없음 |
| Cosmos-Predict2.5 2B/post-trained (제출 범위 외) | 로컬 snapshot `15a82a2ec231bc318692aa0456a36537c806e7d4` | 보조 시각화 실험으로 각 head의 예측을 프롬프트화한 뒤 생성 영상을 실제 미래와 대조. 정량 평가에 포함하지 않음. 자산은 Repository 외부(`HWP_CJ4DPLEX_AI_support/submission_cleanup_20261112/cosmos_full/`)에 보존 | 모델은 NVIDIA 기술. 입력 절단·프롬프트 구성·baseline/ours/GT 시간 정렬은 프로젝트 구현 |

API 이용 조건은 실제 사용 계정·계약에 따른다. 저장 모델 식별자로 provider 내부 checkpoint 버전을 확정하지 않는다. API key·토큰은 제출하지 않으며 현재 재현은 키 없이 실행한다. 원본 영상의 외부 API 전송 허가 여부는 영상 배포 조건과 별개다.

Cosmos [공식 저장소](https://github.com/nvidia-cosmos/cosmos-predict2.5)는 소스 Apache-2.0, 모델은 NVIDIA Open Model License로 구분한다. 현재 로컬 코드에는 3090 환경용 메모리·해상도 변경 이력이 있으며 외부 모델을 직접 개발했다고 주장하지 않는다. Cosmos 생성 자산·로그는 제출 범위 외 보조 시각화로 분류해 Repository에 포함하지 않는다.

## 핵심 OSS와 개발 도구

| 기술 | 버전·조건 | 사용 범위 |
|---|---|---|
| SceneSayer / Action Genome 평가 코드 | SceneSayer commit `1d48739435e3f5e0fd9e8cc012668f31b28b187f`, MIT; 번들 `vendor/benchmark/model/SceneSayer/LICENSE` 보존 | 관계 예측 평가; 임의 재작성한 지표가 아님 |
| TrajNet++ tools | commit `39d56e0ee64ad85b71584e48819042aecd3ca434`, 번들 `TRAJNET_LICENSE` 보존 | 보행 궤적 예측 ADE/FDE/Col-II 공식 채점 |
| PyTorch·NumPy·Pillow·OpenCV·FFmpeg | 실행별 버전은 requirements와 검증 기록 | 텐서·배열 계산, 영상 읽기·오버레이·MP4 인코딩 |
| Codex (OpenAI) | 본 과제 수행 기간의 세션 (과제 수행 10/30~11/11 전반) | **구현 전반을 담당**: API 경로의 Sol 호출 코드·관계 점수 head·관측 잔차 Ridge·등속도 혼합·평가 연결, 로컬 경로의 세 head(Scene graph c04/c06, Affection C01/D02/D04, Trajectory F/D04) 아키텍처 비교·학습 스크립트·evaluate 파이프라인·동결 체크포인트 선택 로직, 데모 렌더러와 Cosmos 보조 시각화의 프롬프트 변환 코드를 작성·실행 |
| Claude Code (Anthropic) | 제출 마감일 2026-11-12의 제출 정리 세션 | 저장소 디렉터리 정리(지원자 지시의 support 이동), 아키텍처 Figure 배선, 공개 명칭 통일(Scene Graph Anticipation · Affection Forecasting · Trajectory Forecasting), 과제원문 요건 대조, Technical Report·Executive Summary·System Architecture·Usage &amp; Scope의 서사 다듬기, README·Dockerfile 작성, 외부 보관 경로 및 깨진 참조 정리, PDF 재생성 |
| 생성형 AI 개발 도구 공통 | 설치된 Ponytail·OMO 스킬을 포함한 보조 플러그인 | 코드·문서 작성, 자료 탐색, 실행·검증 보조 |

모든 코드가 사람이 단독 작성한 것처럼 표현하지 않는다. 지원자가 **문제 정의·과제 분해(세 head)·실험 방향 선택(API 우선 → API 비용 제약 대응으로 로컬 추가)·평가 집합·최종 Demo·제출 범위**를 결정하고, Codex가 그 안에서 세부 구현·학습 루프·평가 코드를 작성·반복 실행했다. 로컬 경로의 평가 프로토콜은 Codex 세션 중 공개 데이터셋 가용 범위에 맞춰 조정되었으며, 그 선택과 결과 해석·보고의 책임은 지원자가 유지한다. Claude Code는 과제 수행이 아니라 **제출 마감일의 산출물 정리**에 한정한다. 과거 작업의 모든 도구 버전을 이 세션에서 소급 추정하지 않는다.

## 직접 구현·수정 범위

프로젝트 구현은 관측 입력 계약, 요청 구성과 캐시 provenance, 과제별 후처리, 동일 집합 비교, 실패·기권 처리, 실행 경계, 결과 연결, 영상 렌더러·GIF 발췌 생성이다. 외부 foundation model, 원천 데이터·주석, 저자 인식기와 공식 평가기는 기존 기술이다. 로컬 실험이 완료된 뒤 사용자 승인으로 최신 동결 모델을 제출 경로에 연결한다. 실험 원본과 이전 모델·결과는 보존하며 이번 연결을 새 학습으로 표현하지 않는다.

## 최신 로컬 모델의 사용·구현 범위

| 구성 | 출처와 버전 | 직접 구현 범위와 한계 |
|---|---|---|
| ImparTail ODE | 저자 저장소 `rohithpeddi/ImparTail`, 확인 commit `f716277faf19de507c0e1400eb7f48cef1cf8248`, 기존 epoch3 checkpoint | 저자 사전학습 모델 재사용. 관측 benchmark 실행·확률 추출·최신 보정 연결은 자체 구현. 가중치 재배포 이용 조건은 코드와 별도 확인 필요 |
| 관계 메모리 gate·보정 head | `iterative_research_20261111/b1/combinations_81112`의 선택 lock·checkpoint·memory SHA | 관측311차원 선형 보정,97차원 메모리 검색,4→26 gate와 양성 희귀도 가중·cap 조합은 자체 실험. 외부 ImparTail을 자체 개발했다고 주장하지 않음 |
| 감정 C01 Baseline·D04 Ours | `b2/followup_reasoned_combinations`의 C01 GRU와 transform, `b2/diagnosis_targeted_experiments`의 D02·D04 계수 | 외부 VEATIC 인식·고정 영상 특징 사용. C01은 PCA32+GRU hidden8·시점별 head, D04는 PCA16·현재 잔차 및 미래 delta Ridge. 자체 학습된 동결 모델이며 이번 제출 준비에서는 재학습하지 않음 |
| 보행 attention·F D04 | `b3/diagnosis_targeted_experiments`의3fold checkpoint·선택 epoch | 기존 자체 attention 및 F 두 상태 반복 축소 구현을 사용. optimizer moment 복원과 epoch0 선택 보존을 구현. 원 논문의 대형 모델 전체 재현이라고 주장하지 않음 |

최신 감정 전체2,740사례와 보행240,890장면은 과거 API 부분집합과 다르다. 보행 평가 원자료에는 UCY 외 다른 원천 그룹도 포함되므로 원 `runplan.json`·자료 provenance의 원천별 조건을 유지한다. 실제 데모용 UCY 두 원영상의 허가만으로 전체 좌표 원자료 재배포 허가를 대신하지 않는다.

최신 보행의 배포 출처는 [TrajNet++ v4.0 release](https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0)의 실사 train이다. CFF·UCY·hotel·LCAS·WildTrack을 사용자 정의3fold로 나눈 결과이며 공개 challenge test 재현이 아니다. 원 배포 metadata와 ZIP 내부 목록은 `official_local_train_20261111/b3/provenance`에 보존돼 있다. 원천별 연구·제출·재배포 조건이 확정되지 않은 자료는 공개 Repository에 포함하지 않는다.

동결 모델·입력·GT는 오프라인 실행 경계 안에 독립 복사하고 SHA를 대조한다. 복사본이 로컬에 존재한다는 사실은 제출 Repository에 대용량 자산을 넣으라는 뜻이 아니다. 원문2.4에 따라 자체 head와 checkpoint에도 평가자가 접근할 별도 저장 위치가 필요하다. 외부 위치와 제출 권리 미확정 항목은 기존 조건표에 남긴다.

대용량 모델과 데이터는 Repository에 넣지 않고 이름·원출처·준비 경로·무결성 정보를 제공해야 한다. 자체 동결 head·API 캐시·provenance의 평가자 접근 가능한 외부 위치는 아직 확정되지 않았다. URL을 만들거나 업로드하지 않은 상태를 배포 완료로 표시하지 않는다.
