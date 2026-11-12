# 두 계열 코드 분리 작업 계약

2026-11-10 사용자가 승인한 복사 기반 분리 작업이다. 원본은 그대로 보존한다. 새 학습·API 호출·다운로드·기존 서버 변경은 하지 않는다.

각 계열은 `baseline/b1,b2,b3`, `src/b1,b2,b3`, `data/b1,b2,b3`, `results/b1,b2,b3`, `reports/b1,b2,b3`, `tests/`를 사용한다. `baseline`은 고정 기준, `src`는 최종 선택과 실제 개선 코드다. 개선 미채택 branch는 baseline 유지와 연구 옵션을 구분한다. 사용하지 않는 전 실험/환경/원본 영상 전체를 복제하지 않는다. 실제 재현에 필요한 입력·GT·응답·가중치·결과는 복사한다. 데이터는 symlink/hardlink 없이 독립 inode로 복사하고 SHA256 출처 manifest를 남긴다. reflink는 허용한다.

## 작업자 소유권

- b1 담당: 두 계열의 baseline/b1, src/b1, data/b1, results/b1, reports/b1, tests/test_b1*.
- b2 담당: 두 계열의 baseline/b2, src/b2, data/b2, results/b2, reports/b2, tests/test_b2*.
- b3 담당: 두 계열의 baseline/b3, src/b3, data/b3, results/b3, reports/b3, tests/test_b3*.
- 총괄: 두 계열 루트 main.py, README.md, 공통 안전 실행기, evidence와 최종 통합 검사.

## 최소 CLI 계약

각 `baseline/bN`과 `src/bN`에는 `entry.py`, `branch.json`을 둔다. entry.py는 status, replay, evaluate, verify 명령을 지원한다. 실제 로컬 추론이 가능한 경로에는 predict 명령을 추가한다. API predict는 저장 입력을 사용하는 후처리만 허용하며 새 유료 요청은 불가하다. 명령에 추가 인수가 필요하면 --help에 명시한다. 결과를 사용자 지정 경로에 저장하거나 기본 results/bN 아래 새 실행 디렉터리를 만든다. 원본 또는 기존 결과를 덮어쓰지 않는다.

branch.json 필수: `branch`, `family`(api/local), `stage`(baseline/final), `method`, `adoption_status`, `python`(기존 환경 실행기 절대 경로), `actions`(문자열 배열), `description`. 실행 환경만 외부 기존 환경을 참조할 수 있다. 런타임 코드·데이터는 해당 계열 내부 파일을 사용한다. 기존 원본 경로는 provenance 설명으로만 남긴다. 복사본을 다른 디렉터리로 옮겨도 root 상대 경로로 찾도록 한다.

root 진입점은 `python3 main.py bN baseline|final <command> ...`로 branch entry.py에 위임한다. root는 API키 제거·외부 네트워크 차단·다른 프로젝트 원본 읽기/쓰기 차단을 적용해 기존 환경에서 실행한다. worker도 직접 entry.py 실행의 유료 fallback을 만들지 않는다.

## 방법 선택

- API b1: raw Sol 기준. src에 rank_residual 구현과 동결 가중치/결과를 포함하되 WC 소폭 개선·NC 손실을 공개. 범용 final은 baseline 유지, 명시적 연구 옵션 가능.
- API b2: raw Sol 기준, src는 기존 observation residual Ridge 개선. 단순 보정량 재튜닝 금지.
- API b3: raw Sol 기준, final은 baseline 유지. 기존 비학습 postprocess는 미채택 연구 옵션으로 보존. 취소된 새 학습 보정기 구현 금지.
- 로컬 b1: ImparTail ODE 기준, final 유지. 기존 head를 새 개선으로 채택하지 않는다.
- 로컬 b2: VEATIC 현재 인식→hold 기준, final 유지. Sol 특징/teacher 없는 코드만 사용한다.
- 로컬 b3: 기존 direct_unshared4_d4 attention 기준, final 유지. Sol 수집 경로를 분리하고 로컬 가중치·입력만으로 실제 예측 확인.

## 완료 근거

복사 파일 source/destination/bytes/SHA, 독립 파일 여부, cache-only 재채점 동일성, 캐시 누락 처리, API/네트워크 차단, 원본 읽기 없는 실행, 원본 보존, stage와 실패 분모를 검사한다. 외부 환경 패키지와 GPU는 재사용하며 환경 자체는 복제하지 않는다. 모델 inference는 필요한 최소 검증만, 학습·튜닝은 하지 않는다.
