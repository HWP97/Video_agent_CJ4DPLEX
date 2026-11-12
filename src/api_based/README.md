# API 기반 예측

기존 코드를 이동하지 않고 복사한 독립 작업 디렉터리다. 저장된 Sol 응답을 재사용하며 이 실행기에는 유료 수집 경로가 없다. 원본 자료와 기존 서버는 그대로 둔다.

| branch | baseline | ours | 상태 |
|---|---|---|---|
| b1 Action Genome | 저장 Sol 관계 예측 | 기존 macro_direct 보정 | NC 개선, WC 악화; 사후 선택 |
| b2 VEATIC | 저장 Sol 미래 VA | selection236 선정 관측 잔차 Ridge λ1 | final MAE 개선, CCC 악화 |
| b3 TrajNet++ | 저장 Sol 수치 궤적 | 0.75 Sol + 0.25 CV | 거리 개선, 충돌 악화; 사후 선택 |

`ours`는 기존 실험에서 선정한 방법이다. `final` 명령은 호환 별칭이다. 세 API branch 모두 baseline과 계산이 다르지만 모든 지표에서 좋아지는 것은 아니다. b1 PGAGS/GAGS는 관측 정답 객체 정보, b3는 관측 좌표를 제공받는 benchmark다.

## 폴더

```text
api_based/
  main.py                 # 공통 실행기
  baseline/b1,b2,b3/       # 기준 방법과 공통 실행 코드
  ours/b1,b2,b3/            # 기존 최종 방법 및 명시적 연구 옵션
  data/b1,b2,b3/           # 복사한 관측 입력, Sol 원문, GT, 동결 가중치
  ../../results/api_based/b1,b2,b3/ # 원 결과와 새 실행 결과
  tests/                  # 실행 경계와 branch 검사
  evidence/               # 전체 분리 검증
```

## 실행

```bash
cd /mnt/RG2/HWP_CJ4DPLEX_AI/src/api_based
python3 main.py list
python3 main.py b1 baseline replay
python3 main.py b1 ours evaluate --limit 1
python3 main.py b2 baseline evaluate
python3 main.py b2 ours evaluate
python3 main.py b3 baseline replay --output results/b3/my_replay
python3 main.py b3 ours evaluate --prediction results/b3/my_replay/predictions.npz --output results/b3/my_score
```

b1의 `--limit 1`은 실행 확인용이다. 전체 재채점은 limit을 생략한다. b1 `replay`는 저장된 전체 사례 점수를 다시 집계하며, `evaluate`는 예측을 평가기에 다시 넣는다. b3 출력 디렉터리는 새 이름을 사용한다. `results/b3/...` 상대경로는 프로젝트 최상위 `results/api_based/b3/...`로 해석한다. 명령별 옵션은 `python3 main.py b1 ours --help`처럼 확인한다.

API 키 없이 동작한다. 캐시가 없으면 실패 또는 누락으로 기록하며 API를 자동 호출하지 않는다. `predict`도 이 계열에서는 저장 응답의 재파싱·후처리다. 새 입력에 대한 API 추론이 아니다.

Python 환경은 각 `branch.json`에 기록된 기존 환경을 재사용한다. 코드·데이터는 이 폴더를 사용하지만 가상환경까지 복제하지 않았다. 환경을 옮길 때는 동일 의존성을 준비하고 `python3 main.py --python /환경/bin/python b2 ours evaluate`로 지정한다. 관측 입력 조건을 바꾼 뒤 기존 Sol 응답을 새 추론 결과로 사용하면 안 된다.

## 상세 기록

- [b1 방법·캐시·평가](ours/b1/README.md)
- [b2 방법·캐시·평가](ours/b2/README.md)
- [b3 방법·캐시·평가](ours/b3/README.md)
- [Technical Report](../../docs/TECHNICAL_REPORT.md)

파일 복사는 symlink/hardlink 대신 별도 파일로 했다. 출처 경로와 SHA256은 branch별 copy manifest에 남는다. provenance의 과거 절대경로는 실행 의존성이 아니다. 원본 데이터의 이용 조건은 그대로 적용되며 이 복사본이 새로운 공개 배포 허가를 뜻하지 않는다.
