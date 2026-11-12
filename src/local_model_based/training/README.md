# 공식 학습 영역 실험의 코드 보관본

실제로 검증한 로컬 실험 코드를 이곳에 보관한다. 실험 실행과 데이터·가중치·예측·점수·보고서는 제출 폴더 밖의 `/mnt/RG2/HWP_CJ4DPLEX_AI_support/official_local_train_20261111/`에 둔다. 기존 baseline/ours 실행기와 서비스 기본값은 변경하지 않는다.

| 폴더 | 내용 | 원본 실행 위치 |
|---|---|---|
| `b1_official_20261111/` | 공식 AG train500의 작은 head와 최종 v3 구간 정합성 교정본 | support 실행 루트의 `b1/config_sweep_v3/`, 첫 실험군은 `b1/subset500/` |
| `b2_official_20261111/` | 공식 앞70%로 학습한 VEATIC 로컬 실험의 동결 소스 | support 실행 루트의 `b2/attempt_fp32/` |

이 폴더는 동결 소스의 보관본이다. 현재 과학 코드의 ROOT는 파일 위치를 기준으로 계산하므로 **복사본의 학습 진입점을 여기서 직접 실행하지 않는다.** 아래 명령은 실제 데이터와 가중치가 있는 support 실행 위치를 사용하며, 파일을 쓰지 않고 결과를 검증한다.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B \
  /mnt/RG2/HWP_CJ4DPLEX_AI_support/official_local_train_20261111/b2/attempt_fp32/readonly_check.py
```

별도 폴더로 재추론·재채점하는 명령은 각 `SOURCE_MANIFEST.json`의 `commands`에 있다. 재사용 함수는 `reusable_functions`, 공개 가중치·입력 해시는 `assets_sha256`, 검증 환경은 `environment_versions`를 따른다. 가중치·환경·GT·원본 영상·실험 결과는 이 코드 보관본에 포함하지 않는다.

독립 설치 또는 다른 장비로 옮기는 CLI 통합까지 완료했다고 주장하지 않는다. 경로를 바꿀 때는 데이터·모델 파일을 함께 확인하고, 입력 또는 과학 코드가 달라지면 별도 실행 버전으로 기록해야 한다.

b1의 `offline_case.py`는 명시적인 실험 디렉터리와 코드 디렉터리를 인수로 받는다. 이 소스 보관본으로 실제54조건 예측·저장 결과 전용 재채점을 실행해 전체 실행 기록과 대조했다. 10개 단위 검사도 복사본에서 통과했다. 산출물은 support의 별도 `qa_hwp_root_*_20261111` 폴더에만 기록했다. 각 `runplan.json`은 import와 단위 검사용 동결 설정이며 모델·GT를 포함하지 않는다.
