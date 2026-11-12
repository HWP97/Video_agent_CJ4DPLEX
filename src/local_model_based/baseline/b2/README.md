# 감정 상태 예측 baseline: C01 직접 미래 GRU

현재 기본 baseline은 사용자가 지정한 C01이다. PCA32와 상대시각을 소형 GRU에 입력하고 관측 통계·horizon head로 네 미래 VA를 직접 출력한다. 현재값 head가 없으므로 current 지표는 해당 없음이다. 현재값 유지 hold는 `--method baseline_hold` 추가 비교군으로 남긴다.

계열 루트에서 `python3 main.py b2 baseline predict`, `evaluate`, `verify`를 실행한다. 같은2,740개 관측과 official_test438에서 D04와 비교한다. [정확한 구조·점수·재현 범위](../../ours/b2/LATEST.md)를 참조한다. 학습·관측·가중치 원본은 보존했으며 새 학습은 하지 않았다.

이전 hold 문서는 `legacy_hold/README.md`, 이전 관측 인식/hold 실행기는 `legacy_entry.py`로 보존했다. 기존 원영상 인식 명령은 기존 호환 Python으로 `runtime_guard.py baseline/b2/legacy_entry.py predict --recognize`를 실행하는 경로이며 현재 기본 C01 CLI와 구분한다.
