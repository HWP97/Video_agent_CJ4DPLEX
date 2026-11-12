# 환경 재현

실제 검증은 Ubuntu의 Python 3.10.22와 RTX3090 24GB에서 수행했다. b1/b3와 b2는 기존에 서로 다른 Torch/NumPy 환경을 사용했으므로 한 환경에 의존성을 강제로 합치지 않는다. b2의 공식 코드와 einops 보조 구현은 복사본 vendor에 있다.

```bash
python3.10 -m venv .venv_ag
.venv_ag/bin/python -m pip install -r requirements.txt
python3.10 -m venv .venv_b2
.venv_b2/bin/python -m pip install -r requirements/b2.txt
python3 src/api_based/main.py --python "$PWD/.venv_ag/bin/python" b1 baseline status
python3 src/local_model_based/main.py --python "$PWD/.venv_b2/bin/python" b2 baseline status
```

위 설치 명령은 새 장비용 안내다. 이번 정리에서는 기존 환경으로 실제 실행했으며 깨끗한 새 환경 설치까지 수행하지 않았다. 정확한 설치 환경 전체 패키지 목록은 `ag_b3.observed.json`, `b2.observed.json`에 별도로 기록한다. 실행에 필요하지 않은 과거 환경 패키지까지 자동 설치하지 않는다. GPU 드라이버는 각 CUDA wheel을 지원해야 한다.

최신 로컬 모델 세 과제는 `requirements/local.txt`를 사용한다. 기존 AG Python3.10 환경에서 pydantic2.11.10·tokenizers0.21.4를 추가로 확인했다. 위의 별도 감정 환경은 이전 인식기와 API 경로의 안내이며 최신 D04 동결 특징 계산에는 강제하지 않는다.

```bash
python3.10 -m venv .venv_local
.venv_local/bin/python -m pip install -r requirements/local.txt
python3 main.py --python "$PWD/.venv_local/bin/python" local affect ours status
```

기존 AG·보행 목록의 scikit-image0.25.2는 NumPy≥1.24를 요구해 고정 NumPy1.23.5와 충돌했다. 해당 경로와 최신 동결 head에는 활성 import가 없어 AG·보행 목록에서 제거했다. 이전 감정 인식기의 공식 dataset 코드는 이를 사용하므로 원 연구의 데이터 로더·재학습 환경 요구와 최신 동결 모델의 실행 목록을 구분한다. 설치된 원 환경은 변경하지 않았으며 원 연구 전체 재학습 환경이나 새 장비 설치 성공을 검증한 것은 아니다.
