# 최신 감정 모델 학습 소스 보존본

`data/b2/latest/training_source/`의 원문 snapshot을 소스 제출 대상 경로에 동일하게 복사했다. 대용량 data를 제외해도 학습·선택 코드가 남도록 한 것이며 내용을 수정하지 않았다. SHA는 `data/b2/latest/assets.json`과 해당 source manifest를 따른다.

D02 현재 보정과 D04 미래 변화량 모델의 학습·선택·추론·검증 원문이다. 원 연구의 경로 의존을 보존했으며 제출 준비에서 재학습하지 않았다. 실제 동결 추론은 `ours/b2/latest_inference.py`를 따른다.

`c01/`에는 사용자 지정 baseline인 C01 직접 미래 GRU의 models/engine/run, 등록·선택·사전 결정·결과 및 공통 함수를 원문 그대로 보존했다. SHA와 원본 경로는 `data/b2/latest/assets_c01.json`에 있다. 실제 C01 추론은 `ours/b2/c01_inference.py`가 `c01/models.py`의 Expert 구조와 동결 checkpoint/transform을 사용한다. GRU는 원train1169·seed20261111·20epoch이며 MTP는 사용하지 않는다. 이번 준비에서는 재학습하지 않았다.
