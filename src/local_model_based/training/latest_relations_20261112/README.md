# 최신 관계 모델 학습 소스 보존본

`data/b1/latest/source/`의 원문 snapshot을 소스 제출 대상 경로에 동일하게 복사했다. `.gitignore`에서 대용량 data를 제외할 때 학습 소스까지 빠지지 않도록 한 것이다. 파일 내용은 수정하지 않았고 `SOURCE_MANIFEST.json`의 원본·복사 SHA를 따른다. manifest의 기존 destination은 data 보존본 경로다.

SGCLS c04·PredCls c06의 모델·학습·평가·선택 관련 코드다. 원 연구 경로 의존을 유지한 보존본이며 제출 준비에서 재학습하지 않았다. 실제 동결 추론 진입점은 `ours/b1/latest_cli.py`와 `latest_inference.py`다. 이전 학습 실험과 혼동하지 않는다.
