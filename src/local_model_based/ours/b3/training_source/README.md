# 최신 D04 학습 소스 보존본

이 폴더는 제출 검토를 위한 원문 소스 snapshot이다. 이번 제출 준비에서 학습을 재실행하지 않았다. `SOURCE_MANIFEST.json`은 원본 경로와 설치 경로의 같은 SHA256을 기록한다. 이미 bundle에 있던 공식 입력 어댑터·전처리·attention 소스와 F 모델 소스는 중복 복사하지 않고 기존 파일을 참조한다.

- `latest/train.py`: epoch0 포함 선택, optimizer moment 복원, 학습·refit.
- `latest/common.py`, `register.py`, `registration.json`: 관측 특징·그룹 분할·사전 학습 조건.
- `parent/models.py`, `engine.py`: F wrapper·두 상태 출력·학습 목적과 이전 parent의 학습 코드.
- `receipts/fold1/2/3.json`: 실제 D04 full 실행의 동결 receipt.
- `latest/infer.py`, `score.py`, `finalize.py`: 원래 추론·충돌 평가·집계.
- `latest/DECISIONS.md`, `RESULTS.md`, `execution_deviation.json`: 사전 조건·실패 결과·실행 편차 보존.

원문 코드의 상대 디렉터리와 외부 학습 자료 의존성을 그대로 보존했다. snapshot만으로 독립 장비의 재학습이 가능하다고 주장하지 않는다. 학습 원자료·parent optimizer checkpoint·그룹별 내부 분할 자산은 원 support 실험 경로에 있으며 평가자 외부 자료 안내가 별도로 필요하다. 현재 배포 추론은 snapshot을 실행하지 않고 bundle 내부 동결 자산과 최신 진입점을 사용한다.
