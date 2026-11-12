# 보행 궤적 예측: 최신 로컬 D04

현재 Ours는 최신 진단 실험의 **B3-D04**다. 사용자의 최신 모델 교체 요청에 따라 optimizer moment 유지·epoch0 포함 조건을 배포했다. 실험 전체에서 하나의 전역 winner를 사전 선택한 기록은 없으므로, 이를 사전 선정 우승 모델이나 성능 회복으로 표현하지 않는다.

|같은3fold240,890장면|ADE↓|FDE↓|관측 이웃 GT 충돌률%↓|
|---|---:|---:|---:|
|Baseline attention|1.0522110912|2.1705463989|22.0461621487|
|Ours D04|1.0589728412|2.1575283300|22.1540952302|

D04는 attention보다 FDE만 낮고 ADE와 충돌률은 높다. 역사적 F 참조의1.026588/2.100548/21.200963%를 회복하지 못했다. 반복 열람한 평가 영역이며 새 독립 holdout이 아니다. 이전1,943장면 결과와 분모·충돌 정의가 다르므로 같은 표의 개선율로 섞지 않는다.

## 모델과 입력

관측9시점의 ego 특징, 최대8이웃 특징과 mask를 합친125차원을 동결 표준화한다. Linear125→128 encoder와 공유 block으로 두 상태 y·z를 반복 갱신한다. 마지막24출력을12×2로 해석하고 관측 원점·회전·scale로 원 좌표에 복원한다. 미래 GT·Sol 입력·새 gate·충돌·증류 모듈은 추론에 없다.

D04는 같은 F parent에서 optimizer moment/step을 유지하며 학습률0.0001, 최대10epoch, epoch0 포함 선택을 사용한 완료 checkpoint다. 추가 epoch는 fold1=5, fold2=0, fold3=0이다. 배포 준비에서는 새 학습을 하지 않았다. 각 fold의 원본 그룹 평가 경계는 data/b3/latest/runplan.json, 동결 선택은 selection/foldN.json에 보존한다.

## 실행

계열 루트 src/local_model_based에서 실행한다.

```bash
python3 -B main.py b3 ours status
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -B main.py b3 ours predict --device cuda --output results/b3/latest/new_ours
python3 -B main.py b3 ours replay --output results/b3/latest/replayed_ours
python3 -B main.py b3 ours evaluate --prediction results/b3/latest/new_ours/predictions.npz --output results/b3/latest/new_score
python3 -B main.py b3 ours verify
```

predict는 관측 특징에서 실제 forward를 실행하고, replay는 동결된 원 예측을 읽는다. 둘을 같은 새 추론이라고 부르지 않는다. evaluate는240,890개의 정확한 sample_id 순서를 요구한다. ADE/FDE를 GT로 다시 계산하고, 예측이 원 동결 배열과 완전히 같은 파일 구간에서는 SHA로 검증한 기존 충돌 판정을 재사용한다. 값이 다르면 원 scorer의 endpoint/midpoint 충돌을 다시 계산한다. 충돌은 관측에서 보인 이웃의 미래 GT 기준이며 예측 이웃 Col-II가 아니다. 거리 단위는 원 좌표 단위다.

GPU에서 baseline·ours 각각240,890장면을 실제 실행해 원 예측과 최대차이0을 확인했다. GT와 참조 예측 파일 open을 차단한 상태다. CPU는 지원 경로이지만 이 전체 동일성 검증은 GPU0에서 수행했다.

## 보존과 출처

기존 static residual 배포의 소스·설명·branch 설정은 legacy_20261112에 보존했다. 기존 data/b3/ours와 기존 결과는 그대로다. 최신 독립 자산은 data/b3/latest에 복사하고 manifest.json에 원본·배포 SHA를 기록했다. 원 실험 파일을 수정하지 않았으며 runtime_guard의 외부 접근 경계도 유지했다.

원 실험: HWP_CJ4DPLEX_AI_support/iterative_research_20261111/b3/diagnosis_targeted_experiments. 미래 GT는 평가와 Demo에서만 읽는다. Demo는 실제 UCY 원영상이 있는20사례이며 전체 평가를 대표하는 무작위 표본이 아니다.

