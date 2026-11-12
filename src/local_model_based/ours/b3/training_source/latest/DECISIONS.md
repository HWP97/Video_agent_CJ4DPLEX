# b3 진단 후속 실험 사전 결정

사용자의 EXPERIMENT_PROMPT.md 실행 지시에 따른다. 기존 원본과 진단 폴더는 읽기 전용이다. 유료 API, 모델 다운로드, 환경 설치, 서비스 설정 변경은 하지 않는다. GPU0과 CPU 1 thread를 사용한다.

## 추가학습 대조

D01은 저장된 새 F(추가학습 전), D02는 완료 C01 예측을 SHA와 순서를 확인해 재사용한다. D03/D04는 같은 기존 F parent에서 시작한다. AdamW lr0.0001, weight_decay0.01, batch256, gradient clip5, 최대10epoch, seed20261111로 고정한다. D03은 optimizer 재시작, D04는 호환되는 parent moment와 step을 유지하고 lr만 바꾼다. 두 조건은 같은 parent 가중치와 RNG 및 배치 순서로 시작한다. 내부 검증 ADE로 epoch0부터 최소를 선택하며 동률은 짧은 epoch를 택한다. full refit은 각 fold 내부에서 정한 추가 epoch만 수행한다. D02의 lr0.001, 0epoch 제외, 역사적 학습 절차 차이를 명시한다.

## 기본 F 궤적

원본 two_state_deep F와 관측125차원 특징을 그대로 사용한다. G/P/U는 추가하지 않는다. AdamW lr0.001, weight_decay0.01, batch256, clip5, seed20261111 및20261110이다. 최대25epoch의 한 궤적에서 epoch1/3/5/10/20/25만 후보 checkpoint로 남긴다. 각 epoch를 처음부터 따로 학습하지 않는다. 정규화는 해당 내부 train에서만 fit한다.

기존 내부 검증 그룹은 fold1 cff_07, fold2 crowds_zara01, fold3 biwi_hotel이다. 추가 분리는 관측 그룹명의 정렬 순서에서 기존 그룹을 제외한 첫 그룹인 cff_06/crowds_students001/lcas를 가린다. 그룹 수는 각각12/4/3이며 각 분리는 검증 그룹 전체를 학습에서 제외한다. frame 임의 분할은 없다.

각 fold와 seed에서 두 분리의 그룹별 ADE/Kalman ADE 비율 중 최댓값이 최소인 epoch를 robust 정책으로 고른다. 기존 한 분리의 ADE 최소 정책도 동일한6개 후보에서 비교한다. 동률은 짧은 epoch다. 다른 outer fold 점수나 최종 평가 정답을 선택에 합치지 않는다. 모든 fold의 선택 lock을 만든 뒤에만 전체 train refit과 최종 추론을 시작한다. 정책별 두 seed 모델을 따로 평가하며 seed를 test로 선택하지 않는다. 동일 seed에서 정책 epoch가 같으면 fit와 예측을 재사용한다.

## 평가와 해석

관측만 읽는 추론과 GT를 읽는 채점을 별도 프로세스로 수행한다. 최종 범위는3fold 240,890장면 전수이며 ADE/FDE 및 원본 collision_segments의 관측 이웃 충돌률을 사용한다. 원본 attention과 과거 F 및 C01은 완료 예측/채점의 SHA와 입력 순서를 검증해 재사용한다. 새 F0 충돌률은 실제 원본 scorer로 계산한다. 최소512사례 예측 재생, 독립 집계, SHA, 누락/중복 검사와 epoch0 거절 검증을 남긴다.

반복 열람한 평가 영역이며 독립 holdout으로 부르지 않는다. 과거 F와의 차이는 seed·분할·학습길이·선택 정책이 얽힌 역사적 참조다. 이번 통제 내에서만 seed, epoch, optimizer moment의 차이를 해석한다. 성능이 개선되지 않아도 탐색을 늘리지 않는다.
