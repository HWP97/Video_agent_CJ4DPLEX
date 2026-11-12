# 최신 동일 평가집합의 로컬 attention

현재 baseline은 최신 D04와 같은3fold240,890장면의 동결 attention이다. `predict`는 실제 관측 특징부터 forward를 실행하고 `replay`는 저장 출력을 재생한다. 기존1,943장면의 진입점과 branch 설정은 legacy_20261112에 보존했고 기존 데이터와 결과도 유지했다.

계열 루트에서 `python3 -B main.py b3 baseline status|replay|predict|evaluate|verify`를 사용한다. `predict --device cuda`는 GPU를 사용한다. `evaluate --prediction <predictions.npz>`는 sample_id 순서를 함께 확인하며 최신 데이터와 다른 예측을 혼합하지 않는다. 새 출력 폴더를 지정해야 하며 이미 있는 결과를 덮어쓰지 않는다.

검증 수치: ADE1.0522110911774152, FDE2.1705463989021982, 관측 이웃 GT 충돌률22.04616214869858%. 충돌 정의와 최신 실행 계약은 `../../ours/b3/README.md`를 따른다. 기존 결과의 예측 이웃 Col-II와 같은 지표라고 표시하지 않는다.
