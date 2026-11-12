# 최신 보행 로컬 모델의 실험·학습 출처

현재 배포는 B3-D04다. 이전 static residual의 상세 문서는 legacy_20261112/EXPERIMENTS_AND_TRAINING.md에 보존했다. 최신 baseline·ours는 같은3fold240,890장면을 사용하며 기존1,943장면과 별개다.

학습 자료는 기존에 확보한 TrajNet++ 공개 real train의 관측·미래 좌표다. outer 그룹은 CFF, UCY, 기타 사이트의 사용자 정의3그룹이며 원문 challenge hidden test 재현이 아니다. 각 fold는 자기 train 내부에서 epoch를 선택하고 전체 train refit 뒤 다른 그룹을 평가했다. 정확한 그룹과 파일은 data/b3/latest/runplan.json을 따른다.

D04는 기존 F parent, 같은 RNG·배치 순서의 D03 대조와 비교하여 optimizer moment와 step을 보존한 조건이다. 학습률0.0001·weight_decay0.01·batch256·clip5·최대10epoch·seed20261111이며 epoch0을 후보에 포함했다. fold1은5epoch, fold2/3은0epoch를 선택했다. 손해인 추가 학습을 거절하는 동작은 검증됐지만 최종 평가 개선은 보장하지 못했다.

F parent와 이번 D04에는 Sol 응답·교사 증류가 없다. 원 데이터 미래 정답은 학습 감독이고 배포 추론 입력에는 없다. 배포 준비에서 새 학습·API·모델 다운로드는0회다. 현재 모델은 최신 조건의 명시적 배포 선택이며 test 지표가 가장 좋은 모델을 골랐다는 주장은 하지 않는다.

원 실험의 DECISIONS.md·RESULTS.md·COMPLETION_RECEIPT.json·selection_fold1/2/3.json이 실험 근거다. 배포 manifest.json은 복사한 동결 checkpoint·관측·평가·참조 예측114파일의 원본과 설치 SHA를 남긴다. 원 실험 보고서는 외부 support 폴더에 보존한다.

