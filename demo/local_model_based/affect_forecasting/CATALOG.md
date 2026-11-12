# 감정 상태 예측 Demo

최신 official_test438에서 같은 사례의 **C01 직접 미래 GRU baseline**과 D04를 비교했다. C01 MAE−D04 MAE 순서로 정렬하고 원본 그룹별 한 구간씩20개를 골랐다. 모집단을 대표하는 무작위 표본이 아니다. C01은 현재값 head가 없어 current 평가는 해당 없음이다. hold는 추가 비교군이며 이 Demo의 baseline이 아니다.

실제 모델의 특징 프레임 구간1.8초를 관측으로 보여주고 미래3초를0.5배속으로 재생한다. manifest의 보수적8초 관측 시작과 실제 특징15프레임 범위를 구분한다. 영상은196프레임·25fps·7.84초이며 점수와 표시 시각은 원본 기준이다. 예측값은 원 평가와 같이 [-1,1]로 제한한다.

`catalog.json`에20개 영상·포스터·동일 사례의 지표가, `cases.json`에 고정 예측·GT·원시각이 있다. `provenance.json`은 선택 입력 SHA다. 원영상은 기존 VEATIC 자산을 사용한다. 원영상 전달·공개 권리는 별도 확인 대상이다.

`target_review_0.jpg`부터 `target_review_2.jpg`까지 새선정20개의 관측 종료 프레임을 직접 검토했다. 확실한9개 대상에만 머리/얼굴 ROI를 지정했으며 가림·불확실11개는 null로 남겼다. ROI는 표시용이며 모델이 사람을 검출하거나 감정을 새로 인식했다는 뜻이 아니다. 추적은 가림·컷·점 부족에서 중단하고 원영상과 수치 비교는 계속 제공한다.

기존 hold/D04 Demo는 별도 support의 `submission_preparation_20261111/local_affect_hold_archive`에 SHA와 함께 보존했다. 현재 공개 파일은 모두 새 C01/D04 비교다.
