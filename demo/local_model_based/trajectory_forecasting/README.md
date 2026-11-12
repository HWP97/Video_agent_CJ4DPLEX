# 로컬 궤적 예측 Demo

아래 번호로 새로 선정한20개 영상이다. 기존 영상과 분리한 이 폴더에서 확인하면 된다.

- 실제 원영상은 **UCY zara01·students003 두 개**이며 각각10사례다. 두 원영상을 번갈아 배치했다.
- 같은 인물은 한 번만 선택했고, 같은 원영상 안의8초 구간은 서로 겹치지 않는다.
- 원720×576 기준 ours 평균/끝/최대 오차12/20/25px 이하를 먼저 적용했다. 실제 선정20개의 최대치는 각각 **5.37/13.12/13.12px**다.
- GT 이동길이와 시작→끝 변위가30px 이상이며 GT·ours 미래점은 전부 원영상 안에 있다. 정지 궤적만 모은 목록이 아니다.
- 관측3.2초는1배속, 미래4.8초는0.5배속이다. 파랑은 ours D04, 빨강은 GT, 주황은 baseline attention이다.
- GT를 이용한 Demo 선별이며 전체 평가 점수나 모델을 바꾸지 않았다. 좋은 예시 목록을 전체 성능 개선으로 해석하지 않는다.

|번호|사례|MP4|WebM|평균 / 끝 오차(px)|
|---|---|---|---|---:|
|01|crowds_zara01_57|[영상](demo_01_crowds_zara01_57.mp4)|[브라우저용](demo_01_crowds_zara01_57.webm)|3.70 / 6.81|
|02|crowds_students003_3356|[영상](demo_02_crowds_students003_3356.mp4)|[브라우저용](demo_02_crowds_students003_3356.webm)|4.11 / 3.02|
|03|crowds_zara01_222|[영상](demo_03_crowds_zara01_222.mp4)|[브라우저용](demo_03_crowds_zara01_222.webm)|4.87 / 13.11|
|04|crowds_students003_2473|[영상](demo_04_crowds_students003_2473.mp4)|[브라우저용](demo_04_crowds_students003_2473.webm)|3.82 / 7.62|
|05|crowds_zara01_258|[영상](demo_05_crowds_zara01_258.mp4)|[브라우저용](demo_05_crowds_zara01_258.webm)|3.27 / 4.69|
|06|crowds_students003_642|[영상](demo_06_crowds_students003_642.mp4)|[브라우저용](demo_06_crowds_students003_642.webm)|3.92 / 5.97|
|07|crowds_zara01_372|[영상](demo_07_crowds_zara01_372.mp4)|[브라우저용](demo_07_crowds_zara01_372.webm)|3.21 / 7.89|
|08|crowds_students003_2456|[영상](demo_08_crowds_students003_2456.mp4)|[브라우저용](demo_08_crowds_students003_2456.webm)|3.69 / 8.10|
|09|crowds_zara01_389|[영상](demo_09_crowds_zara01_389.mp4)|[브라우저용](demo_09_crowds_zara01_389.webm)|5.23 / 8.11|
|10|crowds_students003_704|[영상](demo_10_crowds_students003_704.mp4)|[브라우저용](demo_10_crowds_students003_704.webm)|2.05 / 1.98|
|11|crowds_zara01_556|[영상](demo_11_crowds_zara01_556.mp4)|[브라우저용](demo_11_crowds_zara01_556.webm)|3.58 / 6.73|
|12|crowds_students003_3526|[영상](demo_12_crowds_students003_3526.mp4)|[브라우저용](demo_12_crowds_students003_3526.webm)|2.59 / 2.52|
|13|crowds_zara01_671|[영상](demo_13_crowds_zara01_671.mp4)|[브라우저용](demo_13_crowds_zara01_671.webm)|3.88 / 10.16|
|14|crowds_students003_2823|[영상](demo_14_crowds_students003_2823.mp4)|[브라우저용](demo_14_crowds_students003_2823.webm)|1.62 / 5.11|
|15|crowds_zara01_777|[영상](demo_15_crowds_zara01_777.mp4)|[브라우저용](demo_15_crowds_zara01_777.webm)|3.24 / 6.28|
|16|crowds_students003_2964|[영상](demo_16_crowds_students003_2964.mp4)|[브라우저용](demo_16_crowds_students003_2964.webm)|1.02 / 2.40|
|17|crowds_zara01_868|[영상](demo_17_crowds_zara01_868.mp4)|[브라우저용](demo_17_crowds_zara01_868.webm)|3.59 / 10.13|
|18|crowds_students003_3101|[영상](demo_18_crowds_students003_3101.mp4)|[브라우저용](demo_18_crowds_students003_3101.webm)|2.24 / 3.70|
|19|crowds_zara01_888|[영상](demo_19_crowds_zara01_888.mp4)|[브라우저용](demo_19_crowds_zara01_888.webm)|5.37 / 4.14|
|20|crowds_students003_2297|[영상](demo_20_crowds_students003_2297.mp4)|[브라우저용](demo_20_crowds_students003_2297.webm)|4.56 / 3.45|

선정 기준과 실제 값은 [selection_manifest.json](selection_manifest.json), 원주석420점 대조는 [projection_verification.json](projection_verification.json), 영상 검증은 [verification.json](verification.json)을 참고한다. 이전20개는 support의 `submission_preparation_20261111/trajectory_before_accuracy_selection/local_model_based/`에 보존했다.

