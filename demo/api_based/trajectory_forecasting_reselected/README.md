# API 궤적 예측 Demo 재선정

기존1번은 파일 그대로 유지하고 기존2~20번19개 sample ID를 제외했다. GT 평균/끝/최대 오차12/20/25px 기준을 유지하면서, 새19개는 baseline보다 평균오차1px 초과 감소·두 예측의 평균거리4px 이상·최대거리8px 이상을 만족한다. 모든 값은 원720×576 기준이다.

Scene과 시간 구간 중복은 허용한다. 가능한 다른 인물을 먼저 선정한 후 수량을 채웠으며독립 원영상이 아니다. 관측1배속·미래0.5배속, 원 예측과 전체 평가 점수는 변경하지 않았다. .

|번호|사례|MP4|WebM|GT 평균오차(px)|평균 예측 간 거리(px)|
|---|---|---|---|---:|---:|
|01|crowds_zara01_576|[영상](demo_01_crowds_zara01_576.mp4)|[브라우저용](demo_01_crowds_zara01_576.webm)|6.72|16.01|
|02|crowds_students003_3349|[영상](demo_02_crowds_students003_3349.mp4)|[브라우저용](demo_02_crowds_students003_3349.webm)|3.27|13.85|
|03|crowds_zara01_631|[영상](demo_03_crowds_zara01_631.mp4)|[브라우저용](demo_03_crowds_zara01_631.webm)|6.86|14.72|
|04|crowds_zara01_675|[영상](demo_04_crowds_zara01_675.mp4)|[브라우저용](demo_04_crowds_zara01_675.webm)|7.76|12.02|
|05|crowds_students003_3499|[영상](demo_05_crowds_students003_3499.mp4)|[브라우저용](demo_05_crowds_students003_3499.webm)|5.45|11.42|
|06|crowds_zara01_708|[영상](demo_06_crowds_zara01_708.mp4)|[브라우저용](demo_06_crowds_zara01_708.webm)|9.59|9.43|
|07|crowds_students003_3679|[영상](demo_07_crowds_students003_3679.mp4)|[브라우저용](demo_07_crowds_students003_3679.webm)|9.66|13.82|
|08|crowds_students003_3432|[영상](demo_08_crowds_students003_3432.mp4)|[브라우저용](demo_08_crowds_students003_3432.webm)|4.51|10.21|
|09|crowds_students003_3654|[영상](demo_09_crowds_students003_3654.mp4)|[브라우저용](demo_09_crowds_students003_3654.webm)|7.68|12.14|
|10|crowds_zara01_772|[영상](demo_10_crowds_zara01_772.mp4)|[브라우저용](demo_10_crowds_zara01_772.webm)|10.01|6.56|
|11|crowds_students003_3570|[영상](demo_11_crowds_students003_3570.mp4)|[브라우저용](demo_11_crowds_students003_3570.webm)|8.22|6.59|
|12|crowds_zara01_601|[영상](demo_12_crowds_zara01_601.mp4)|[브라우저용](demo_12_crowds_zara01_601.webm)|6.22|5.79|
|13|crowds_students003_3343|[영상](demo_13_crowds_students003_3343.mp4)|[브라우저용](demo_13_crowds_students003_3343.webm)|11.14|10.35|
|14|crowds_zara01_613|[영상](demo_14_crowds_zara01_613.mp4)|[브라우저용](demo_14_crowds_zara01_613.webm)|9.43|5.13|
|15|crowds_students003_3417|[영상](demo_15_crowds_students003_3417.mp4)|[브라우저용](demo_15_crowds_students003_3417.webm)|1.65|4.88|
|16|crowds_zara01_590|[영상](demo_16_crowds_zara01_590.mp4)|[브라우저용](demo_16_crowds_zara01_590.webm)|6.80|9.59|
|17|crowds_students003_3500|[영상](demo_17_crowds_students003_3500.mp4)|[브라우저용](demo_17_crowds_students003_3500.webm)|4.54|8.07|
|18|crowds_students003_3501|[영상](demo_18_crowds_students003_3501.mp4)|[브라우저용](demo_18_crowds_students003_3501.webm)|7.68|5.16|
|19|crowds_zara01_707|[영상](demo_19_crowds_zara01_707.mp4)|[브라우저용](demo_19_crowds_zara01_707.webm)|9.04|7.02|
|20|crowds_students003_3647|[영상](demo_20_crowds_students003_3647.mp4)|[브라우저용](demo_20_crowds_students003_3647.webm)|3.59|7.48|
