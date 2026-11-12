# 로컬 궤적 예측 Demo

최신 동일3fold 평가 중 실제 UCY 원영상2개에서10사례씩 선정했다. 원720×576에서 ours 평균/끝/최대 오차12/20/25px 이하, GT 이동길이와 시작→끝 변위30px 이상, GT와 ours 미래점 모두 영상 안 조건을 먼저 적용했다. 같은 인물은 한 번만, 같은 원영상의8초 구간은 비중복으로 MILP 동시 선택했다. 두 원영상을 번갈아 배치한다. 전체 평가의 무작위 표본이 아니며 전체 성능 개선을 뜻하지 않는다. 실제 기준값은 selection_manifest.json을 따른다.

관측9점·3.2초는1배속, 미래12점·4.8초는0.5배속으로 표시한다. 원영상25fps·프레임 offset0·VSP420점 최대축오차0.005 이내를 확인했다. 거리 지표는 원 좌표 단위이며 충돌은 관측 이웃 GT 기준이다.

|Demo|사례|Baseline ADE|D04 ADE|영상|
|---|---|---:|---:|---|
|01|crowds_zara01_57|0.053259|0.087715|[MP4](demo_01_crowds_zara01_57.mp4)|
|02|crowds_students003_3356|0.322897|0.095725|[MP4](demo_02_crowds_students003_3356.mp4)|
|03|crowds_zara01_222|0.120381|0.110927|[MP4](demo_03_crowds_zara01_222.mp4)|
|04|crowds_students003_2473|0.070535|0.084165|[MP4](demo_04_crowds_students003_2473.mp4)|
|05|crowds_zara01_258|0.390896|0.075422|[MP4](demo_05_crowds_zara01_258.mp4)|
|06|crowds_students003_642|0.234845|0.086737|[MP4](demo_06_crowds_students003_642.mp4)|
|07|crowds_zara01_372|0.076065|0.070813|[MP4](demo_07_crowds_zara01_372.mp4)|
|08|crowds_students003_2456|0.089844|0.081960|[MP4](demo_08_crowds_students003_2456.mp4)|
|09|crowds_zara01_389|0.232035|0.119762|[MP4](demo_09_crowds_zara01_389.mp4)|
|10|crowds_students003_704|0.126332|0.043860|[MP4](demo_10_crowds_students003_704.mp4)|
|11|crowds_zara01_556|0.078818|0.082666|[MP4](demo_11_crowds_zara01_556.mp4)|
|12|crowds_students003_3526|0.053763|0.060256|[MP4](demo_12_crowds_students003_3526.mp4)|
|13|crowds_zara01_671|0.154980|0.084682|[MP4](demo_13_crowds_zara01_671.mp4)|
|14|crowds_students003_2823|0.083738|0.036503|[MP4](demo_14_crowds_students003_2823.mp4)|
|15|crowds_zara01_777|0.040472|0.070832|[MP4](demo_15_crowds_zara01_777.mp4)|
|16|crowds_students003_2964|0.072165|0.022623|[MP4](demo_16_crowds_students003_2964.mp4)|
|17|crowds_zara01_868|0.086946|0.078714|[MP4](demo_17_crowds_zara01_868.mp4)|
|18|crowds_students003_3101|0.074912|0.048214|[MP4](demo_18_crowds_students003_3101.mp4)|
|19|crowds_zara01_888|0.115374|0.113797|[MP4](demo_19_crowds_zara01_888.mp4)|
|20|crowds_students003_2297|0.175558|0.104813|[MP4](demo_20_crowds_students003_2297.mp4)|

원영상 출처와 라이선스는 기존 UCY 원영상 기록을 따른다. 예측·정답 전체는 cases.json, 투영 원주석 대조는 projection_verification.json, 영상 검증은 verification.json에 있다.
