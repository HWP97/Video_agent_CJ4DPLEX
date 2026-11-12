# 로컬 관계 예측 Demo

동일 사례의 NC R@20 차이로 원영상 중복 없이20개를 선택했다. 모집단의 무작위 표본이 아니다. 실제 ImparTail 관측 prior에서 최신 선택 head를 재계산한 뒤 공식 평가기로 GT와 대조했다. 미래 GT는 렌더링·평가에서만 읽는다.

|Demo|사례|Baseline|Ours|영상|
|---:|---|---:|---:|---|
|1|predcls:GF7DV.mp4_0.5|46.154|84.615|[demo_01_predcls_GF7DV.mp4](demo_01_predcls_GF7DV.mp4)|
|2|predcls:6PESI.mp4_0.5|60.000|95.000|[demo_02_predcls_6PESI.mp4](demo_02_predcls_6PESI.mp4)|
|3|predcls:DLFPX.mp4_0.5|65.476|100.000|[demo_03_predcls_DLFPX.mp4](demo_03_predcls_DLFPX.mp4)|
|4|predcls:4ALLQ.mp4_0.5|58.681|91.667|[demo_04_predcls_4ALLQ.mp4](demo_04_predcls_4ALLQ.mp4)|
|5|predcls:NDDHA.mp4_0.5|47.460|80.243|[demo_05_predcls_NDDHA.mp4](demo_05_predcls_NDDHA.mp4)|
|6|predcls:LXLW4.mp4_0.5|61.041|92.478|[demo_06_predcls_LXLW4.mp4](demo_06_predcls_LXLW4.mp4)|
|7|predcls:YB7MW.mp4_0.5|60.000|90.000|[demo_07_predcls_YB7MW.mp4](demo_07_predcls_YB7MW.mp4)|
|8|sgcls:KTDG0.mp4_0.5|28.571|57.143|[demo_08_sgcls_KTDG0.mp4](demo_08_sgcls_KTDG0.mp4)|
|9|predcls:CG83G.mp4_0.5|58.333|85.185|[demo_09_predcls_CG83G.mp4](demo_09_predcls_CG83G.mp4)|
|10|predcls:GF6IX.mp4_0.5|63.111|89.889|[demo_10_predcls_GF6IX.mp4](demo_10_predcls_GF6IX.mp4)|
|11|predcls:MRTH4.mp4_0.5|68.268|94.643|[demo_11_predcls_MRTH4.mp4](demo_11_predcls_MRTH4.mp4)|
|12|predcls:HB02D.mp4_0.5|44.375|70.625|[demo_12_predcls_HB02D.mp4](demo_12_predcls_HB02D.mp4)|
|13|predcls:ODD3K.mp4_0.5|59.176|85.400|[demo_13_predcls_ODD3K.mp4](demo_13_predcls_ODD3K.mp4)|
|14|sgcls:BDZNW.mp4_0.5|50.000|76.190|[demo_14_sgcls_BDZNW.mp4](demo_14_sgcls_BDZNW.mp4)|
|15|predcls:Y3AQ4.mp4_0.5|66.953|92.533|[demo_15_predcls_Y3AQ4.mp4](demo_15_predcls_Y3AQ4.mp4)|
|16|predcls:NFA8R.mp4_0.5|44.444|69.697|[demo_16_predcls_NFA8R.mp4](demo_16_predcls_NFA8R.mp4)|
|17|predcls:39I5H.mp4_0.5|60.278|84.444|[demo_17_predcls_39I5H.mp4](demo_17_predcls_39I5H.mp4)|
|18|sgcls:8H570.mp4_0.5|35.648|59.722|[demo_18_sgcls_8H570.mp4](demo_18_sgcls_8H570.mp4)|
|19|predcls:13XM4.mp4_0.5|59.387|83.097|[demo_19_predcls_13XM4.mp4](demo_19_predcls_13XM4.mp4)|
|20|predcls:BPZ55.mp4_0.5|65.833|89.167|[demo_20_predcls_BPZ55.mp4](demo_20_predcls_BPZ55.mp4)|

관측50%는 주석 프레임 기준이다. 원 PTS를 유지하고 미래만0.5배속으로 표시한다. 연두색 GT 일치는 공식 NC 평가기의 클래스·방향·관계·IoU=0 판정이며 GT 위치는 예측 박스가 아니다. 원영상 전달 권리는 제출 전에 별도로 확인한다.
