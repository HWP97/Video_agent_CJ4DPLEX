# 관계 예측 demo

관측 50%에서 같은 사례의 NC R@20 차이를 기준으로 원영상 ID 중복 없이 demo 20개를 골랐다. 값은 %이며 차이는 %p다. 이미 본 평가 결과로 선별한 시연 사례이며 전체 성능의 무작위 표본이 아니다.

연속 원영상과 정확한 주석 프레임의 GT 박스를 별도로 보여 준다. 관계 목록은 해당 주석 시점의 공식 NC 상위 20개 순서다. GT 관계는 공간 제약에 따라 화면 일부만 표시하고 전체는 사례 evaluation JSON에 보존한다. 파란색은 ours, 빨간색은 GT다. 박스는 평가용 실제 위치이며 미래 박스 예측으로 해석하지 않는다.

|구분|순위|사례|NC baseline|NC ours|차이|WC baseline|WC ours|영상|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
|demo|1|predcls:X2IUY.mp4_0.5|22.222|100.000|+77.778|22.222|94.444|[demo_01_predcls_X2IUY.mp4](demo_01_predcls_X2IUY.mp4)|
|demo|2|sgcls:U588C.mp4_0.5|15.873|85.714|+69.841|26.984|69.841|[demo_02_sgcls_U588C.mp4](demo_02_sgcls_U588C.mp4)|
|demo|3|sgcls:9SJZ9.mp4_0.5|33.333|100.000|+66.667|66.667|72.222|[demo_03_sgcls_9SJZ9.mp4](demo_03_sgcls_9SJZ9.mp4)|
|demo|4|predcls:KOQGE.mp4_0.5|30.936|95.040|+64.104|43.602|66.768|[demo_04_predcls_KOQGE.mp4](demo_04_predcls_KOQGE.mp4)|
|demo|5|sgcls:KOVTR.mp4_0.5|34.568|98.148|+63.580|67.284|65.432|[demo_05_sgcls_KOVTR.mp4](demo_05_sgcls_KOVTR.mp4)|
|demo|6|predcls:V1BX9.mp4_0.5|13.532|75.408|+61.876|26.548|74.388|[demo_06_predcls_V1BX9.mp4](demo_06_predcls_V1BX9.mp4)|
|demo|7|predcls:YDWIW.mp4_0.5|24.936|86.653|+61.717|38.158|81.058|[demo_07_predcls_YDWIW.mp4](demo_07_predcls_YDWIW.mp4)|
|demo|8|predcls:OPMF7.mp4_0.5|41.667|100.000|+58.333|41.667|83.333|[demo_08_predcls_OPMF7.mp4](demo_08_predcls_OPMF7.mp4)|
|demo|9|predcls:WC5QK.mp4_0.5|35.451|92.301|+56.850|43.530|58.019|[demo_09_predcls_WC5QK.mp4](demo_09_predcls_WC5QK.mp4)|
|demo|10|predcls:0UPBN.mp4_0.5|41.497|97.619|+56.122|22.789|19.048|[demo_10_predcls_0UPBN.mp4](demo_10_predcls_0UPBN.mp4)|
|demo|11|predcls:J48N6.mp4_0.5|44.118|100.000|+55.882|21.569|67.157|[demo_11_predcls_J48N6.mp4](demo_11_predcls_J48N6.mp4)|
|demo|12|predcls:L2502.mp4_0.5|41.468|97.222|+55.754|54.861|72.817|[demo_12_predcls_L2502.mp4](demo_12_predcls_L2502.mp4)|
|demo|13|sgcls:DL6HP.mp4_0.5|27.778|83.333|+55.556|36.111|0.000|[demo_13_sgcls_DL6HP.mp4](demo_13_sgcls_DL6HP.mp4)|
|demo|14|predcls:1QP1S.mp4_0.5|41.905|96.720|+54.815|20.317|77.989|[demo_14_predcls_1QP1S.mp4](demo_14_predcls_1QP1S.mp4)|
|demo|15|predcls:FYYFX.mp4_0.5|31.922|86.552|+54.630|47.354|46.429|[demo_15_predcls_FYYFX.mp4](demo_15_predcls_FYYFX.mp4)|
|demo|16|predcls:E6DLK.mp4_0.5|40.714|95.238|+54.524|39.048|53.810|[demo_16_predcls_E6DLK.mp4](demo_16_predcls_E6DLK.mp4)|
|demo|17|sgcls:LGJAR.mp4_0.5|39.598|91.518|+51.920|17.679|55.268|[demo_17_sgcls_LGJAR.mp4](demo_17_sgcls_LGJAR.mp4)|
|demo|18|sgcls:K1QIV.mp4_0.5|45.455|96.970|+51.515|75.758|33.333|[demo_18_sgcls_K1QIV.mp4](demo_18_sgcls_K1QIV.mp4)|
|demo|19|sgcls:O1PD3.mp4_0.5|48.810|100.000|+51.190|55.357|73.512|[demo_19_sgcls_O1PD3.mp4](demo_19_sgcls_O1PD3.mp4)|
|demo|20|predcls:2ZDFA.mp4_0.5|26.222|76.444|+50.222|33.778|46.889|[demo_20_predcls_2ZDFA.mp4](demo_20_predcls_2ZDFA.mp4)|

## 색과 재생 속도

baseline/ours 목록에서 공식 NC 평가기가 GT와 일치한다고 판정한 관계만 연두색이다. 객체 클래스·관계 방향·predicate와 기존 IoU=0 조건을 그대로 사용한다. 문자열이 같다는 이유만으로 색칠하지 않는다. 각 상위20 예측이 일치한 GT 행 번호는 evaluation JSON에 보존한다.

관측은 원래 속도(1x), cutoff 이후 미래 구간만 절반 속도(0.5x)로 재생한다. 화면 Source t와 cutoff는 원본 시간이며 지표·예측·GT·Cosmos 입력은 바뀌지 않는다. 영상 길이는 cutoff + 2 × (미래 종료 − cutoff)이고 출력15fps의 한 프레임 범위 오차만 허용한다. 기존 파일을 새 렌더로 교체하려면 `--replace`를 명시한다.

## 재현과 출처

프로젝트 루트에서 기존 관계 예측 환경 Python으로 `python -B tools/api_demo/b1_demo.py`를 실행한다. 기존 영상은 덮어쓰지 않고 ffprobe로 검사한다. 새 출력을 만들려면 `--output`에 프로젝트 안의 새 디렉터리를 지정한다.

선정 원본은 `results/api_based/b1/frozen/results_full_*/per_case.jsonl`의 sol/macro_direct다. 로컬 원영상·frame_records는 `/mnt/RG2/my_works/b1/benchmarks/action_genome_20261106/data/`에 있다. 기존 API 번들의 prepared 자료에서 예측 entry와 관측 feature만 decoder에 전달하고, 미래 GT는 채점·렌더링 단계에 사용한다. 20개 사례의 baseline/ours 공식 재채점 평균이 frozen 값과 절대오차 1e-10 이내인지 생성 시 검사한다.

원영상 권리와 배포 허가는 별도 확인 대상이다. 로컬 시각화 제작이 공개 재배포 허가를 의미하지 않는다. [Action Genome 원 논문](https://arxiv.org/abs/1912.06992)의 사람·객체 관계 표현과 [SceneSayer](https://github.com/rohithpeddi/SceneSayer)의 미래 관계 비교를 참고했다.
