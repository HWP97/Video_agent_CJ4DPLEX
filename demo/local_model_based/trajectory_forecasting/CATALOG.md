# Trajectory Forecasting (Local) — Demo Catalog

최종 선정 **5개** 사례. 각 사례의 Baseline → Ours 지표 비교와 sample ID. 관측 1x · cutoff 이후 0.5x 재생. 원본 MP4·포스터는 외부 저장소에 있으며 본 Repository에는 catalog·평가·예측 metadata만 포함한다.

| Demo | sample_id | 지표 | Baseline | Ours |
|---:|---|---|---:|---:|
| 01 | `crowds_zara01_57` | ADE | 0.0533 | 0.0877 |
| 02 | `crowds_students003_3356` | ADE | 0.3229 | 0.0957 |
| 03 | `crowds_zara01_222` | ADE | 0.1204 | 0.1109 |
| 04 | `crowds_students003_2473` | ADE | 0.0705 | 0.0842 |
| 05 | `crowds_zara01_258` | ADE | 0.3909 | 0.0754 |

카탈로그 기록(`catalog.json`)에는 위 정보 외에 영상·포스터·예측 경로, 범주, metric_delta 가 함께 들어 있다.