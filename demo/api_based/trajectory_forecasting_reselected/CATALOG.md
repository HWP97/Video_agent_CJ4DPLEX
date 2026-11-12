# Trajectory Forecasting (API · reselected) — Demo Catalog

최종 선정 **5개** 사례. 각 사례의 Baseline → Ours 지표 비교와 sample ID. 관측 1x · cutoff 이후 0.5x 재생. 원본 MP4·포스터는 외부 저장소에 있으며 본 Repository에는 catalog·평가·예측 metadata만 포함한다.

| Demo | sample_id | 지표 | Baseline | Ours |
|---:|---|---|---:|---:|
| 01 | `crowds_zara01_576` | ADE | 0.3357 | 0.1485 |
| 02 | `crowds_students003_3349` | ADE | 0.3406 | 0.0744 |
| 03 | `crowds_students003_3499` | ADE | 0.3545 | 0.1196 |
| 04 | `crowds_students003_3679` | ADE | 0.4006 | 0.2172 |
| 05 | `crowds_students003_3343` | ADE | 0.3291 | 0.2448 |

카탈로그 기록(`catalog.json`)에는 위 정보 외에 영상·포스터·예측 경로, 범주, metric_delta 가 함께 들어 있다.