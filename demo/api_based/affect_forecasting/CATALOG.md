# Affection Forecasting (API) — Demo Catalog

최종 선정 **5개** 사례. 각 사례의 Baseline → Ours 지표 비교와 sample ID. 관측 1x · cutoff 이후 0.5x 재생. 원본 MP4·포스터는 외부 저장소에 있으며 본 Repository에는 catalog·평가·예측 metadata만 포함한다.

| Demo | sample_id | 지표 | Baseline | Ours |
|---:|---|---|---:|---:|
| 01 | `v099_t0071.00` | MAE | 0.5193 | 0.2606 |
| 02 | `v110_t0029.00` | MAE | 0.3775 | 0.1464 |
| 03 | `v034_t0071.00` | MAE | 0.3706 | 0.1740 |
| 04 | `v050_t0068.00` | MAE | 0.3813 | 0.2141 |
| 05 | `v038_t0104.00` | MAE | 0.1848 | 0.0448 |

카탈로그 기록(`catalog.json`)에는 위 정보 외에 영상·포스터·예측 경로, 범주, metric_delta 가 함께 들어 있다.