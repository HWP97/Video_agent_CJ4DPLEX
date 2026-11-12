# Affection Forecasting (Local) — Demo Catalog

최종 선정 **5개** 사례. 각 사례의 Baseline → Ours 지표 비교와 sample ID. 관측 1x · cutoff 이후 0.5x 재생. 원본 MP4·포스터는 외부 저장소에 있으며 본 Repository에는 catalog·평가·예측 metadata만 포함한다.

| Demo | sample_id | 지표 | Baseline | Ours |
|---:|---|---|---:|---:|
| 01 | `v003_official_test_t00139.0400` | MAE | 0.3731 | 0.1037 |
| 02 | `v067_official_test_t00074.0000` | MAE | 0.3396 | 0.1181 |
| 03 | `v094_official_test_t00074.7600` | MAE | 0.7203 | 0.5624 |
| 04 | `v060_official_test_t00046.6000` | MAE | 0.1819 | 0.0275 |
| 05 | `v058_official_test_t00054.6800` | MAE | 0.4186 | 0.3301 |

카탈로그 기록(`catalog.json`)에는 위 정보 외에 영상·포스터·예측 경로, 범주, metric_delta 가 함께 들어 있다.