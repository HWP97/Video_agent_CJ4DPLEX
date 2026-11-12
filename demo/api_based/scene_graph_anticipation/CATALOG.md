# Scene Graph Anticipation (API) — Demo Catalog

최종 선정 **5개** 사례. 각 사례의 Baseline → Ours 지표 비교와 sample ID. 관측 1x · cutoff 이후 0.5x 재생. 원본 MP4·포스터는 외부 저장소에 있으며 본 Repository에는 catalog·평가·예측 metadata만 포함한다.

| Demo | sample_id | 지표 | Baseline | Ours |
|---:|---|---|---:|---:|
| 01 | `predcls:KOQGE.mp4_0.5` | NC R@20 (%) | 30.9359 | 95.0399 |
| 02 | `predcls:V1BX9.mp4_0.5` | NC R@20 (%) | 13.5317 | 75.4082 |
| 03 | `predcls:YDWIW.mp4_0.5` | NC R@20 (%) | 24.9359 | 86.6528 |
| 04 | `predcls:WC5QK.mp4_0.5` | NC R@20 (%) | 35.4509 | 92.3009 |
| 05 | `predcls:0UPBN.mp4_0.5` | NC R@20 (%) | 41.4966 | 97.6190 |

카탈로그 기록(`catalog.json`)에는 위 정보 외에 영상·포스터·예측 경로, 범주, metric_delta 가 함께 들어 있다.