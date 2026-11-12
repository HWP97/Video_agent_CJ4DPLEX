# Scene Graph Anticipation (Local) — Demo Catalog

최종 선정 **5개** 사례. 각 사례의 Baseline → Ours 지표 비교와 sample ID. 관측 1x · cutoff 이후 0.5x 재생. 원본 MP4·포스터는 외부 저장소에 있으며 본 Repository에는 catalog·평가·예측 metadata만 포함한다.

| Demo | sample_id | 지표 | Baseline | Ours |
|---:|---|---|---:|---:|
| 01 | `predcls:GF7DV.mp4_0.5` | NC R@20 (%) | 46.1538 | 84.6154 |
| 02 | `predcls:6PESI.mp4_0.5` | NC R@20 (%) | 60.0000 | 95.0000 |
| 03 | `predcls:DLFPX.mp4_0.5` | NC R@20 (%) | 65.4762 | 100.0000 |
| 04 | `predcls:4ALLQ.mp4_0.5` | NC R@20 (%) | 58.6806 | 91.6667 |
| 05 | `predcls:NDDHA.mp4_0.5` | NC R@20 (%) | 47.4603 | 80.2431 |

카탈로그 기록(`catalog.json`)에는 위 정보 외에 영상·포스터·예측 경로, 범주, metric_delta 가 함께 들어 있다.