# API Cache (최종 Demo 15개 전용)

세 head의 확정 Demo 15개에 대한 **원본 Sol API 요청·응답**을 signature(SHA256) 기반 폴더 구조로 모았다. 전체 API 캐시의 서브셋이며, 평가자가 재현에 필요한 최소 세트다.

```
api_cache/
├── index.json                                           # 15개 entry (task, rank, sample_id, signature, files)
└── <task>/<signature>/
    ├── input.json      # Sol에 넣은 입력(프레임 경로, 프롬프트, 스키마, provenance)
    ├── request.json    # 실제 HTTP 요청 payload (model, messages, response_format 등)
    └── response.json   # Sol 응답 raw (parsed JSON + 메타)
```

## index.json

| 필드 | 설명 |
|---|---|
| `task` | head 식별자 (`scene_graph_anticipation` · `affect_forecasting` · `trajectory_forecasting`) |
| `rank` | Demo catalog에서의 번호 (API 최종 선정 번호) |
| `sample_id` | catalog의 사례 ID |
| `signature` | 요청 payload SHA256, 캐시 폴더명과 동일 |
| `files` | 캐시 폴더 내 3개 파일의 상대 경로 |

## 사용 방법

```bash
# 캐시 유효성·매칭 요약 보기
python tools/api_demo/replay_cache.py

# 특정 사례만
python tools/api_demo/replay_cache.py --task affect_forecasting --rank 2
```

`replay_cache.py` 는 각 entry 의 `response.json` 에서 모델 출력을 꺼내 해당 Demo catalog(예: `demo/api_based/affect_forecasting/catalog.json`) 의 `baseline_metric` 와 일치하는지 확인한다. 재학습·새 API 호출 없이 "저장된 Sol 응답 → 저장된 Demo 지표" 흐름을 재검증하는 용도다.

## 제한

- 15개 선정 사례만 포함 — 전체 A/B 평가(예: 공통 1,939 trajectory)는 커버하지 않는다. 전체 캐시는 외부 저장소 (`HWP_CJ4DPLEX_AI_support/.../api_cache_full`)에 있다.
- `input.json` 의 프레임 경로는 원 작업 서버 절대 경로(`/mnt/RG2/my_works/...`)다. 참고용이며 실행에 필요하지 않다 — Sol 응답은 이미 `response.json`에 캐시돼 있다.
- API Key 등 비밀값은 포함되지 않는다.
