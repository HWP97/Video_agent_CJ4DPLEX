# b1 ours: 기존 macro_direct

기존 공개 성적을 확인한 뒤 **NC·관측 0.5 성능으로 사후 선택**한 동결 `macro_direct`다. **과거 GAGS WC 개발 채택 기준에서는 accepted=false였고 WC의 R@20은 낮아진다.** 새로운 독립 평가나 과거 채택 성공으로 소급하지 않는다. WC/NC에 따라 방법을 바꾸는 router 없이 전 조건에 같은 방법을 적용한다. baseline은 raw Sol 그대로다.

macro_direct는 저장된 Sol 순위와 관측 특징을 받아 94차원 입력의 선형 head로 26개 관계 확률을 계산한다. 잔차가 없는 기존 Adaptive 모델이며 저장된 두 mode checkpoint를 strict load한다. 새 API 호출·학습·튜닝은 없다.

## 같은 집합의 기존 수치

이미 열람한 전체 공식 test, 관측 0.5, R@20/mR@20(%). PGAGS 1,737사례, GAGS 1,750사례다.

| 모드·제약 | raw Sol baseline | macro_direct ours |
|---|---:|---:|
| GAGS NC | 47.3739 / 52.8927 | **61.7654 / 63.4819** |
| PGAGS NC | 40.6693 / 46.5423 | **52.2368 / 51.3102** |
| GAGS WC | 43.8103 / 32.1818 | 42.1704 / 35.3517 |
| PGAGS WC | 35.2692 / 26.3686 | 33.7820 / 29.6046 |

baseline_residual, llm_transfer, llm_transfer_expansion, masked 확장, fixed_baseline_bottleneck, metric_local_followup의 같은 조건 CSV를 비교했다. macro_direct는 이 후보들의 GAGS NC mR@20 중 최고이며 양 mode에서 baseline R/mR을 함께 높인다. 다만 PGAGS NC mR만 보면 log_rank_residual 51.5928이 조금 더 높다. 모든 지표에서 최고라는 주장은 하지 않는다. 개발·holdout·전체를 섞지 않은 비교 원자료는 `data/b1/selection_audit`에 있다.

기존 WC 주조건을 통과했던 `rank_residual`은 `--research rank_residual`로 보존했다. 기본 ours 선정은 사용자의 기존 성능 중심 요청에 따른 변경이며, 과거 WC 기준의 통과 여부는 receipt에도 적는다.

## 실행

계열 루트에서 실행한다. `final`은 `ours`의 호환 별칭이다.

```bash
python3 main.py b1 ours status
python3 main.py b1 ours verify
python3 main.py b1 ours evaluate --sample predcls:00607.mp4_0.5
python3 main.py b1 ours replay
python3 main.py b1 ours predict --sample sgcls:00607.mp4_0.5
python3 main.py b1 ours evaluate --research rank_residual --limit 1
```

evaluate는 원 공식 평가기로 새로 채점하고 기존 동결 per-case 통계와 정확히 대조한다. limit/sample 생략 시 전체 13,635유효 사례를 재채점한다. replay는 이미 저장된 전체 통계를 합산하며 새 채점과 구별한다. 빈 미래 313개는 따로 보고한다. verify는 이번에 추가한 모델·비교 자료 17개를 SHA/inode로 검증한다. 기존 전체 데이터 검증은 baseline verify에 있다.

predict는 GT 없는 순수 entry/features를 읽으며 prepared/미래 GT 접근을 차단한다. 기본 순수 입력은 양 mode의 00607.mp4 관측0.5이고 다른 입력은 `--input`으로 지정한다. 결과는 최상위 `results/api_based/b1`에 생성한다. 관측 GT bbox/class, 미래 annotation schedule, IoU=0 조건과 공개 test 재사용 한계는 유지한다. 환경·대용량 리소스 준비는 계열 루트 안내를 따른다.
