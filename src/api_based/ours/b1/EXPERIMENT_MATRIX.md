# b1 기존 실험 계열 감사

2026-11-10 재확인. 같은 저장 출력의 재채점·retry·이름 변경은 새로운 학습 방법으로 중복 계산하지 않았다. API 계열은 Sol/Gemini 출력이 추론 입력에 들어가는 방법이며, 후처리가 CPU에서 돌아가도 완전 로컬로 분류하지 않는다. 아래는 실제 파일이 확인된 범위다.

| 원본 위치·실험군 | API 적용 | 완전 로컬 적용 | 비교·선택 판단 |
|---|---|---|---|
| my_works/experiments/pilot 및 expanded, b1/runs | Gemini direct | Qwen4/8 direct, structured, last-frame | 소수 일반영상 자유서술 품질. Action Genome 관계 지표와 직접 비교하지 않음 |
| my_works/b1/results/2026-11-05, experiments/benchmarks/2026-11-05·11-06의 EPIC | Sol | V-JEPA 공개 probe, Qwen, 빈도·전이 기준 | EPIC 행동 anticipation. train 빈도/transition 및 GT-history oracle 구분. 수집 중/중단 범위가 있어 AG 최고 방법 선택에 합치지 않음 |
| my_works/b1/results/2026-11-07·11-08, action_genome_20261106 | raw Sol | Qwen4, SceneSayer ODE, 동일 encoder hold | 최종 13,635유효/313빈미래. retry는 같은 방법의 실패 복구이며 새 방법 아님 |
| previous_exp/A/001_ag_relation_postprocess | A1 Sol 점수 보정, A2 Sol+로컬 bank 결합 | 순수 로컬 새 후처리 실험 아님; ODE/hold는 비교군 | A1 선택 raw 유지, A2 비채택. 공통 객체와 전체 후보 분모를 구분 |
| previous_exp/A/002_ag_object_recognition | 로컬 관측 class+고정 Sol 관계 hybrid | 순수 로컬 새 모델 아님 | class 정확도·관계 R/mR 하락으로 비채택. GT class 치환은 oracle |
| paper_experiments_20261109/b1: ImparTail | 없음 | 공개 pretrained cap10/40/70 비교·40/70 선택 | 저자 공개 backbone 재사용. 우리가 backbone 학습한 결과가 아님 |
| 같은 b1: OOTSM 축소 | 없음 | Qwen4 관측 그래프 텍스트 GOA/OORA, 4변형 | 공식 fine-tuned OOTSM 가중치 재현 아님; 큰 pretrained를 로컬 추론한 축소 실험 |
| 같은 b1: FDSG 축소 | 없음 | symbolic TDM/SDE, drift-only, box-residual 제거 | 공식 train GT를 사용한 축소 oracle 조건. DETR 시각 backbone 재현이 아니므로 실제 관측 영상 방법과 단순 순위 비교 금지 |
| paper_experiments_20261109/combinations | 없음 | gt_transfer, predicted_direct, restore_aux, identity_anchor 두 변형; encoder_hold/current-hold ablation | 실제 공식 train 지도학습. 두 mode 개발 채택 실패, ImparTail 유지 |
| paper_experiments_20261109/full_benchmark | raw Sol 비교군 | ImparTail 및 위 head 진단의 전수 적용·같은 bank 비교 | combinations 학습 결과의 재평가로 중복 학습 실험 아님. test 증가로 개발 실패를 뒤집지 않음 |
| baseline_residual_20261110/b1 | Sol 조건부 local_linear/local_mlp | **미적용** | 이름 local은 실행 위치/관측 특징을 뜻함. Sol 의존 API 후처리. local_mlp의 작은 NC 개선 기록 존재 |
| llm_transfer_20261110/b1 | direct/anchor/linear, temporal-order vs bag, wave2 | **미적용** | Sol feature를 쓰는 학습 실험. 완전 로컬 teacher-free 변형으로 실행했다고 주장하지 않음 |
| llm_transfer_expansion_20261110/b1 | category/memory, MTP/bag, memory_linear | **미적용** | memory_linear 개발 통과이나 GAGS holdout NC mR은 raw보다 하락. 전체와 holdout 구분 |
| llm_transfer_expansion_20261110/additional_candidates/b1_masked | mask 종류×direct/linear/memory/Sol, 1/3step denoise | **미적용** | 실제 전체 성적 CSV 있음. 최고 GAGS NC mR이 macro_direct보다 낮음 |
| fixed_baseline_bottleneck_20261110/b1 | family head, Sol/head 결합, order/diversity 진단 | **미적용** | 다른 폴더 결과를 합친 행은 원 실험과 중복 제거. 원래 성적 파일은 summary_full.csv |
| metric_local_followup_20261110/b1 | macro_direct/residual, rare_guard, rank_residual, log_rank_residual | **미적용** | 기본 ours=macro_direct 사후 NC 선택. 옛 WC 개발 gate는 rank_residual만 통과; macro_direct WC R 하락 공개 |
| research_runs/branch_comparison_20261109 | A 재채점 부분 산출물 | b1 새 방법 없음 | SCOPE_CORRECTION.md에 중단 기록. 완료 성적으로 중복 계수하지 않음 |
| research_plans/2026-11-09_* 및 research_runs/NEW_METHODS.md | 사건×시점 head 계획 | frozen V-JEPA 과제 head 계획 | b1 실행 조건 미충족/미실행. b3 social/motion의 실제 head 학습을 b1 실행으로 옮기지 않음 |
| api_local_feasibility_20261110 | 기존 조사 | 기존 조사 | 새 추론·학습 성적이 아닌 계획/감사 |

루트 디렉터리 목록을 확인했으며 `baseline_local_followup_20261110`이라는 폴더는 현재 `/mnt/RG2`에 없다. `metric_local_followup_20261110`을 이름만 보고 완전 로컬 실험으로 오인하면 안 된다. C와 sol_postprocess_b3는 b3, VACE/Wan/cosmos/huggingface는 여기서 b1 학습 결과를 입증하는 성적 산출물이 확인되지 않은 모델·참고 자산이다.

최고 방법 판단에 사용한 실제 CSV 6개 계열의 관측0.5 전체/holdout 752행은 `HWP_CJ4DPLEX_AI/src/api_based/data/b1/selection_audit/comparison_50.json`에 보존했다. `ours_assets.json`에는 17개 추가 파일의 원본/복사본 SHA·bytes·inode가 있다. 주 비교는 GAGS NC 전체 1,750사례의 mR과 양 mode R/mR 비하락이며, API ours의 선택이 과거 WC 기준을 바꾼 사후 결정임을 명시했다.

기록을 확인하지 못한 것을 실행 성공으로 표시하지 않았다. 초기 일반영상·EPIC·상징 GT oracle 결과는 Action Genome 관측 이미지 benchmark와 입력·라벨·분모가 달라 직접 최고점 경쟁에 섞지 않는다.
