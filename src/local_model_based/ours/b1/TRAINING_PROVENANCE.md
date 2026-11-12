# 로컬 관계 예측 학습 자료와 실행 근거

## 최신 배포와 이전 이력의 구분

현재 배포는 [README](README.md)의 최신 메모리 gate·제한 잔차 head다. 최신 출처는 `data/b1/latest/provenance.json`, 모델 선택은 `combined_selection_lock.json`, 학습 receipt는 `sgcls_c04_gate_positive_cap3.json`과 `predcls_c06_gate_positive_cap15.json`에 있다. 원 실험 폴더는 `HWP_CJ4DPLEX_AI_support/iterative_research_20261111/b1/combinations_81112`이며 이번 배포 교체에서 추가 학습하지 않았다.

**아래는 이전 배포의 학습 이력이다.** 아래의 ImparTail 유지·adopted=false는 과거48,404파라미터 head에 대한 기록이며 최신 조합 head의 선택 결과를 뜻하지 않는다. 변경 전 문서는 `legacy_20261112/TRAINING_PROVENANCE.md`에 그대로 보존했다.

최신 선택 모델의 학습·입력·선택 소스는 `data/b1/latest/source/`에 원문 그대로 보존했고 `training_sources.json`으로 원본·복사본 SHA를 연결했다. 상위 관계 head와 그 import가 요구하는 감정 모델 구성 코드도 포함했다. 이 snapshot의 재학습 CLI는 실행하지 않았다. 원 경로·원 학습 입력이 필요한 연구 기록이며, 검증된 배포 경로는 README의 동결 head 추론이다. 학습 데이터 준비부터 새 장비에서 재학습까지 확인했다고 주장하지 않는다.

## 이전 배포 기록

사용자가 이번에 새 train 파일을 준 것이 아니다. 과거 실험 코드가 기존 공개 Action Genome 주석과 Charades 영상을 사용하여 공식 train에서 500개 영상 ID를 고른 기록이다. 이번 ours 변경에서는 학습·다운로드를 하지 않았다.

## 공개 pretrained와 자체 head의 구분

ImparTail ODE는 저자가 공개한 epoch3 checkpoint를 재사용했다. PGAGS cap40/GAGS cap70 선정은 기존 개발 비교 결과이며 우리가 ImparTail backbone을 학습한 것이 아니다. 원 저자 학습의 정확한 표본 수·학습 시간·모든 hyperparameter는 로컬 학습 receipt가 없어 미확인이다. 다운로드 provenance는 `paper_experiments_20261109/b1/manifest.json`, 선정은 `b1/protocol/impartail_selected.json`에 있다.

우리가 실제 학습한 것은 별도 48,404 parameter RelationDynamics head다. `gt_transfer`, `predicted_direct`, `unanchored_restore_aux`, `identity_anchor_aux`, `identity_anchor_no_aux`를 8epoch 학습했고 원 `selection.json`의 두 mode `adopted=false`를 유지한다. head 훈련 실패는 미실행을 뜻하지 않는다. 훈련은 끝났지만 개선 채택 기준을 통과하지 못했다.

## 원본 데이터 경로

- 공식 주석: `/mnt/RG2/my_works/b1/benchmarks/action_genome_20261106/data/annotations/person_bbox.pkl`, `object_bbox_and_relationship.pkl`, `object_bbox_and_relationship_filtersmall.pkl`.
- train 선정 500영상: `/mnt/RG2/paper_experiments_20261109/b1/protocol/fdsg_train.json`의 `official_split=train`, `videos`.
- train/dev 분할: `/mnt/RG2/paper_experiments_20261109/combinations/protocol/split.json`: training400, development100, pilot10(training의 부분집합).
- 실제 eligibility: 같은 `protocol/train_manifest.json`: PGAGS training392 ready+8 ineligible/development100 ready, GAGS training400/development100 ready.
- 관측 영상·프레임: `/mnt/RG2/paper_experiments_20261109/combinations/assets/visual/data/videos`, `frames`.
- 감독 입력: 같은 `assets/supervision/{mode}_{video}.json`의 observation/future_GT. 관측 compact는 assets/compact_train 및 compact_dev.

`prepare_train.py`는 `AG('train', 'full', ...)`을 호출한다. 원 AG loader는 `metadata.set == self._phase`로 train/test를 가른다. `supervision.py::targets`는 공식 관측·미래 관계 GT를 감독으로 사용하며 추론 adapter는 이 모듈을 import하지 않는다. Sol teacher, Sol 응답 또는 Sol feature를 입력하는 학습이 아니다. 반면 API macro_direct는 이미 열람한 공개 test 내부 development를 다시 fit263/selection88로 나눈 별도 Sol 후처리이므로 아래 로컬 train400/dev100과 혼동하면 안 된다.

## 실제 명령과 receipt

작업 디렉터리 `/mnt/RG2/paper_experiments_20261109/combinations`에서 과거 `run_when_ready.sh`가 다음 명령을 실행한 기록이다. 이번에는 재실행하지 않았다.

```bash
/mnt/RG2/my_works/b1/benchmarks/action_genome_20261106/model/env/bin/python train_and_select.py > evidence/train_and_select.log 2>&1
```

| 단계·모드 | training groups / examples | 선택 진단 head의 8epoch 시간 | 실제 receipt |
|---|---:|---:|---|
| 개발용 PGAGS fit | 392 / 9,502 | identity_anchor_no_aux 4.081736초 | results/development_sgcls_training.json |
| 개발용 GAGS fit | 400 / 11,080 | identity_anchor_aux 4.807243초 | results/development_predcls_training.json |
| 선택 뒤 진단 refit PGAGS | 492 / 11,842 | identity_anchor_no_aux 5.081987초 | results/final_sgcls_training.json |
| 선택 뒤 진단 refit GAGS | 500 / 13,704 | identity_anchor_aux 5.887682초 | results/final_predcls_training.json |

시간은 캐시가 준비된 소형 head 학습 루프이며 영상 준비·encoder feature·채점을 포함하지 않는다. 각 receipt에는 다섯 변형의 시간·loss·parameter·epoch가 따로 있다. 개발 100그룹은 학습 전체가 아니라 선택용 분할이다. refit은 train+dev를 합친 것이며 별도 독립 성능 검증이 아니다. 공식 train 개발에서 baseline R비감소와 mR증가를 동시에 충족하지 못해 ours는 pretrained ImparTail을 유지한다.

그 밖의 실제 로컬 학습은 `paper_experiments_20261109/b1`의 symbolic FDSG 축소 실험이다. `results/fdsg_training.json`에 공식 train에서 만든 29,244 예시·8epoch, SDE14.262526초/drift-only10.416653초/no-box-residual12.627046초 CPU 학습 기록이 있다. GT symbolic state 입력이므로 시각 DETR backbone 재현이나 현재 관측 이미지 benchmark의 직접 대체 성능으로 주장하지 않는다. OOTSM 축소는 Qwen4 로컬 추론이며 새 fine-tuning 기록이 아니다.
