# 로컬 리소스와 제출 전 준비

이미 확보한 리소스를 별도 파일로 복사했다. 이번에는 모델 다운로드·학습·API 수집을 실행하지 않았다. 기존 복사 manifest는 작업 루트의 `HWP_CJ4DPLEX_AI_support/reports/<계열>/`에 있고 최신 자산의 SHA는 아래 manifest를 따른다.

| 리소스 | 출처 | 현재 위치와 준비 상태 |
|---|---|---|
| 관계 예측 ImparTail PGAGS/GAGS | [저자 코드](https://github.com/rohithpeddi/ImparTail), [저자 checkpoint 배포](https://utdallas.box.com/s/exbgy57fjnkt5aahoz9z07njtw1y2z9o) | `src/local_model_based/data/b1/weights/`; 선정된 `ode_partial_40_sgcls_future_3_epoch_3.tar`, `ode_partial_70_predcls_future_3_epoch_3.tar` 복사 |
| 관계 예측 Faster R-CNN | [SceneSayer 저자 코드의 준비 안내](https://github.com/rohithpeddi/SceneSayer) | `src/local_model_based/baseline/b1/vendor/benchmark/model/SceneSayer/fasterRCNN/models/faster_rcnn_ag.pth` 복사 |
| 감정 상태 VEATIC 인식 | [저자 코드](https://github.com/AlbusPeter/VEATIC) | `src/local_model_based/data/b2/weights/one_stream_vit.pth` 복사 |
| 이전 보행 소형 attention | 기존 프로젝트의 자체 학습 결과 | `src/local_model_based/data/b3/models/*_direct_unshared4.pt` 3개 fold 보존; 최신 경로는 아래 별도 표 참조 |
| API 원응답·관측 입력 | 기존 Sol 고정 요청과 응답 | `src/api_based/data/bN/`; 실제 입력 signature가 일치하는 재생에만 사용 |
| 관측 데이터·GT | Action Genome / VEATIC / TrajNet++ 기존 확보분 | `src/<계열>/data/bN/`; 관측 입력과 평가 GT를 구분 |

## 최신 로컬 교체 자산

| 과제 | 동결 모델·입력 위치 | 무결성 근거 |
|---|---|---|
| 관계 예측 | `src/local_model_based/data/b1/latest/`의 SGCLS c04·PredCls c06 checkpoint와 각 memory NPZ. 기존 ImparTail 관측 graph/prior도 필요 | 같은 폴더 `provenance.json`, `training_sources.json` |
| 감정 상태 예측 | `src/local_model_based/data/b2/latest/model/c01/`의 GRU·PCA32, `model/`의 D02·D04 계수와 transform, `inputs.npz`·`manifest.json` | 같은 latest의 `assets.json`·`assets_c01.json`·학습/추론 receipt |
| 보행 궤적 예측 | `src/local_model_based/data/b3/latest/models/`, `inputs/`, `indices/`, 평가용 `ground_truth/` | 같은 latest의 `manifest.json`·`runplan.json`·`selection/` |

최신 세 자산 묶음의 현재 로컬 크기는 대략3MB·327MB·1.2GB다. 파일별 manifest의 정확한 이름·SHA를 기준으로 준비하며 큰 데이터·가중치를 Repository에 넣지 않는다. 자체 모델과 입력 묶음의 평가자 다운로드 주소는 아직 미확정이다. 공개 원자료 링크만 제공하는 것으로 자체 head와 관측 특징 준비를 대신하지 않는다.

학습 소스는 대용량 data ignore에 가려지지 않도록 `src/local_model_based/training/latest_relations_20261112/`, `training/latest_affect_20261112/`와 `ours/b3/training_source/`에도 원문 snapshot을 보존한다. 학습·선택 코드를 제공하되 해당 snapshot만으로 새 장비 재학습을 검증했다고 주장하지 않는다.

외부 가중치를 새 장비에 준비할 때는 저자 배포에서 동일 파일을 받아 위 위치에 넣고 복사 manifest의 SHA256과 대조해야 한다. 공개 코드 라이선스와 영상·가중치의 이용 조건은 동일하다고 간주하지 않는다. 자세한 기존 확인 내용은 branch별 reports에 보존했다. 여기의 링크는 기존 확보 출처이며 이번 작업에서 재다운로드 성공을 의미하지 않는다.

ImparTail의 보존된 README와 소스 목록에서는 명시적 LICENSE를 확인하지 못했다. 이것만으로 원 저자의 모든 허가 조건을 단정하지 않으며, 코드·가중치의 제출 및 재배포 범위는 별도 확인 항목이다. 하위 SceneSayer의 MIT를 ImparTail 전체와 가중치에 자동 적용하지 않는다.

과제원문 2.4에 따라 가중치를 Repository에 직접 포함하지 않는다. 현재 로컬 리소스는 `.gitignore`로 제외되어 있다. 자체 학습 가중치, 평가자의 데이터 접근 및 원영상 이용 조건을 확인한 배포 범위와 외부 저장 위치를 확정하기 전에는 이 전체 폴더를 공개 업로드하지 않는다. 해당 배포 준비는 이번 디렉터리 정리 완료와 별개다.

## 데모 영상 원본 외부 보관

GitHub 용량과 원영상 라이선스 때문에 Demo 원본 MP4·포스터 JPG, Cosmos 보조 시각화 전체 자산은 Repository에 포함하지 않았다. 원본·Cosmos 자산은 작업 서버의 `HWP_CJ4DPLEX_AI_support/submission_cleanup_20261112/demo_media/` 및 `cosmos_full/`에 원 상대 경로를 보존해 이동했다. Repository에는 각 head의 `catalog.json`·`CATALOG.md`·`cases.json`·평가/예측 JSON 등 metadata만 유지한다. 평가자용 외부 다운로드 URL은 자체 학습 가중치와 함께 확정 후 README·본 문서에 반영한다. README의 Demo 미리보기는 저용량 GIF로 제공하므로 원본 MP4 없이도 사례 확인이 가능하다.

검증은 현재 장비에서 기존 Python 환경을 재사용했다. vendor의 일부 기존 확장 바이너리는 Linux/Python ABI에 의존할 수 있어 새 장비에서는 해당 확장 빌드 또는 호환 경로 확인이 필요하다. 깨끗한 환경 재설치는 아직 검증하지 않았다.
