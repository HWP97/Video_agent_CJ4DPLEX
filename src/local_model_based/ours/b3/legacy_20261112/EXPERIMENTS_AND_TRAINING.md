# b3 기존 실험 적용 범위와 학습 출처

이번 선택은 기존 공개 validation 결과를 본 **사후 선택**이다. 새 학습·API·다운로드는 없다. 비선택 후보는 아래 원 실험의 프로토콜·manifest·결과표를 읽었으며 모두 새로 실행했다는 뜻이 아니다. 선정한 API/로컬 ours만 실제 실행과 공식 재채점을 다시 했다.

|방법/원리|API 계열의 기존 적용|완전 로컬의 기존 적용|확인 근거와 제한|
|---|---|---|---|
|raw Sol|저장 응답1939개|비교 기준으로만 읽었음|전체요청1943 중누락4|
|hold/CV/KF, selector, damping|Sol과 비교, 후처리anchor|비학습 좌표 알고리즘|C의TrajNet7방법 및 MOT17별도 pixel표|
|zero-initialized residual|Sol conditioned Ridge, fit219|자기이력·social·collision MLP|baseline_residual; API fit219는 독립평가가 아님|
|shared recurrent attention|Sol reference만|single/shared2/shared4/dynamic/unshared|llm_transfer의37출력; baseline은direct_unshared4_d4|
|kNN memory, TTT 원리|로컬 잔차를 Sol로 transport|direct/KF/local memory, prefix update, static ridge|expansion의60방법 묶음; 실제LLM재현이 아님|
|FAST, TRM|로컬 잔차 transport|DCT/rawBPE/continuous, two-state/deep supervision|additional_candidates; pretrained로봇정책/LLM이 아님|
|LLaDA masking, DeepSeek MTP|Sol correction transport|masked/parallel, AR/auxiliary MTP|metric_local_followup의 축소 구현|
|Chronos-2|없음|pretrained joint/independent zero-shot|target-only, 이웃없음. local ADE0.735533/0.686207|
|Resonance, SocialMOIF|Sol 비교만|공식8관측 Re 및 자체9관측 축소코어|paper_experiments. 공식Re mean20은단일평균, minADE20 oracle과구별|
|Sol/CV 비학습 후처리|0.75Sol+0.25CV를ours로선정|Sol이필요하므로완전로컬아님|sol_postprocess, 충돌악화공개|
|local_static_r1.0|이번API ours에사용하지않음|기존attention+train-only19특징Ridge,ours선정|기존동결계수만사용; 신규fit0|

같은 입력 조건 안에서 비교했다. Sol은 모든 관측 이웃을 텍스트로 받고 attention은 최대8이웃 요약을 받으므로 API 대 로컬 수치를 동일 모델의 공정한 입력 ablation처럼 합치지 않는다. 본 로컬 비교는 같은 attention anchor·관측·fold를 유지한다. Re8관측과9관측, MOT17 pixel과TrajNet nominal meter, best-of20과단일예측은 별도다. Re oracle minADE20=0.291021은 GT로후보를고르는 지표이므로 ours선정에서 제외했다.

## 흩어진 자료 확인 범위

- `research_runs/social_forecast_v1`: run summary·manifest·training_data 파서 확인. 초기 social MLP는최종attention과다르다.
- `paper_experiments_20261109/b3`: summary·oracle_summary·조건·sourcemanifest. 원논문 전체재현과축소코어를구별했다.
- `baseline_residual_20261110/b3`: headline·모델·train/Sol분리. Sol Ridge전체0.578161은fit219포함, holdout1720 ADE0.602557이다.
- `llm_transfer_20261110/b3`, `llm_transfer_expansion_20261110/b3`, `additional_candidates/b3_tokens_recursive`: summary·동결설정·기존코드/가중치 확인.
- `metric_local_followup_20261110/b3`: prior_mapping·INVENTORY·method_artifacts·통합결과, C카탈로그, fit/holdout 구분 확인.
- `sol_postprocess_b3_20261110`: 전체후보·개발219선택·holdout1720·결과. 학습보정기추가없음.
- `C`: 기존selector 결과와TrajNet7방법카탈로그 확인. 카탈로그 충돌열은과거값이므로새재채점이라고하지않는다.
- `my_works/b3`: legacy_execution_summary·카메라/객체보고서·MOT17 A/B표 확인. 초기9영상은미래GT없어ADE/FDE미측정, MOT17은pixel/대각선%라현재표에합산하지않는다.
- `previous_exp`: 아카이브README의A/B/BCD/D_legacy 구분확인. 중복QA/재생폴더를새모델로세지않았다. 모든과거영상파일을재실행하지않았다.
- `fixed_baseline_bottleneck_20261110`: b1/b2만존재하며b3하위실험없음.

원표의2,408행과 C/fit-holdout추가표는 외부지원폴더 `reports/{family}/b3/candidate_inventory.json`에 출처SHA와함께보존했다. 행수는실험수와같지않으며 cohort·중복baseline을포함한다. 과거동영상/서버, 모든후보의재학습·재추론, 공식hiddenchallenge동일성은미검증이다.

## 사용자가 train set을 주지 않았는데 학습한 근거

현재포장은학습하지않았다. **과거실험은이미확보된공개ETH/UCY좌표를학습했다.** 사용자전용train을받았거나 Sol이학습정답을생성한것이아니다. 실제출처는 `my_works/experiments/benchmarks/2026-11-06/001_frozen_full/b3/data/author_conversion/output/train/`의 hotel/students001/students003/zara01/zara03 다섯NDJSON이다.

저자변환기 `vita-epfl/trajnetplusplusdataset`의고정commit은 `75e697d6b2f55c0ade2b2b5f7bbf53a5f34b2d50`이다. `sources/trajnetplusplusdataset/source.json`과 `evidence/trajnet_conversion.log`가원변환출처다. 변환기기본분할은시간순train0.6/val0.2이고추가분류를거친출력을사용했다. **공식배포hidden v4 challenge train과동일하다고주장하지않는다.** 이번에다운로드를다시검증하거나실행하지않았다.

`research_runs/social_forecast_v1/data.py:training_data`는각scene의21좌표를읽어앞9개를관측,뒤12개를train정답으로분리한다. `llm_transfer_20261110/b3/prepare.py`가 이를토큰화하고 `cache/train.npz`, `train_labels.npz`, `train_index.json`을만든다. 지금다섯원파일SHA·scene합7774, labels[7774,12,2]의유한함, validation1943, dataset별train/validation프레임교집합0을직접검사했다. 증거는지원폴더 `training_data_audit.json`이다.

|target|학습그룹/수|개발그룹/수|평가수|선택attention기존학습초|
|---|---|---|---:|---:|
|hotel|students/6512|zara/1099|19|31.670|
|students|zara/1099|hotel/163|1591|5.228|
|zara|students/6512|hotel/163|333|19.983|

attention은 `llm_transfer_20261110/b3/train.py`의전체소형망학습이며 기존 `training_receipt.json`의36fit/477.259초는다른구조실험까지포함한다. 선택3체크포인트는 `models/*_direct_unshared4.{pt,json}`의SHA와train_seconds로연결된다. 과거명령진입점은기존Python환경의 `python -B train.py`이며재실행하지않았다.

static residual은 `llm_transfer_expansion_20261110/b3/experiment.py develop`에서 같은fold train 관측19특징과 **train GT−train attention예측** 잔차로 구한폐형식Ridge다. `models/{hotel,students,zara}.npz['local_static_1.0']`가현재재사용하는19×24계수다. `development_receipt.json`의215.190초는메모리·다른ridge후보·개발채점까지포함하며이계수단독학습시간이아니다. 새묶음에는이계수만추출했고train라벨/메모리는추가복사하지않았다.

attention과static 학습에는Sol응답/교사가없다. API의Sol-conditioned Ridge는별도과거실험이며현재API ours는학습없는고정결합이다. target원본그룹train/dev제외와프레임분리는있지만, **이번최고방법선택은공개validation을본뒤의사후선정**이므로검증셋으로방법을골랐다는한계가남는다.

