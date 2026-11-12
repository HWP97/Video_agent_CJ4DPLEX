# b3 로컬 ours: attention + 기존 static residual

`ours`는 기존 `local_static_r1.0`다. 고정baseline `direct_unshared4_d4` attention에 과거train-only Ridge 계수를적용한다. 새학습·API·다운로드없이실제forward를실행한다. 같은입력·fold·전체1,943장면의 **로컬끼리 비교**다.

|방법 / 전체1,943|ADE↓|FDE↓|Col-II%↓|제공/누락|
|---|---:|---:|---:|---:|
|baseline attention|0.622295|1.292326|22.490993|1943/0|
|ours local_static_r1.0|0.615758|1.287116|22.079259|1943/0|
|기존static_r0.1 참고|0.616395|1.289657|21.358724|1943/0|
|기존shared4_d4 참고|0.621401|1.295052|22.902728|1943/0|
|기존Chronos independent 참고|0.686207|1.449964|22.490993|1943/0|

ours는기존동일조건로컬후보중ADE최저이고baseline보다FDE·충돌도낮다. r0.1은충돌이더낮지만ADE가높다는tradeoff가있다. **공개validation성적을본사후선정**이며새hidden검증은아니다. 공통1,939에도같은평가기를적용한다. 참고후보는기존기록이며이번에모두재추론하지않았다.

입력은9관측·12미래·0.4초·nominal meter, attention은최대8이웃요약을쓴다. ours의추가계수는정규화된관측위치18개+상수1개를입력으로12×2잔차를낸다. fold별기존19×24계수만복사했다. 미래GT·Sol이추론입력에들어가지않는다. 새영상pixel→meter서비스연결은별도다.

계열루트 `HWP_CJ4DPLEX_AI/src/local_model_based`에서:
```bash
python3 main.py b3 ours predict --device cpu --output results/b3/my_ours
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 main.py b3 ours predict --device cuda --output results/b3/my_gpu
python3 main.py b3 ours evaluate --prediction results/b3/my_gpu/predictions.npz --output results/b3/my_score
python3 main.py b3 ours verify --output results/b3/my_verify
```

results/는 `HWP_CJ4DPLEX_AI/results/local_model_based`로매핑된다. 이번 `b3/ours_gpu` 실제예측은기존동일방법과maxdiff0이고공식score가동일하다. CPU결과도유한하지만GPU와미세한부동소수점차이가있다. ours의replay도같은동결계산을재실행한다. baseline코드는별도고정이며알고리즘을바꾸지않았다.

[전체방법/API·로컬적용표와학습출처](EXPERIMENTS_AND_TRAINING.md)에train이사용자제공셋이아닌공개저자변환데이터임과fold/receipt/시간을기록했다. 원실험은 `llm_transfer_expansion_20261110/b3`, 계수·예측의출처SHA는 `data/b3/ours/manifest.json`이다. 원천데이터·공식평가기이용조건은유지한다. 대용량데이터는Git추적대상이아니며실행에는로컬복사자산과기존환경이필요하다.

