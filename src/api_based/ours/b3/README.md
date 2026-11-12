# b3 API ours: 기존 고정 Sol/CV 결합

`ours`는 기존 `cv_blend_0.75`, 즉 **0.75×Sol+0.25×CV**다. baseline raw Sol은 그대로다. 기존 허용 비학습 후보 중 공개 validation의 ADE가 가장 낮은 방법을 사후 선택했다. 새 학습이나 API 요청은 없다.

|공통1,939 / 전체요청1,943|ADE↓|FDE↓|Col-II%↓|제공/누락|
|---|---:|---:|---:|---:|
|baseline raw Sol|0.588297|1.301639|18.875709|1939/4|
|ours cv_blend_0.75|0.571526|1.247040|19.133574|1939/4|

ADE−2.85%, FDE−4.19%지만 충돌률은 **+0.257865%p 악화**한다. 과거엄격한비악화gate에서는미채택됐으나, 이번요청의최고기존거리성적선정으로ours에연결했다. 모든지표우위나새hidden평가성공은아니다. 개발219로이미고정된비율을다시탐색하지않는다. 같은holdout1720에서도ADE0.611296→0.593962, FDE1.359179→1.302938, 충돌20.697674→20.988372%였다.

입력은 공개좌표관측9점·미래12점·0.4초·nominal meter다. 원시Sol response를새로파싱하고parsed/숫자캐시와일치검사한뒤기존관측CV와결합한다. 캐시가없으면실패한다. 새영상서비스가아니다.

계열루트 `HWP_CJ4DPLEX_AI/src/api_based`에서:
```bash
python3 main.py b3 baseline status
python3 main.py b3 ours predict --output results/b3/my_ours
python3 main.py b3 ours evaluate --prediction results/b3/my_ours/predictions.npz --output results/b3/my_score
python3 main.py b3 ours verify --output results/b3/my_verify
```

results/ 인수는 `HWP_CJ4DPLEX_AI/results/api_based`로매핑된다. 이번실행은 `b3/ours_selected`, `ours_score`, `ours_verify`에있다. 과거selected배열과maxdiff0, 공식점수동일성을확인했다. baseline과ours는같은평가기와집합으로비교한다. API누락4를제외해전체제공으로표시하지않는다.

[전체방법/API·로컬적용표와학습출처](EXPERIMENTS_AND_TRAINING.md)를함께참조한다. source버전·원본/복사SHA는 `data/b3/ours/manifest.json`, 원실험은 `sol_postprocess_b3_20261110`이다. 기존공식평가기라이선스와원천데이터조건을유지한다. 대용량데이터는Git추적대상이아니며실행에는로컬복사자산과branch.json의기존환경이필요하다.

