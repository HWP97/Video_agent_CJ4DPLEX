# b1 ours: ImparTail 유지

**기존 채택 기준을 충족한 로컬 개선이 없어 baseline과 같은 선정 pretrained ImparTail을 유지한다.** ours 이름이 새 학습 방법이나 성능 개선을 뜻하지 않는다. API·Sol teacher/feature를 사용하지 않는다.

공식 train 내부 개발 100그룹씩에서 채택 기준은 기존 ImparTail 대비 R@20 비감소와 mR@20 엄격 증가였다. PGAGS에서 진단 head identity_anchor_no_aux는 R/mR 64.5413/50.6919 → 63.6999/59.4927로 R이 감소했다. GAGS identity_anchor_aux는 67.1797/57.9908 → 68.9321/55.5795로 mR이 감소했다. 따라서 두 mode 모두 원 selection의 adopted=false를 유지한다. 이미 본 test의 증가만으로 이 개발 실패를 뒤집지 않는다.

훈련 데이터는 새 사용자 제공 파일이 아니라 기존 공개 AG/Charades의 공식 train에서 고른 500영상이다. PGAGS는 392 fit+100 dev(8 ineligible), GAGS는 400 fit+100 dev로 48,404 parameter head를 8epoch 실제 학습했다. 선택 뒤 진단용 refit은 492/500영상이다. 공식 관계 GT가 감독이며 Sol teacher/feature는 없었다. 개발100을 전체 학습100으로 오해하면 안 된다. 공개 ImparTail backbone 자체는 저자 pretrained 재사용이고 우리가 학습하지 않았다.

PGAGS는 cap40 sgcls, GAGS는 cap70 predcls 공개 checkpoint다. 공식 source·SceneSayer 호환 source·관측 전용 prefix adapter를 사용한다. baseline과 같은 저장 예측으로 새 공식 재채점이 가능하다. 이미 본 전체 test NC 관측0.5의 ImparTail R/mR은 PGAGS 48.03/33.65, GAGS 63.31/42.30이다. head와 bbox adapter가 다른 주표를 인과적 head 효과로 해석하지 않는다. 같은 bank 비교는 복사한 controlled_full.csv를 따른다.

계열 루트 실행:

```bash
python3 main.py b1 ours status
python3 main.py b1 ours evaluate --sample predcls:00607.mp4_0.5
python3 main.py b1 ours predict --sample predcls:00607.mp4_0.5
```

`final` 별칭도 유지한다. 결과는 최상위 `results/local_model_based/b1`에 저장한다. 예측은 관측 19프레임에서 Faster R-CNN·tracker·ImparTail을 실제 실행하며 기존 출력과 비교한다. 관측 GT bbox, GAGS의 관측 class, 미래 annotation 시각 schedule이 필요한 benchmark다. IoU=0이며 GT 없는 영상 서비스 정확도로 주장하지 않는다. 제공된 새 추론용 원본 해상도 프레임은 대표 사례의 관측0.5까지다. 전체 저장 예측 13,635개와 GT는 모두 있어 전체 재채점은 가능하다.
