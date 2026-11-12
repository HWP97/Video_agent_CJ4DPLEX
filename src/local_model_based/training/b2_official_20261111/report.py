"""실제 완료 산출물만 읽어 한국어 결과표와 실행 원장을 만든다."""
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT,read,save,digest
from learning import metric
from catalog import main as catalog

def main():
    assert read(ROOT/'verification.json')['passed'] and read(ROOT/'pipeline_final.json')['status']=='completed'
    scores=read(ROOT/'scores.json');frozen=read(ROOT/'all_frozen.json');selected=frozen['selected'];rows=read(ROOT/'manifest.json')
    methods=sorted({r['method'] for r in scores});assert len(methods)==55 and len(scores)==220
    table={(r['method'],r['split']):r for r in scores};catalog()
    source_groups=np.array([r['group'] for r in rows]);splits=np.array([r['split'] for r in rows])
    labels={r['sample_id']:r for s in ['train','dev','eval'] for r in read(ROOT/f'labels_{s}.json')};truth=np.array([labels[r['sample_id']]['future'] for r in rows])
    group_scores=[];target_scores=[]
    for name in dict.fromkeys(['hold',selected]):
        p=np.load(ROOT/'predictions'/(name+'.npz'))['prediction']
        for split in ['official_test','additional']:
            for group in sorted(set(source_groups[splits==split])):
                mask=(splits==split)&(source_groups==group);group_scores.append({'method':name,'split':split,'group':group,**metric(truth[mask],p[mask])})
            for vid in sorted({r['video_id'] for r in rows if r['split']==split}):
                mask=np.array([r['split']==split and r['video_id']==vid for r in rows]);target_scores.append({'method':name,'split':split,'video_target_id':vid,**metric(truth[mask],p[mask])})
    save(ROOT/'group_scores.json',group_scores);save(ROOT/'target_scores.json',target_scores)
    pairs={}
    for i,r in enumerate(rows):
        if r['split']=='additional':pairs.setdefault((r['group'],round(r['cutoff_s'],8)),[]).append(i)
    assert len(pairs)==434 and all(len(v)==2 for v in pairs.values())
    paired=np.array(list(pairs.values()));selected_prediction=np.load(ROOT/'predictions'/(selected+'.npz'))['prediction']
    gap=float(np.abs(selected_prediction[paired[:,0]]-selected_prediction[paired[:,1]]).max())
    lower_bound=float((np.abs(truth[paired[:,0]]-truth[paired[:,1]])/2).mean())
    save(ROOT/'additional_pair_diagnostic.json',{'unique_source_times':434,'target_windows':868,'originals':13,'targets':26,
         'selected_prediction_pair_max_abs_difference':gap,'common_prediction_MAE_lower_bound_from_pair_GT':lower_bound,
         'meaning':'scoring-only oracle lower bound for identical per-pair predictions; not an executable model or input'})
    lines=['# 공식 train 기반 b2 로컬 결과','',f'개발집합에서 동결한 방법은 **{selected}**다. 평가 결과를 보고 다시 선택하지 않았다. 55개 VA 설정의 전체2740개 입력에 대해 추론했으며, 공식 뒤30%438개와 추가대상868개를 별도 채점했다.',
      '', '**공식438에서 선택 GRU는 MAE가 낮지만 CCC가 하락했다. 두 지표 동시 개선 성공이 아니며 기존 서비스 기본값은 변경하지 않는다. 추가대상의 CCC도 음수여서 일반화가 약하다. test 결과를 보고 다른 후보로 교체하지 않았다.**',
      '', '과거 내부251개 학습/421개 평가 결과와 분리한 새 실행이다. 공개 인식기는 동결이며, 새 API·교사·다운로드·대형 인식기 학습은0이다. 아래 수치는 실제 파일에서 생성했다.',
      '', '## 동결 선택과 기준선', '', '| split | 방법 | n | MAE↓ | CCC8↑ | 제공률 | raw MAE |', '|---|---|---:|---:|---:|---:|---:|']
    for split in ['dev','official_test','additional']:
        for name in dict.fromkeys(['hold',selected]):
            r=table[name,split];lines.append(f"| {split} | {name} | {r['n']} | {r['MAE']:.6f} | {r['CCC_mean8']:.6f} | {r['coverage']:.1%} | {r['raw_MAE']:.6f} |")
    lines+=['','## 공식438 동일집합: hold·Ridge·선택 GRU','', '| 방법 | 학습 | MAE↓ | CCC8↑ | 제공률 |','|---|---|---:|---:|---:|']
    for name in dict.fromkeys(['hold','localpool_ridge','numeric_ridge_1','numeric_ridge_10','numeric_ridge_100','numeric_meanloss_1',selected]):
        r=table[name,'official_test'];training='추가학습 없음' if name=='hold' else '공식앞70% train1169'
        lines.append(f"| {name} | {training} | {r['MAE']:.6f} | {r['CCC_mean8']:.6f} | {r['coverage']:.1%} |")
    lines+=['','채택은 dev MAE 감소·CCC 비감소 조건으로만 판단했다. 원형 full-scene 인식기는 두 대상의 동일 영상 입력을 구분하지 못한다. 추가868개는13원본·434고유 source-time·26대상이다. 원본별 보조표는 원본 내 두 대상/시간창을 먼저 평균한 뒤 원본 동일 가중이며, 기본표는 대상별 창 동일 가중이다.',
      f'같은 입력의 두 대상에 대한 선택모델 예측 차이는 최대{gap:.9g}다. 두 대상 GT 차이만으로 계산한 공통예측 MAE 하한은{lower_bound:.6f}이며, 이는 채점용 oracle 진단으로 모델 입력이나 실행 가능한 baseline이 아니다.',
      '', '## 모든 VA 설정', '', '| 방법 | dev MAE | dev CCC | 공식test MAE | 공식test CCC | 추가대상 MAE | 추가대상 CCC |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in methods:
        a,b,c=[table[name,s] for s in ['dev','official_test','additional']]
        lines.append(f"| {name} | {a['MAE']:.6f} | {a['CCC_mean8']:.6f} | {b['MAE']:.6f} | {b['CCC_mean8']:.6f} | {c['MAE']:.6f} | {c['CCC_mean8']:.6f} |")
    departure=read(ROOT/'departure_model.json');ds=read(ROOT/'departure_scores.json')
    lines+=['','## 별도 변화분류·기권', '',f"Logistic C는 공식train grouped3fold에서 {departure['C']}로 선택했다. dev70% 같은 제공 개수 비교에서 동결한 정책은 `{departure['selected_policy']}`다. CRC 보정은 dev20원본만 사용했으며 분포무관 보장 주장은 하지 않는다.",
      '', '| split | 정책 | 제공/전체 | 오류↓ | 변화 recall↑ | 원본평균 위험 |', '|---|---|---:|---:|---:|---:|']
    for r in ds:
        if r['split'] in ['official_test','additional'] and (r['requested_coverage']==.7 or r['method'].startswith('CRC')):
            error='N/A' if r['error'] is None else f"{r['error']:.6f}"
            lines.append(f"| {r['split']} | {r['method']} | {r['committed']}/{r['n']} | {error} | {r['alarm_recall']:.6f} | {r['group_macro_risk']:.6f} |")
    bench=read(ROOT/'device_benchmark.json');pilot=read(ROOT/'pilot_equivalence.json')
    lines+=['','## 실행·검증·한계','',f"- 학습 장치 실측: CPU {bench['seconds_per_step']['cpu']:.6f}초/step, GPU0 {bench['seconds_per_step']['cuda:0']:.6f}초/step. 선택 `{bench['selected']}`. 초기5step 제외 후20step이며 합성 배치 시간 측정용 모델은 버렸다.",
      f"- FP32 첫 실제 RGB↔cache 차이 {pilot['max_abs_difference']:.9g}; 기존 허용치2e-5를 유지했다. TF32 실패 시도의 특징·부분학습·로그는 상위 attempt_tf32에 보존했고 재사용하지 않았다.",
      '- 공개 인식기의 공식train 노출 때문에 dev는 head 개발용이다. 공식test는 같은 원본의 뒤30% 시간분할이다. 추가13원본만 train/dev 원본과 다르다.',
      '- numeric λ1/10/100은 합제곱손실(평균손실 유효λ=λ/1169), pool/linear λ1은 평균제곱손실이다. numeric_meanloss_1이 정규화 대조다.',
      '- 55개는 예측 설정 수다. 저장된 신경망17개와 회귀/평균 npz12개, 별도의 current-offset·departure classifier·retrieval bank가 있으며 55개 신경망을 새로 학습한 것이 아니다. 장치속도 측정용 합성배치 모델2개는 버렸다.',
      '- Chronos/dense는 별도10Hz 확장입력이다. 기본15프레임 방법과 입력량이 다르며 동일입력 비교군4개를 함께 보고한다.',
      '- dense가 읽는 원본metadata에는 GT통계·대상설명이 함께 있으나 frame_pts_s만 사용한다. 실제GT배열/labels는 추론에서 차단했고 GT통계·대상설명·미래안내이미지는 입력으로 쓰지 않았다.',
      '- MCIS 공식train 미확인, 사건 onset label·호환 V-JEPA VA head 부재는 실행 불가 항목이다. 로컬 대응을 원논문 전체 재현으로 부르지 않는다.',
      '- 실제코드 정적 LSP는 미설치로 미실행했다. Python parse·실제 경계 실패 probe·RGB 대조·전체추론·독립 재채점/선택 검사를 수행했다.',
      '', '전체 raw/bounded·축별 점수: `scores.json/csv`. 원본별/대상별: `group_scores.json`, `target_scores.json`. 상태와 미실행 근거: `experiment_checklist.json`. 실제 명령·종료코드·시간: `pipeline_*_progress.json`. 재현 명령과 제약: `README.md`.']
    lines+=['','현재 인식/미래변화의 signed 분해는 `FAILURE_ANALYSIS.md`와 `recognition_change_decomposition.json`에 있다. MAE끼리 차감하지 않았고 모델 선택을 바꾸지 않았다. 새 출력 폴더의55개 재추론·220행 재채점은 모두 차이0이며 `replay_verification.json`에 기록했다.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    ledger=[]
    for name in methods:
        ledger.append({'method':name,'status':'completed','prediction':str(ROOT/'predictions'/(name+'.npz')),'prediction_sha256':digest(ROOT/'predictions'/(name+'.npz')),
                       'train_source':'official first98 first70%, head train1169/77groups','dev_samples':265,'official_test_samples':438,'additional_target_samples':868,
                       'selected':name==selected,'scores':{s:table[name,s] for s in ['train','dev','official_test','additional']}})
    save(ROOT/'experiment_ledger.json',ledger)
    save(ROOT/'completion_receipt.json',{'VA_methods':55,'VA_predictions':55*2740,'eval_target_samples_each':1306,'score_rows':220,'selected':selected,
         'verification_sha256':digest(ROOT/'verification.json'),'all_frozen_sha256':digest(ROOT/'all_frozen.json'),'result_sha256':digest(ROOT/'RESULTS.md'),
         'blocked_rows_not_completed':True,'API_calls':0,'original_HWP_modified_by_this_task':False})
    print(json.dumps({'methods':55,'selected':selected,'predictions':55*2740}),flush=True)

if __name__=='__main__':main()
