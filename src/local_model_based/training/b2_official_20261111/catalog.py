"""문서별 기존·제안 실험을 실제 산출물 상태와 연결한다."""
from common import ROOT,read,save,digest

def main():
    base='/mnt/RG2/'
    rows=[]
    def add(source,idea,methods,baseline,reason=None,protocol='official_VEATIC_temporal_forecast'):
        predictions=[str(ROOT/'predictions'/(m+'.npz')) for m in methods]
        scores=ROOT/'scores.json';done=scores.exists() and all(__import__('pathlib').Path(p).exists() for p in predictions)
        status='completed' if methods and done else 'configured_pending_execution' if methods else 'blocked' if reason else 'not_applicable'
        rows.append({'source_document':base+source,'idea':idea,'configs':methods,'same_input_baseline':baseline,
                     'train_source':'first98 first floor70%, head train77 effective original groups/1169 windows; dev20/265',
                     'protocol':protocol,'status':status,'predictions':predictions,'score':str(scores) if done else None,'not_executed_reason':reason})
    configs=read(ROOT/'runplan.json')['configs']
    add('my_works/b2/scripts/feature_baselines.py','공식 동결 인식기의 상태유지와 선형추세',['hold','linear_trend'],'hold')
    add('HWP_CJ4DPLEX_AI/docs/LOCAL_TRAINING_NEXT.md','공식 train local6 직접 Ridge',[c['method'] for c in configs if c['family']=='numeric_ridge'],'hold')
    add('HWP_CJ4DPLEX_AI/docs/LOCAL_TRAINING_NEXT.md','같은 λ 표기의 손실 정규화 분리 대조',['numeric_meanloss_1'],'numeric_ridge_1')
    add('llm_transfer_20261110/b2/wave2.py','관측 시각평균 Ridge',['localpool_ridge'],'hold')
    add('llm_transfer_20261110/b2/experiment.py','직접 GRU와 시간순서 대조학습6개',[c['method'] for c in configs if c['family']=='gru'],'hold')
    for family,idea in [('mean','train 미래평균'),('retrieval','group 제외 로컬 exemplar retrieval10개'),('summary','미래요약 보조감독 MLP6개'),('linear','저랭크·직접 선형4개')]:
        add('llm_transfer_expansion_20261110/b2/models.py',idea,[c['method'] for c in configs if c['family']==family],'hold')
    adapter=read(ROOT/'adapter_runplan.json')
    add('llm_transfer_expansion_20261110/additional_candidates/b2_chronos/run.py','Chronos 공개가중치4개',[c['method'] for c in adapter['chronos_configs']],'linear_short/long and summary_ridge_short/long',protocol='same8s window, separately expanded10Hz recognition inputs')
    add('llm_transfer_expansion_20261110/additional_candidates/b2_chronos/run.py','확장입력의 동일입력 비교군',adapter['dense_local_controls'],'hold',protocol='separate10Hz recognition input')
    add('paper_experiments_20261109/b2/hi_ef.py','맥락 집계 아이디어의 VEATIC 회귀 대응',['context_'+v for v in read(ROOT/'supplement_runplan.json')['context_modes']],'context_current',protocol='adaptation not original MCIS reproduction')
    add('previous_exp/B/experiment.py','Sol 변화량 alpha의 로컬 대응',['local_alpha_'+f'{v:g}' for v in [0,.25,.5,.75,1]],'localpool_ridge and hold')
    add('research_plans/2026-11-09_submission/계획_B2_현재값보정.md','원본그룹 중앙값 현재offset의 로컬 대응',['offset_'+v for v in ['hold','linear_trend','localpool_ridge']],'same uncorrected methods')
    add('paper_experiments_20261109/b2/PROTOCOL.md','원MCIS 7class Hi-EF 공식train 재현',[],'clipIII classifier',reason='공개85행 exploratory tensor는 공식train split/encoder provenance 미확인. VEATIC VA를7class로 변조하지 않음.',protocol='MCIS separate_task blocked_official_train_unverified')
    add('paper_experiments_20261109/b2/selective.py','변화분류 Logistic3 C + original-group CRC + samecoverage 평가',[],'training prior and same-k hash_control',protocol='separate departure classification')
    rows[-1].update(status='completed' if (ROOT/'departure_scores.json').exists() else 'configured_pending_execution',
        configs=['C0.1','C1','C10','CRC0.1','CRC0.2','CRC0.3','coverage0.5','coverage0.7','coverage0.9','coverage1.0'],
        predictions=[str(ROOT/'departure_predictions.npz')],score=str(ROOT/'departure_scores.json'))
    for source,idea,reason in [
        ('baseline_residual_20261110/b2/model.py','API currentbias/잔차 Ridge9개','Sol입력 필요. 숫자 로컬6 direct Ridge를 별도 대응하고 API교사·응답은 사용하지 않음.'),
        ('fixed_baseline_bottleneck_20261110/b2/experiment.py','API residual 축소·직교화·offset·direct','API residual 원형은 실행 금지. 로컬 alpha/offset/direct는 별도 대응하며 API 학습을 실행했다고 세지 않음.'),
        ('research_runs/NEW_METHODS.md','사건 종류×발생시점 head','대상별 onset label과 호환 과제 head가 없음. 연속VA를 임의 onset class로 바꾸지 않음.'),
        ('research_runs/NEW_METHODS.md','frozen V-JEPA 특징+VA head','호환 VA train 특징·head 미확보; 새 대형모델 다운로드 금지. 기존 타과제encoder를 VA성공으로 재명명하지 않음.'),
        ('research_plans/2026-11-09_submission/계획_Q1_시간진단.md','Qwen 시간순서·숫자출력 진단','AG 관계 b1 과제 문서이며 b2 VA 실험 아님. b2 GRU순서 대조와 혼동하지 않음.')]:
        add(source,idea,[],'해당 원과제 baseline',reason=reason)
    sources=['paper_experiments_20261109/DESIGN.md','paper_experiments_20261109/CONTRIBUTION_PLAN.md','paper_experiments_20261109/START_HERE.md',
             'paper_experiments_20261109/b2/PROTOCOL.md','paper_experiments_20261109/b2/SOURCE_MAPPING.md',
             'HWP_CJ4DPLEX_AI/src/local_model_based/ours/b2/EXPERIMENTS_AND_TRAINING.md','HWP_CJ4DPLEX_AI/docs/LOCAL_TRAINING_NEXT.md','HWP_CJ4DPLEX_AI/docs/B2_TRAIN_DATA.md']
    save(ROOT/'experiment_checklist.json',{'experiments':rows,'source_document_hashes':{base+p:digest(base+p) for p in sources},
       'other_folders':{'metric_local_followup_20261110':'b1/b3 only, no b2 experiment','sol_postprocess_b3_20261110':'b3 only','C':'motion/web, no new b2 head','my_works':'early14/65 and full1258 are earlier sample protocols, not new method families'},
       'not_claimed':'blocked/separate-task rows are not completed experiments; new MCIS official train reproduction not performed'})

if __name__=='__main__':main()
