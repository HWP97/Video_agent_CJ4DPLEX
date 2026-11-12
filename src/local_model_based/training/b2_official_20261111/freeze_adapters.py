"""기존 실험의 재현 범위와 별도 입력 설정을 학습 전에 기록한다."""
from common import ROOT,read,save,digest

def main():
    files=['/mnt/RG2/llm_transfer_20261110/b2/temporal.py','/mnt/RG2/llm_transfer_20261110/b2/experiment.py',
           '/mnt/RG2/llm_transfer_expansion_20261110/b2/models.py',
           '/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/b2_chronos/run.py',
           '/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/b2_chronos/prepare.py',
           '/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/assets/chronos2/adapter.py']
    config={'primary_count':33,'primary_plan_sha256':digest(ROOT/'runplan.json'),
       'chronos_configs':[{'method':f'chronos_{length}_{mode}','length_s':seconds,'joint':mode=='joint','rate_hz':10,'prediction_steps':30,'indices':[4,9,19,29]} for length,seconds in [('short',1.),('long',7.2)] for mode in ['independent','joint']],
       'dense_local_controls':['linear_short','linear_long','summary_ridge_short','summary_ridge_long'],
       'dense_input':'기존 Chronos 실험의 별도 확장입력. 10Hz 각 시점 이전5프레임 인식; 기본33개 입력과 다름',
       'numeric_ridge':'로컬6변수 표준화 직접 미래예측 대응. λ1/10/100, 합제곱손실 Ridge; API조건/교사 없음',
       'local_pool':'기존 PCAmean32,last32,local6+영열10; train PCA/표준화; 평균제곱손실 λ1',
       'incompatible':[{'family':'HiEF','status':'excluded_separate_task','reason':'감정 이진/분류 head이며 미래VA 8좌표 예측과 다른 과제'},
                       {'family':'CRC/departure','status':'excluded_separate_task','reason':'현재GT 대비 변화 여부 분류·선택적 제공률 과제; VA예측과 합산하지 않음'},
                       {'family':'Sol residual/currentbias/teacher/OOF','status':'excluded_API_conditioned','reason':'API교사 금지; 대응 direct GRU/meanpool/linear/summary/memory만 포함'},
                       {'family':'b2 연구 제안만 존재하는 모델','status':'not_previously_executed','reason':'새 모델 탐색·새 대형 인식기 학습 범위 밖'}],
       'source_hashes':{p:digest(p) for p in files},'seed':42,'backbone_fit':False,'API_calls':0}
    path=ROOT/'adapter_runplan.json'
    if path.exists():assert read(path)==config
    else:save(path,config)
    receipt=read(ROOT/'manifest_receipt.json');receipt.pop('groups_disjoint',None)
    receipt['train_dev_groups_disjoint']=True;receipt['official_test_split']='same original videos, held-out later time segment'
    rows=read(ROOT/'manifest.json');receipt['effective_group_counts']={s:len({r['group'] for r in rows if r['split']==s}) for s in ['train','dev','official_test','additional']}
    save(ROOT/'manifest_receipt.json',receipt)

if __name__=='__main__':main()
