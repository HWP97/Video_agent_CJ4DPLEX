"""eval 점수를 열기 전에 최종 ours 후보와 대조군 구분을 기록한다."""
import json
from adapter_io import ROOT,sha

rows=[]
for canonical in sorted({r['canonical'] for r in json.loads((ROOT/'method_registry.json').read_text())}):
    family,name=canonical.split('/',1);eligible=False;kind='대조군 또는 별도 선택 별칭';controls=[]
    if family=='loop' and any(s in name for s in ('shared2_d','shared4_d','dynamic_d')) and 'unshared' not in name:
        eligible=True;kind='attention 반복 구조 변경';anchor='kf' if name.startswith('kf_') else 'direct';controls=[f'loop/{anchor}_single_d1',f'loop/{anchor}_unshared4_d4',f'loop/{anchor}_unshared4_small_d4']
    if family=='fast_trm' and name in ('direct_dct_bpe','local_dct_bpe','direct_two_state_deep','local_two_state_deep'):
        eligible=True;kind='독립 토큰 모델' if name.startswith('direct_') else 'attention 기준 잔차 토큰 모델';anchor=name.split('_',1)[0];controls=[f'fast_trm/{anchor}_continuous',f'fast_trm/{anchor}_raw_bpe'] if 'bpe' in name else [f'fast_trm/{anchor}_single_state',f'fast_trm/{anchor}_two_state_last']
    if family=='new' and name in ('direct_masked','local_masked','direct_mtp','local_mtp'):
        eligible=True;kind='독립 masked/MTP 모델' if name.startswith('direct_') else 'attention 기준 masked/MTP 잔차 모델';controls=['new/'+name.rsplit('_',1)[0]+('_parallel' if name.endswith('masked') else '_ar')]
    if family=='memory_ttt' and not any(s in name for s in ('selected','shuffled','identity','disabled','ordinaryfit')):
        eligible=any(s in name for s in ('_memory_k','_adapt_r','_static_r','memory_direct_','direct_prefix_adapted'))
        if eligible:
            kind='관측 prefix 적응 또는 train-only 기억/정적 잔차';anchor='local' if name.startswith('local') else 'kalman';controls=[f'memory_ttt/{anchor}_update_disabled',f'memory_ttt/{anchor}_identity_update']
            if name.startswith('memory_direct_'):controls=['C/constant_velocity','C/kalman']
            if name.startswith('direct_prefix_adapted'):controls=['memory_ttt/'+name.replace('adapted','ordinaryfit_unadapted'),'memory_ttt/'+name.replace('adapted','ordinaryfit_updated'),'memory_ttt/'+name.replace('adapted','update_disabled')]
    if family=='residual' and name in ('kalman_self','kalman_social','kalman_social_collision'):
        eligible=True;kind='독립 KF 잔차 MLP';controls=['residual/kalman_zero','residual/kalman_self'] if name!='kalman_self' else ['residual/kalman_zero']
    if family=='papers' and name in ('re_default','re_selected','moif_default','moif_selected'):
        eligible=True;kind='축소 논문 구조 독립 모델';controls=['papers/self_control']
    rows.append({'canonical':canonical,'eligible_for_ours':eligible,'method_type':kind,'ablation_comparators':controls})
result={'runplan_sha256':sha(ROOT/'runplan.json'),'catalog_source_sha256':sha(__file__),'baseline':'loop/direct_unshared4_d4','gate':'current fold dev ADE strictly lower, FDE no higher, ColI_observed_GT_percent no higher; then minimum dev ADE, FDE, collision, method name; none passing => baseline 유지/개선없음','excluded':'raw hold/CV/KF/Chronos/official pretrained Re, same-baseline aliases, deliberate shuffled/identity/disabled/zero/random controls; no oracle best-of-K','input_policy':'evaluation GT may have been read for data preparation and metric QA; never model input or selection metric; final eval scoring locked until all-fold selection freeze','methods':rows,'eligible_count':sum(r['eligible_for_ours'] for r in rows)}
path=ROOT/'ours_candidate_catalog.json'
if path.exists():assert json.loads(path.read_text())==result,'catalog_is_already_frozen'
else:path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'canonical_methods':len(rows),'eligible':result['eligible_count'],'catalog':str(path)},ensure_ascii=False))
