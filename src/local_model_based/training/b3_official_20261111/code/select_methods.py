"""각 fold dev 점수만 사용하여 원 family gate와 최종 ours를 동결한다."""
import json
import sys
from adapter_io import ROOT,sha


def passes(value, base, epsilon=0.) -> bool:
    return value['provided']==base['provided'] and value['ADE']<base['ADE']-epsilon and value['FDE']<=base['FDE']+epsilon and value['ColI_observed_GT_percent']<=base['ColI_observed_GT_percent']+(1e-12 if epsilon else 0.)


def validate_scores(summary, receipt, fold) -> None:
    assert receipt['fold']==fold and receipt['role']=='dev' and receipt['runplan']==sha(ROOT/'runplan.json')
    assert receipt['summary_sha256']==sha(summary) and receipt['source']==sha(ROOT/'code/score_predictions.py')
    assert receipt['metric_contract']==sha(ROOT/'metric_amendment.json') and receipt['collision_source']==sha(ROOT/'code/collision_metric.py')
    path=ROOT/receipt['prediction_manifest'];assert sha(path)==receipt['prediction_manifest_sha256'];manifest=json.loads(path.read_text())
    assert manifest['fold']==fold and manifest['role']=='dev' and manifest['runplan_sha256']==sha(ROOT/'runplan.json') and manifest['source_sha256']==sha(ROOT/'code/freeze_predictions.py') and manifest['n']==receipt['scenes']
    assert receipt['predictions']=={name:value['sha256'] for name,value in manifest['verified_predictions'].items()}
    for value in manifest['verified_predictions'].values():
        assert sha(ROOT/value['path'])==value['sha256'] and sha(ROOT/value['receipt'])==value['receipt_sha256']
    for value in receipt['files']:
        assert sha(ROOT/value['artifact'])==value['score_sha256']
        for folder,suffix,key in [('data/targets','_labels.npz','GT_sha256'),('data/indices','_index.json','index_sha256'),('evaluation_only','.npz','evaluation_cache_sha256')]:assert sha(ROOT/folder/(value['file']+suffix))==value[key]


def main() -> None:
    fold=int(sys.argv[1]);directory=ROOT/'selections';directory.mkdir(exist_ok=True);out=directory/f'fold{fold}.json'
    if out.exists():
        previous=json.loads(out.read_text());assert previous['source']==sha(__file__) and previous['candidate_catalog']==sha(ROOT/'ours_candidate_catalog.json') and previous['attention_transfer_amendment']==sha(ROOT/'attention_transfer_amendment.json') and previous['runplan']==sha(ROOT/'runplan.json')
        snapshot=directory/f'fold{fold}_dev_scores.json';assert sha(snapshot)==previous['dev_scores'];validate_scores(snapshot,previous['score_receipt'],fold);print('already_frozen',fold);return
    path=ROOT/'scores'/f'fold{fold}/dev/summary.json';score_receipt=json.loads((path.parent/'receipt.json').read_text());validate_scores(path,score_receipt,fold);rows=json.loads(path.read_text());metrics={r['method']:r for r in rows}
    catalog=json.loads((ROOT/'ours_candidate_catalog.json').read_text());extension=json.loads((ROOT/'attention_transfer_amendment.json').read_text());catalog['methods']+=extension['methods'];candidates=[r['canonical'] for r in catalog['methods'] if r['eligible_for_ours']]
    missing=sorted(set(candidates)-metrics.keys());assert not missing,('required_candidates_missing',missing)
    base='loop/direct_unshared4_d4';aliases={};details={}
    order=lambda name:(metrics[name]['ADE'],metrics[name]['FDE'],metrics[name]['ColI_observed_GT_percent'],name)
    for family,candidate in [('fast','fast_trm/local_dct_bpe'),('trm','fast_trm/local_two_state_deep')]:
        passed=passes(metrics[candidate],metrics[base]);aliases['fast_trm/selected_'+family]=candidate if passed else base;details[family]={'candidate':candidate,'passed':passed}
    for family,control in [('masked','parallel'),('mtp','ar')]:
        candidate='new/local_'+family;guards={name:passes(metrics[candidate],metrics[name]) for name in [base,'new/local_'+control]}
        aliases['new/selected_'+family]=candidate if all(guards.values()) else base;details[family]=guards
    for family,names in [('memory',[f'memory_ttt/local_memory_k{k}' for k in (1,5,20)]),('adapt',[f'memory_ttt/local_adapt_r{r}' for r in ('0.1','1.0','10.0','1e-05','0.001')])]:
        candidate=min(names,key=lambda name:metrics[name]['ADE']);passed=passes(metrics[candidate],metrics[base]);aliases['memory_ttt/selected_'+family]=candidate if passed else base;details[family]={'candidate':candidate,'passed':passed,'tie_order':names}
    names=['residual/'+name for name in ('kalman_self','kalman_social','kalman_social_collision')]
    residual_order=lambda name:(metrics[name]['ADE'],metrics[name]['FDE'],metrics[name]['ColI_observed_GT_percent'],names.index(name))
    passing=[name for name in names if passes(metrics[name],metrics['C/kalman'],1e-6)]
    aliases['residual/local_selected']=min(passing,key=residual_order) if passing else 'residual/kalman_zero'
    aliases['residual/local_best_dev_diagnostic']=min(names,key=residual_order)
    reference=min(('C/kalman','residual/kalman_self'),key=lambda name:metrics[name]['ADE'])
    loop=[name for name in metrics if name.startswith('loop/') and any(part in name for part in ('shared2_d','shared4_d','dynamic_d')) and 'unshared' not in name]
    assert len(loop)==24,len(loop)
    passing=[name for name in loop if passes(metrics[name],metrics[reference],1e-6) and metrics[name]['ADE']<metrics['loop/'+('kf' if name.startswith('loop/kf_') else 'direct')+'_single_d1']['ADE']-1e-6]
    aliases['loop/development_selected']=min(passing,key=order) if passing else reference
    aliases['loop/idea_diagnostic']=min(loop,key=order)
    details['loop']={'reference':reference,'passing':passing}
    passing=[name for name in candidates if passes(metrics[name],metrics[base])];winner=min(passing,key=order) if passing else base
    aliases['ours/selected']=winner
    snapshot=directory/f'fold{fold}_dev_scores.json';snapshot.write_bytes(path.read_bytes())
    result={'fold':fold,'source':sha(__file__),'runplan':sha(ROOT/'runplan.json'),'candidate_catalog':sha(ROOT/'ours_candidate_catalog.json'),'attention_transfer_amendment':sha(ROOT/'attention_transfer_amendment.json'),'dev_scores':sha(snapshot),'score_receipt':json.loads((path.parent/'receipt.json').read_text()),'aliases':aliases,'family_details':details,'ours':{'selected':winner,'status':'개선 후보 선택' if winner!=base else '유지/개선없음','passing_candidates':passing,'ablation_comparators':next((r['ablation_comparators'] for r in catalog['methods'] if r['canonical']==winner),[])},'selection_input':'only this fold dev metrics; eval scoring not opened'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({'fold':fold,'aliases':aliases,'ours':result['ours']},ensure_ascii=False))
    paths=[directory/f'fold{i}.json' for i in (1,2,3)]
    if all(p.exists() for p in paths):
        for p in paths:
            r=json.loads(p.read_text());assert r['candidate_catalog']==sha(ROOT/'ours_candidate_catalog.json') and r['source']==sha(__file__) and r['attention_transfer_amendment']==sha(ROOT/'attention_transfer_amendment.json')
        (ROOT/'all_fold_selections_frozen.json').write_text(json.dumps({'folds':{p.name:sha(p) for p in paths},'candidate_catalog':sha(ROOT/'ours_candidate_catalog.json'),'attention_transfer_amendment':sha(ROOT/'attention_transfer_amendment.json'),'runplan':sha(ROOT/'runplan.json'),'eval_scores_used':False},indent=2))


if __name__=='__main__':main()
