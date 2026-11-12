"""동일 240,890장면의 원래 재현과 추가 attention 전이를 분리해 집계한다."""
import csv
import json
from adapter_io import ROOT,sha

directory=ROOT/'evaluation';directory.mkdir(exist_ok=True);plan=json.loads((ROOT/'runplan.json').read_text());folds={};status=[];receipts={}
for fold in (1,2,3):
    path=ROOT/'scores'/f'fold{fold}/eval';receipt=json.loads((path/'receipt.json').read_text());summary=path/'summary.json';assert sha(summary)==receipt['summary_sha256']
    for row in receipt['files']:assert sha(ROOT/row['artifact'])==row['score_sha256']
    folds[fold]={r['method']:r for r in json.loads(summary.read_text())};receipts[str(fold)]=sha(path/'receipt.json')
    manifest=json.loads((ROOT/'prediction_manifests'/f'fold{fold}_eval.json').read_text());status+=manifest['expected_status_rows']
(directory/'method_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
with (directory/'method_status.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(status[0]));writer.writeheader();writer.writerows(status)
assert len(status)==3*158 and all(r['status']=='prediction_verified' for r in status),'incomplete_full_scope'
assert all(sum(row['scenes'] for row in plan['groups'][plan['folds'][f-1]['evaluation']])==next(iter(folds[f].values()))['requested'] for f in folds)
fields=('ADE','FDE','ColI_all_GT_percent','ColI_observed_GT_percent')

def aggregate(name: str) -> dict:
    values=[folds[f][name] for f in (1,2,3)];n=sum(v['provided'] for v in values);assert n==240890 and all(v['missing']==0 for v in values)
    return {'method':name,'requested':240890,'provided':n,'missing':0,**{field:sum(v[field]*v['provided'] for v in values)/n for field in fields},'ColII_predicted_neighbors':None,'units':'m (official format); individual calibration not independently verified'}

all_methods=sorted(set.intersection(*(set(rows) for rows in folds.values())));raw=[aggregate(name) for name in all_methods]
lookup={r['method']:r for r in raw};original=json.loads((ROOT/'method_registry.json').read_text());extension=json.loads((ROOT/'attention_transfer_amendment.json').read_text())['methods'];expanded=[]
for scope,registry in [('original_reproduction',original),('attention_transfer_addition',extension)]:
    for row in registry:
        name=row['canonical'];key='C/'+name.split('/',1)[1] if name.startswith('analytic/') else name
        expanded.append({**lookup[key],'method':row['method'],'canonical':name,'scope':scope})
for name,rows in [('all_verified_raw_methods',raw),('full_registry_methods',expanded)]:
    (directory/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    with (directory/(name+'.csv')).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
choices=[]
for fold in (1,2,3):
    selection=json.loads((ROOT/'selections'/f'fold{fold}.json').read_text());entry=plan['folds'][fold-1]
    choices.append({'fold':fold,'train_group':entry['train'],'train_n':sum(r['scenes'] for r in plan['groups'][entry['train']]),'dev_group':entry['development'],'eval_group':entry['evaluation'],'selection':selection['ours'],'baseline_eval':folds[fold]['loop/direct_unshared4_d4'],'ours_eval':folds[fold]['ours/selected'],'ablation_eval':{name:folds[fold][name] for name in selection['ours']['ablation_comparators'] if name in folds[fold]}})
comparison={'protocol':'custom 3-supergroup CV on official public real training data; not official hidden challenge score','scene_weighted':{name:lookup[name] for name in ['loop/direct_unshared4_d4','ours/selected','C/hold','C/constant_velocity','C/kalman']},'group_macro':{name:{field:sum(folds[f][name][field] for f in (1,2,3))/3 for field in fields} for name in ['loop/direct_unshared4_d4','ours/selected']},'per_fold':choices,'evaluation_scene_count':240890,'original_registry_rows':147,'original_canonical':133,'additional_configs':11,'selection_used_other_fold_eval':False,'ColII_predicted_neighbors':None,'public_hidden_GT_score':None}
(directory/'baseline_ours_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
(directory/'receipt.json').write_text(json.dumps({'source':sha(__file__),'score_receipts':receipts,'selection_lock':sha(ROOT/'all_fold_selections_frozen.json'),'outputs':{p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='receipt.json'},'complete':True},indent=2))
print(json.dumps({'full_registry_rows':len(expanded),'status_rows':len(status),'all_verified_raw_methods':len(raw),'scenes':240890,'primary_comparison':comparison['scene_weighted']},ensure_ascii=False),flush=True)
