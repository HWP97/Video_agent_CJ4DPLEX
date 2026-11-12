"""완료 receipt와 shape/finite/order 계약을 확인한 예측만 채점 manifest에 넣는다."""
from pathlib import Path
import hashlib
import json
import numpy as np
from adapter_io import ROOT,sha


def freeze(fold,role):
    plan=json.loads((ROOT/'runplan.json').read_text());group=plan['folds'][fold-1][{'dev':'development','eval':'evaluation'}[role]]
    metadata=[r for item in plan['groups'][group] for r in json.loads((ROOT/'data/indices'/(Path(item['file']).stem+'_index.json')).read_text())]
    order=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest();ids=[r['sample_id'] for r in metadata];found={};evidence={}
    def accept(name,path,digest,receipt,order_mode):
        assert sha(path)==digest,(name,'hash')
        array=np.load(path,mmap_mode='r');assert array.shape==(len(ids),12,2) and np.isfinite(array).all(),name
        found[name]=path;evidence[name]={'path':str(path.relative_to(ROOT)),'sha256':digest,'receipt':str(receipt.relative_to(ROOT)),'receipt_sha256':sha(receipt),'order_sha256':order,'order_verification':order_mode}
    attention=ROOT/'attention'/f'fold{fold}'
    completed=attention/(role+'_inference_receipt.json')
    if role=='eval' and (attention/'eval_inference_receipt_role_bound.json').exists():completed=attention/'eval_inference_receipt_role_bound.json'
    if completed.exists():
        value=json.loads(completed.read_text());assert value['fold']==fold and value['role']==role and value['order_sha256']==order and value['n']==len(ids) and value['runplan']==sha(ROOT/'runplan.json')
        if 'original_receipt_sha256' in value:assert value['original_receipt_sha256']==sha(attention/'eval_inference_receipt.json')
        for name,model in value['models'].items():
            assert sha(attention/(name+'.pt'))==model['checkpoint'] and sha(attention/(name+'.json'))==model['receipt']
        for name,item in value['outputs'].items():accept('loop/'+name,attention/item['path'],item['sha256'],completed,'producer metadata SHA equality')
    baseline=attention/f'direct_unshared4_{role}.npy';receipt=baseline.with_suffix('.json')
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['sample_ids']==ids and r['runplan_sha256']==sha(ROOT/'runplan.json') and r['checkpoint_sha256']==sha(attention/'direct_unshared4.pt')
        accept('loop/direct_unshared4_d4',baseline,r['sha256'],receipt,'exact sample_ids equality')
    for folder,prefix in [('fast','fast_trm'),('masked','new'),('memory','memory_ttt'),('residual','residual'),('reduced_papers','papers'),('selector','C')]:
        base=ROOT/'adapters'/folder/f'fold{fold}';receipt=base/f'{role}_inference_receipt.json'
        if not receipt.exists():receipt=base/f'{role}_receipt.json'
        if not receipt.exists():continue
        value=json.loads(receipt.read_text())
        if isinstance(value.get('role'),dict):assert value['role']['order_sha256']==order and value['role']['n']==len(ids);mode='producer metadata SHA equality'
        else:assert value.get('n')==len(ids);mode='producer deterministic load_role order; controller binds metadata SHA'
        for name,digest in value['outputs'].items():
            accept(prefix+'/'+name,base/role/(name+'.npy'),digest,receipt,mode)
    for receipt in (ROOT/'adapters/chronos/groups'/group).glob('*.json'):
        value=json.loads(receipt.read_text());path=receipt.with_suffix('.npy')
        if 'signature' in value and path.exists():assert value['signature']['n']==len(ids);accept('fast_trm/'+receipt.stem,path,value['sha256'],receipt,'group order follows load_role')
    receipt=ROOT/'adapters/official_re/groups'/group/'receipt.json'
    if receipt.exists():
        value=json.loads(receipt.read_text());assert value['signature']['metadata_order_sha256']==order and value['signature']['n']==len(ids)
        for name,digest in value['outputs'].items():accept('papers/'+name,receipt.parent/(name+'.npy'),digest,receipt,'producer metadata SHA equality')
    selected=ROOT/'selections'/f'fold{fold}.json'
    if selected.exists():
        value=json.loads(selected.read_text())
        for name,chosen in value['aliases'].items():
            if chosen in found:accept(name,found[chosen],evidence[chosen]['sha256'],selected,'frozen dev alias of verified '+chosen)
    diagnostic=ROOT/'selections'/f'fold{fold}'/(role+'_diagnostic.json')
    if diagnostic.exists():
        value=json.loads(diagnostic.read_text());assert value['order_sha256']==order and value['n']==len(ids)
        accept('loop/idea_diagnostic_no_neighbors',diagnostic.parent/(role+'_no_neighbors.npy'),value['sha256'],diagnostic,'producer metadata SHA equality')
    statuses=[]
    expected=json.loads((ROOT/'method_registry.json').read_text())
    amendment=ROOT/'attention_transfer_amendment.json'
    if amendment.exists():expected+=json.loads(amendment.read_text())['methods']
    for row in expected:
        canonical=row['canonical'];key='C/'+canonical.split('/',1)[1] if canonical.startswith('analytic/') else canonical
        statuses.append({'method':row['method'],'canonical':canonical,'fold':fold,'role':role,'status':'prediction_verified' if key in found else 'not_ready_or_pending_selection','prediction':key if key in found else None,'requested':len(ids)})
    destination=ROOT/'prediction_manifests';destination.mkdir(exist_ok=True)
    out=destination/f'fold{fold}_{role}.json';temporary=out.with_suffix('.tmp')
    temporary.write_text(json.dumps({'fold':fold,'role':role,'n':len(ids),'runplan_sha256':sha(ROOT/'runplan.json'),'source_sha256':sha(__file__),'metadata_order_sha256':order,'verified_predictions':evidence,'expected_status_rows':statuses,'complete_scope':all(r['status']=='prediction_verified' for r in statuses)},indent=2));temporary.replace(out)
    return found
