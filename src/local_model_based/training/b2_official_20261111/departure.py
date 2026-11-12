"""VA회귀와 별개인 변화분류·선택적 제공률의 공식 train 대응 실험."""
import hashlib
import sys
import time
from types import SimpleNamespace
import numpy as np
from common import ROOT,read,save,guard,digest,freeze_contract
from selective import fit_departure,crc_threshold,departure_metrics
from learning import npz

def inputs():
    rows=read(ROOT/'manifest.json');a=np.load(ROOT/'inputs.npz')
    features=np.concatenate([np.repeat(a['numeric'][:,None,:],4,1),np.broadcast_to(np.array([.5,1,2,3])[None,:,None],(len(rows),4,1))],2)
    return rows,features,np.array([r['split'] for r in rows]),np.array([r['group'] for r in rows])

def targets(rows,files):
    table={r['sample_id']:r for f in files for r in read(ROOT/f'labels_{f}.json')}
    keep=np.array([r['sample_id'] in table for r in rows])
    return keep,np.array([table[r['sample_id']]['future'] for r in rows if r['sample_id'] in table]),np.array([table[r['sample_id']]['current'] for r in rows if r['sample_id'] in table])

def train():
    fingerprint=freeze_contract(ROOT/'departure_contract.json',{'hashes':{name:digest(ROOT/name) for name in
        ['departure.py','selective.py','inputs.npz','labels_train.json','labels_dev.json','supplement_runplan.json']}})
    if (ROOT/'departure_model.json').exists():
        assert read(ROOT/'departure_model.json')['training_fingerprint']==fingerprint
        return
    started=time.perf_counter();rows,x,splits,groups=inputs();mask,y,current=targets(rows,['train','dev'])
    data=SimpleNamespace(features=x[mask],truth=y,current_gt=current,groups=groups[mask],split=np.where(splits[mask]=='train','development','calibration'))
    result=fit_departure(data);dev=data.split=='calibration';truth=np.linalg.norm(y-current[:,None,:],axis=-1)>result.threshold
    confidence=np.maximum(result.probabilities[dev],1-result.probabilities[dev]).ravel();errors=((result.probabilities[dev]>=.5)!=truth[dev]).ravel()
    thresholds={};curves={}
    for alpha in [.1,.2,.3]:
        threshold,curve=crc_threshold(confidence,errors,np.repeat(groups[mask][dev],4),alpha);thresholds[str(alpha)]=threshold;curves[str(alpha)]=curve
    count=int(np.ceil(.7*len(confidence)));confidence_keep=np.zeros(len(confidence),bool);confidence_keep[np.argsort(-confidence,kind='stable')[:count]]=True
    identities=[hashlib.sha256(f"{r['sample_id']}|{h}".encode()).hexdigest() for r in rows if r['split']=='dev' for h in [.5,1.,2.,3.]]
    hash_keep=np.zeros(len(confidence),bool);hash_keep[np.argsort(identities)[:count]]=True
    dev_probability=result.probabilities[dev].ravel();dev_truth=truth[dev].ravel();dev_groups=np.repeat(groups[mask][dev],4);horizons=np.tile([.5,1,2,3],int(dev.sum()))
    policy={name:departure_metrics(dev_truth,dev_probability,keep,dev_groups,horizons) for name,keep in [('confidence',confidence_keep),('hash_control',hash_keep)]}
    selected='confidence70' if policy['confidence']['error']<policy['hash_control']['error'] and policy['confidence']['alarm_recall']>=policy['hash_control']['alarm_recall'] else 'hash_control70'
    save(ROOT/'departure_model.json',{'C':result.c,'departure_threshold':result.threshold,'development':result.development,'coefficients':result.coefficients,
         'crc_thresholds':thresholds,'crc_curves':curves,'train_samples':int((splits=='train').sum()),'calibration_samples':int(dev.sum()),
         'prior':float(truth[data.split=='development'].mean()),'seconds':time.perf_counter()-started,
         'dev_samecoverage70':policy,'selected_policy':selected,
         'eval_GT_opened':False,'formal_risk_guarantee':False,'meaning':'continuous VA target departure classification; not future VA regression','training_fingerprint':fingerprint})

def infer():
    started=time.perf_counter();rows,x,_,_=inputs();model=read(ROOT/'departure_model.json');c=model['coefficients']
    logits=((x.reshape(-1,7)-np.array(c['mean']))/np.array(c['scale']))@np.array(c['coef']).T+np.array(c['intercept'])
    p=1/(1+np.exp(-logits));npz(ROOT/'departure_predictions.npz',probability=p.reshape(-1,4))
    save(ROOT/'departure_inference.json',{'samples':len(rows),'seconds':time.perf_counter()-started,'GT_access':False,'sha256':digest(ROOT/'departure_predictions.npz'),'model_sha256':digest(ROOT/'departure_model.json')})

def score():
    receipt=read(ROOT/'departure_inference.json');assert digest(ROOT/'departure_model.json')==receipt['model_sha256'] and digest(ROOT/'departure_predictions.npz')==receipt['sha256']
    rows,_,splits,groups=inputs();_,y,current=targets(rows,['train','dev','eval']);model=read(ROOT/'departure_model.json')
    p=np.load(ROOT/'departure_predictions.npz')['probability'];truth=np.linalg.norm(y-current[:,None,:],axis=-1)>model['departure_threshold']
    hold=np.load(ROOT/'predictions/hold.npz')['prediction'];results=[]
    for split in ['dev','official_test','additional']:
        mask=splits==split;pp=p[mask].ravel();yy=truth[mask].ravel();gg=np.repeat(groups[mask],4);hh=np.tile([.5,1,2,3],mask.sum())
        confidence=np.maximum(pp,1-pp);order=np.argsort(-confidence,kind='stable')
        identities=[hashlib.sha256(f"{r['sample_id']}|{h}".encode()).hexdigest() for r in rows if r['split']==split for h in [.5,1.,2.,3.]]
        hash_order=np.argsort(identities);conditions=[]
        for coverage in [.5,.7,.9,1.]:
            k=int(np.ceil(coverage*len(pp)))
            for name,ranking in [('confidence',order),('hash_control',hash_order)]:
                committed=np.zeros(len(pp),bool);committed[ranking[:k]]=True
                conditions.append((name,coverage,pp,committed))
        for alpha,threshold in model['crc_thresholds'].items():conditions.append(('CRC_alpha_'+alpha,None,pp,confidence>=threshold))
        conditions.append(('training_prior',1.,np.full(len(pp),model['prior']),np.ones(len(pp),bool)))
        for name,coverage,prob,committed in conditions:
            result={'split':split,'method':name,'requested_coverage':coverage,**departure_metrics(yy,prob,committed,gg,hh)}
            errors=np.abs(y[mask]-hold[mask]).mean(2).ravel();result['retained_hold_VA_MAE']=float(errors[committed].mean()) if committed.any() else None
            results.append(result)
    save(ROOT/'departure_scores.json',results)
    save(ROOT/'departure_evaluation_receipt.json',{'model_sha256':digest(ROOT/'departure_model.json'),'score_rows':len(results),'eval_samples':int(np.isin(splits,['official_test','additional']).sum()),'status':'completed','formal_risk_guarantee':False})

if __name__=='__main__':
    stage=sys.argv[1];guard(stage);{'train':train,'infer':infer,'score':score}[stage]()
