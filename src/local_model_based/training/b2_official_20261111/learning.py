"""공식 train만으로 기존 로컬 head들을 재학습하고 별도 프로세스에서 추론한다."""
import csv
import sys
import time
import numpy as np
import torch
from common import ROOT,read,save,digest,guard,freeze_contract
from temporal import Temporal,views,contrast,fit_projection,project
from models import train_summary,SummaryMLP,neural_predict,neighbors,retrieve,linear_fit

def metric(y,p,clip=True):
    p=np.clip(p,-1,1) if clip else p;a=y.reshape(-1,8);b=p.reshape(-1,8)
    c=2*((a-a.mean(0))*(b-b.mean(0))).mean(0)/np.maximum(a.var(0)+b.var(0)+(a.mean(0)-b.mean(0))**2,1e-15)
    return {'n':len(y),'MAE':float(np.abs(y-p).mean()),'CCC_mean8':float(c.mean()),'CCC_dimensions':c.tolist(),'MAE_dimensions':np.abs(y-p).mean(0).ravel().tolist()}

def labels(split,rows):
    table={r['sample_id']:r for r in read(ROOT/f'labels_{split}.json')}
    return np.array([table[r['sample_id']]['future'] for r in rows if r['sample_id'] in table])

def npz(path,**values):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez(stream,**values)
    temporary.replace(path)

def load():
    rows=read(ROOT/'manifest.json');a=np.load(ROOT/'inputs.npz')
    splits=np.array([r['split'] for r in rows]);groups=np.array([r['group'] for r in rows])
    return rows,a,splits,groups

def representation(a):
    p=np.load(ROOT/'projection.npz')
    numeric=np.concatenate([np.zeros((len(a['numeric']),10),dtype=np.float32),a['numeric']],1)
    x,num=project(a['x'],numeric,tuple(p[f'p{i}'] for i in range(5)))
    visual=np.concatenate([x.mean(1).numpy(),x[:,-1].numpy()],1)
    visual=(visual-p['vmean'])/p['vscale']
    direct=np.concatenate([visual,num[:,10:].numpy()],1)
    pool=np.concatenate([x.mean(1).numpy(),x[:,-1].numpy(),num.numpy()],1)
    return x,num,visual,direct,pool

def train_gru(c,x,num,target,device):
    torch.manual_seed(42);generator=torch.Generator().manual_seed(42)
    model=Temporal('direct',c['hidden']).to(device);x=x.to(device);num=num.to(device);target=target.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['decay'])
    trace=[]
    for epoch in range(c['epochs']):
        losses=[]
        for idx in torch.randperm(len(x),generator=generator).split(c['batch']):
            xb=x[idx];mse=(model(xb,num[idx])-target[idx]).square().mean()
            aux=contrast(model,xb,views(xb,generator));loss=mse+c['weight']*aux
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append([float(mse.detach()),float(aux.detach())])
        if epoch%30==29:trace.append({'epoch':epoch+1,'MSE':float(np.mean(losses,0)[0]),'InfoNCE':float(np.mean(losses,0)[1])})
    return model.cpu(),trace

def train():
    rows,a,splits,groups=load();tr=splits=='train';dev=splits=='dev'
    y=labels('train',rows);dy=labels('dev',rows)
    from device_benchmark import choose
    device=choose()
    fingerprint=freeze_contract(ROOT/'training_sources.json',{'script_hashes':{name:digest(ROOT/name) for name in ['learning.py','temporal.py','models.py','common.py']},
         'labels_train_sha256':digest(ROOT/'labels_train.json'),'labels_dev_sha256':digest(ROOT/'labels_dev.json'),
         'manifest_sha256':digest(ROOT/'manifest.json'),'inputs_sha256':digest(ROOT/'inputs.npz'),'runplan_sha256':digest(ROOT/'runplan.json'),
         'train_samples':int(tr.sum()),'dev_samples':int(dev.sum()),'eval_labels_accessed':False,'neural_training_device':device,
         'device_benchmark_sha256':digest(ROOT/'device_benchmark.json')})
    models=ROOT/'trained_models';models.mkdir(exist_ok=True)
    if not (ROOT/'projection.npz').exists():
        started=time.perf_counter()
        numeric=np.concatenate([np.zeros((tr.sum(),10),dtype=np.float32),a['numeric'][tr]],1)
        p=fit_projection(a['x'][tr],numeric)
        xx,_=project(a['x'][tr],numeric,p)
        v=np.concatenate([xx.mean(1).numpy(),xx[:,-1].numpy()],1)
        npz(ROOT/'projection.npz',**{f'p{i}':value for i,value in enumerate(p)},vmean=v.mean(0),vscale=np.maximum(v.std(0),1e-4))
        save(ROOT/'projection_receipt.json',{'seconds':time.perf_counter()-started,'fit_samples':int(tr.sum()),'fit_split':'train','PCA':'full float64 SVD,32 components','sha256':digest(ROOT/'projection.npz'),'training_fingerprint':fingerprint})
    projection_receipt=read(ROOT/'projection_receipt.json')
    assert projection_receipt['training_fingerprint']==fingerprint and digest(ROOT/'projection.npz')==projection_receipt['sha256']
    x,num,visual,direct,pool=representation(a)
    if (ROOT/'memory.npz').exists():
        mr=read(ROOT/'memory_receipt.json');assert mr['training_fingerprint']==fingerprint and digest(ROOT/'memory.npz')==mr['sha256']
    else:
        npz(ROOT/'memory.npz',future=y,groups=groups[tr],hold=np.repeat(a['recognition'][tr,-1:,],4,1),numeric=num[tr,10:].numpy(),visual=visual[tr])
        save(ROOT/'memory_receipt.json',{'training_fingerprint':fingerprint,'sha256':digest(ROOT/'memory.npz')})
    receipts=[]
    for c in read(ROOT/'runplan.json')['configs']:
        name=c['method'];receipt=models/(name+'.json')
        if receipt.exists():
            saved=read(receipt)
            assert saved['config']==c and saved['training_fingerprint']==fingerprint
            if saved.get('model_file'):assert digest(models/saved['model_file'])==saved['model_sha256']
            receipts.append(saved);continue
        started=time.perf_counter();family=c['family'];trace=[];model_file=None
        if family in ('numeric_ridge','meanpool','linear'):
            features=num[:,10:].numpy() if family=='numeric_ridge' or c.get('numeric_only') else pool if family=='meanpool' else direct
            if family=='numeric_ridge':
                z=features[tr];mean=z.mean(0);z=z-mean;center=y.reshape(-1,8).mean(0)
                # This newly specified local counterpart uses summed squared loss.
                coef=np.linalg.solve(z.T@z+c['penalty']*np.eye(z.shape[1]),z.T@(y.reshape(-1,8)-center))
            else:mean,center,coef=linear_fit(features[tr],y,c.get('rank',8))
            model_file=name+'.npz';npz(models/model_file,mean=mean,center=center,coef=coef)
        elif family=='gru':
            model,trace=train_gru(c,x[tr],num[tr],torch.tensor(y,dtype=torch.float32),device)
            model_file=name+'.pt';torch.save(model.state_dict(),models/model_file)
        elif family=='summary':
            features=num[:,10:].numpy() if c['numeric_only'] else direct
            model,trace=train_summary(features[tr],y,y,c['weight'],c['hidden'],c['decay'])
            model_file=name+'.pt';torch.save(model.state_dict(),models/model_file)
        elif family=='mean':
            model_file=name+'.npz';npz(models/model_file,mean=y.mean(0))
        pred=predict_one(c,a,x,num,visual,direct,pool,groups,dev)
        result={'method':name,'config':c,'training_samples':int(tr.sum()) if family!='baseline' else 0,
                'seconds':time.perf_counter()-started,'dev':metric(dy,pred),'trace':trace,'training_device':device if family=='gru' else 'cpu',
                'model_file':model_file,'model_sha256':digest(models/model_file) if model_file else None,'training_fingerprint':fingerprint}
        save(receipt,result);receipts.append(result);print(name,result['seconds'],result['dev'],flush=True)
    baseline=next(r['dev'] for r in receipts if r['method']=='hold')
    eligible=[r for r in receipts if r['dev']['MAE']<baseline['MAE'] and r['dev']['CCC_mean8']>=baseline['CCC_mean8']]
    selected=min(eligible,key=lambda r:r['dev']['MAE'])['method'] if eligible else 'hold'
    paths=[ROOT/'projection.npz',ROOT/'memory.npz',ROOT/'runplan.json',ROOT/'inputs.npz']+list(models.glob('*'))
    save(ROOT/'primary_frozen.json',{'selected':selected,'methods':receipts,'eval_GT_opened':False,'hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths}})

def predict_one(c,a,x,num,visual,direct,pool,groups,mask):
    family=c['family'];path=ROOT/'trained_models'/c['method'];hold=np.repeat(a['recognition'][mask,-1:,],4,1)
    if family=='baseline':return hold if c['method']=='hold' else a['linear'][mask]
    if family in ('numeric_ridge','meanpool','linear'):
        features=num[:,10:].numpy() if family=='numeric_ridge' or c.get('numeric_only') else pool if family=='meanpool' else direct
        m=np.load(path.with_suffix('.npz'));return ((features[mask]-m['mean'])@m['coef']+m['center']).reshape(-1,4,2)
    if family=='mean':return np.repeat(np.load(path.with_suffix('.npz'))['mean'][None],mask.sum(),0)
    if family=='gru':
        model=Temporal('direct',c['hidden']);model.load_state_dict(torch.load(str(path)+'.pt',weights_only=True));model.eval()
        with torch.no_grad():return model(x[mask],num[mask]).numpy()
    if family=='summary':
        features=num[:,10:].numpy() if c['numeric_only'] else direct
        model=SummaryMLP(features.shape[1],c['hidden']);model.load_state_dict(torch.load(str(path)+'.pt',weights_only=True))
        return neural_predict(model,features[mask])
    if family=='retrieval':
        m=np.load(ROOT/'memory.npz');q=num[:,10:].numpy()/np.sqrt(6.);keys=m['numeric']/np.sqrt(6.)
        if c['combined']:
            q=np.concatenate([q,visual/np.sqrt(64.)],1)/np.sqrt(2.);keys=np.concatenate([keys,m['visual']/np.sqrt(64.)],1)/np.sqrt(2.)
        idx,w=neighbors(q[mask],keys,groups[mask],m['groups'],c['k'],c['weighted'])
        return retrieve(m['future']-m['hold'] if c['centered'] else m['future'],idx,w)+(hold if c['centered'] else 0)
    raise ValueError(family)

def infer():
    frozen=read(ROOT/'primary_frozen.json')
    for p,h in frozen['hashes'].items():assert digest(ROOT/p)==h,p
    rows,a,splits,groups=load();x,num,visual,direct,pool=representation(a)
    directory=ROOT/'predictions';directory.mkdir(exist_ok=True);times=[]
    for c in read(ROOT/'runplan.json')['configs']:
        started=time.perf_counter();pred=predict_one(c,a,x,num,visual,direct,pool,groups,np.ones(len(rows),bool))
        assert pred.shape==(len(rows),4,2) and np.isfinite(pred).all()
        path=directory/(c['method']+'.npz');npz(path,raw=pred,prediction=np.clip(pred,-1,1))
        times.append({'method':c['method'],'samples':len(rows),'seconds':time.perf_counter()-started,'sha256':digest(path),'GT_access':False})
    save(ROOT/'primary_inference.json',times)

def score():
    rows=read(ROOT/'manifest.json');table={r['sample_id']:r for s in ['train','dev','eval'] for r in read(ROOT/f'labels_{s}.json')}
    y=np.array([table[r['sample_id']]['future'] for r in rows]);splits=np.array([r['split'] for r in rows]);results=[]
    for path in sorted((ROOT/'predictions').glob('*.npz')):
        a=np.load(path);pred=a['prediction']
        for split in ['train','dev','official_test','additional']:
            mask=splits==split;raw=metric(y[mask],a['raw'][mask],clip=False)
            result={'method':path.stem,'split':split,'coverage':float(np.isfinite(pred[mask]).all((1,2)).mean()),**metric(y[mask],pred[mask]),
                    'raw_MAE':raw['MAE'],'raw_CCC_mean8':raw['CCC_mean8'],'raw_CCC_dimensions':raw['CCC_dimensions'],
                    'clipped_fraction':float((np.abs(a['raw'][mask])>1).mean())}
            results.append(result)
    save(ROOT/'scores.json',results)
    with (ROOT/'scores.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','split','n','coverage','MAE','CCC_mean8','raw_MAE','raw_CCC_mean8','clipped_fraction'],extrasaction='ignore');writer.writeheader();writer.writerows(results)

if __name__=='__main__':
    stage=sys.argv[1];torch.set_num_threads(4);guard(stage)
    {'train':train,'infer':infer,'score':score}[stage]()
