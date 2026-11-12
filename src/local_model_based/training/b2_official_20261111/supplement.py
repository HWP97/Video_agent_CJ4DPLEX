"""공식 train의 작은 맥락 head와 로컬 후처리: 원논문 재현과 구별하는 대응 실험."""
import sys
import time
import numpy as np
import torch
from torch import nn
from common import ROOT,read,save,digest,guard,freeze_contract
from learning import load,representation,npz,metric,labels

MODES=['current','mean','concat','concat_budget','lstm_attention']

class Context(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.projection=nn.Linear(32,32)
        self.lstm=nn.LSTM(32,32,batch_first=True) if mode=='lstm_attention' else nn.Identity()
        self.attention=nn.MultiheadAttention(32,4,batch_first=True) if mode=='lstm_attention' else nn.Identity()
        self.head=nn.Sequential(nn.Linear(96,124),nn.Tanh(),nn.Linear(124,8)) if mode=='concat_budget' else nn.Linear(96 if mode=='concat' else 32,8)
    def forward(self,x):
        z=torch.tanh(self.projection(x))
        if self.mode=='current':z=z[:,2]
        elif self.mode=='mean':z=z.mean(1)
        elif self.mode in ('concat','concat_budget'):z=z.flatten(1)
        else:
            z,_=self.lstm(z);z,_=self.attention(z,z,z,need_weights=False);z=z.mean(1)
        return self.head(z).reshape(-1,4,2)

def freeze():
    config={'context_modes':MODES,'context_input':'same15frames grouped by original three recognition sequences, each5features mean after train-only PCA32',
            'context_targets':'future VA regression8, not MCIS emotion classes','context_seed':20261109,'context_epochs':100,'optimizer':'Adam','lr':.001,'weight_decay':.0001,
            'concat_budget_hidden':124,'capacity_note':'124 copied from original 7class experiment; output8 means parameter equality no longer exact; both counts reported',
            'alpha':[0,.25,.5,.75,1],'alpha_anchor':'raw recognizer last VA','alpha_future':'localpool_ridge','offset':'median of original-group median currentGT minus recognition; train only',
            'offset_forecasts':['hold','linear_trend','localpool_ridge'],'offset_gate':'train leave-one-group-out current raw MAE lower AND both axis CCC nondecrease',
            'CRC_C':[.1,1,10],'CRC_alpha':[.1,.2,.3],'CRC_calibration':'dev20groups','CRC_classifier_select':'train grouped3fold OOF AUPRC',
            'selection':'dev only; MAE lower and CCC nondecrease vs hold','API_calls':0}
    path=ROOT/'supplement_runplan.json'
    if path.exists():assert read(path)==config
    else:save(path,config)
    objective={'numeric_ridge_1/10/100':{'objective':'sum squared error + lambda*L2','effective_mean_loss_lambda':[v/1169 for v in [1,10,100]],'classification':'new local numeric counterpart'},
               'numeric_meanloss_1':{'objective':'mean squared error + 1*L2','effective_mean_loss_lambda':1,'classification':'separate normalization control frozen before first fit'},
               'localpool_ridge/direct_linear':{'objective':'mean squared error + 1*L2','effective_mean_loss_lambda':1,'classification':'existing local recipe'}}
    path=ROOT/'objective_runplan.json'
    if path.exists():assert read(path)==objective
    else:save(path,objective)

def contexts(rows,x):
    return torch.stack([torch.stack([x[i,[r['frame_indices'].index(f) for f in seq]].mean(0) for seq in r['recognition_sequences']]) for i,r in enumerate(rows)])

def current_metrics(y,p):
    c=2*((y-y.mean(0))*(p-p.mean(0))).mean(0)/np.maximum(y.var(0)+p.var(0)+(y.mean(0)-p.mean(0))**2,1e-15)
    return {'MAE':float(np.abs(y-p).mean()),'CCC_axes':c.tolist()}

def train():
    freeze();rows,a,splits,groups=load();tr=splits=='train';dev=splits=='dev'
    fingerprint=freeze_contract(ROOT/'supplement_contract.json',{'hashes':{name:digest(ROOT/name) for name in
        ['supplement.py','models.py','common.py','projection.npz','inputs.npz','labels_train.json','labels_dev.json','supplement_runplan.json','objective_runplan.json','device_benchmark.json']}})
    x,num,*_=representation(a);xx=contexts(rows,x);y=labels('train',rows);dy=labels('dev',rows);results=[]
    device=read(ROOT/'device_benchmark.json')['selected'];torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    train_x=xx[tr].to(device);dev_x=xx[dev].to(device)
    numeric_started=time.perf_counter()
    numeric=num[:,10:].numpy();mean=numeric[tr].mean(0);z=numeric[tr]-mean;center=y.reshape(-1,8).mean(0)
    coef=np.linalg.solve(z.T@z/len(z)+np.eye(6),z.T@(y.reshape(-1,8)-center)/len(z))
    npz(ROOT/'trained_models/numeric_meanloss_1.npz',mean=mean,center=center,coef=coef)
    pred=((numeric[dev]-mean)@coef+center).reshape(-1,4,2)
    results.append({'method':'numeric_meanloss_1','dev':metric(dy,pred),'training_samples':int(tr.sum()),'objective':'meanSSE+1L2','seconds':time.perf_counter()-numeric_started})
    for mode in MODES:
        receipt=ROOT/'trained_models'/f'context_{mode}.json';path=receipt.with_suffix('.pt')
        if receipt.exists():
            result=read(receipt);assert digest(path)==result['sha256'] and result['training_fingerprint']==fingerprint;results.append(result);continue
        started=time.perf_counter();torch.manual_seed(20261109);model=Context(mode).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);yy=torch.tensor(y,dtype=torch.float32,device=device);trace=[]
        for epoch in range(100):
            model.train();opt.zero_grad();loss=(model(train_x)-yy).square().mean();assert torch.isfinite(loss);loss.backward();opt.step()
            if epoch%20==19:trace.append({'epoch':epoch+1,'MSE':float(loss.detach())})
        model.eval()
        with torch.no_grad():p=model(dev_x).cpu().numpy()
        model.cpu()
        torch.save(model.state_dict(),path)
        result={'method':'context_'+mode,'dev':metric(dy,p),'seconds':time.perf_counter()-started,'trace':trace,'parameters':sum(v.numel() for v in model.parameters()),'sha256':digest(path),'training_samples':int(tr.sum()),'training_device':device,'training_fingerprint':fingerprint}
        save(receipt,result);results.append(result);print(result['method'],result['seconds'],result['dev'],flush=True)
    offset_started=time.perf_counter();table={r['sample_id']:r for r in read(ROOT/'labels_train.json')}
    current=np.array([table[r['sample_id']]['current'] for r in rows if r['split']=='train']);estimated=a['recognition'][tr,-1];tg=groups[tr]
    residual=current-estimated;unique=np.unique(tg);medians=np.array([np.median(residual[tg==g],0) for g in unique]);offset=np.median(medians,0)
    oof=estimated.copy()
    for i,g in enumerate(unique):oof[tg==g]+=np.median(np.delete(medians,i,0),0)
    before=current_metrics(current,estimated);after=current_metrics(current,oof)
    eligible=after['MAE']<before['MAE'] and np.all(np.array(after['CCC_axes'])>=np.array(before['CCC_axes']))
    save(ROOT/'offset_model.json',{'offset':offset.tolist(),'train_groups':len(unique),'fit_samples':len(current),'OOF_raw_before':before,'OOF_raw_after':after,'eligible':bool(eligible),
         'meaning':'current GT used only as training target; no future labels used to fit offset','seconds':time.perf_counter()-offset_started})
    save(ROOT/'supplement_training.json',results)

def infer():
    rows,a,splits,groups=load();x,num,*_=representation(a);xx=contexts(rows,x);directory=ROOT/'predictions';times=[]
    m=np.load(ROOT/'trained_models/numeric_meanloss_1.npz');raw=((num[:,10:].numpy()-m['mean'])@m['coef']+m['center']).reshape(-1,4,2)
    npz(directory/'numeric_meanloss_1.npz',raw=raw,prediction=np.clip(raw,-1,1))
    for mode in MODES:
        started=time.perf_counter();model=Context(mode);path=ROOT/'trained_models'/f'context_{mode}.pt'
        assert digest(path)==read(path.with_suffix('.json'))['sha256']
        model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        with torch.no_grad():raw=model(xx).numpy()
        npz(directory/f'context_{mode}.npz',raw=raw,prediction=np.clip(raw,-1,1));times.append({'method':'context_'+mode,'samples':len(raw),'seconds':time.perf_counter()-started})
    current=a['recognition'][:,-1:,];future=np.load(directory/'localpool_ridge.npz')['raw']
    for alpha in read(ROOT/'supplement_runplan.json')['alpha']:
        raw=np.repeat(current,4,1) if alpha==0 else future if alpha==1 else current+alpha*(future-current)
        npz(directory/f'local_alpha_{alpha:g}.npz',raw=raw,prediction=np.clip(raw,-1,1))
    offset=np.array(read(ROOT/'offset_model.json')['offset'])
    for name in read(ROOT/'supplement_runplan.json')['offset_forecasts']:
        raw=np.load(directory/(name+'.npz'))['raw']+offset
        npz(directory/f'offset_{name}.npz',raw=raw,prediction=np.clip(raw,-1,1))
    save(ROOT/'supplement_inference.json',{'context':times,'postprocessing_samples_each':len(rows),'GT_access':False,'offset_results':'diagnostic regardless of OOF gate; gate controls adoption eligibility'})

def select():
    rows=read(ROOT/'manifest.json');dev=np.array([r['split']=='dev' for r in rows]);y=labels('dev',rows)
    results=[];offset_eligible=read(ROOT/'offset_model.json')['eligible']
    for path in sorted((ROOT/'predictions').glob('*.npz')):
        p=np.load(path)['prediction'];results.append({'method':path.stem,'dev':metric(y,p[dev]),'offset_gate':offset_eligible if path.stem.startswith('offset_') else True})
    ref=next(r['dev'] for r in results if r['method']=='hold')
    eligible=[r for r in results if r['dev']['MAE']<ref['MAE'] and r['dev']['CCC_mean8']>=ref['CCC_mean8'] and r['offset_gate']]
    selected=min(eligible,key=lambda r:r['dev']['MAE'])['method'] if eligible else 'hold'
    save(ROOT/'all_frozen.json',{'selected':selected,'dev_results':results,'eval_GT_opened':False,'offset_OOF_gate':offset_eligible,
        'hashes':{str(p.relative_to(ROOT)):digest(p) for p in list((ROOT/'predictions').glob('*.npz'))+list((ROOT/'trained_models').glob('*'))+
            [ROOT/name for name in ['manifest.json','runplan.json','adapter_runplan.json','supplement_runplan.json','objective_runplan.json','offset_model.json','projection.npz','memory.npz','training_sources.json','supplement_contract.json','device_benchmark.json']]},
        'selection':'dev MAE decrease AND CCC nondecrease vs hold; optional offset current OOF gate'})

if __name__=='__main__':
    stage=sys.argv[1];torch.set_num_threads(4);guard('infer' if stage=='infer' else 'train')
    {'freeze':freeze,'train':train,'infer':infer,'select':select}[stage]()
