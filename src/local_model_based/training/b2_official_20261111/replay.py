"""가중치를 바꾸지 않고55개 전체 추론과 별도 채점을 새 하위폴더에 기록한다."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,read,save,digest,guard
from learning import load,representation,predict_one,npz,metric
from supplement import Context,contexts
from chronos_run import ASSET,summaries

def infer(destination):
    assert not destination.exists(),destination
    destination.mkdir(parents=True)
    frozen=read(ROOT/'all_frozen.json')
    for name,h in frozen['hashes'].items():assert digest(ROOT/name)==h,name
    started=time.perf_counter();rows,a,splits,groups=load();x,num,visual,direct,pool=representation(a);outputs={}
    for c in read(ROOT/'runplan.json')['configs']:
        outputs[c['method']]=predict_one(c,a,x,num,visual,direct,pool,groups,np.ones(len(rows),bool))
    xx=contexts(rows,x)
    for mode in read(ROOT/'supplement_runplan.json')['context_modes']:
        model=Context(mode);model.load_state_dict(torch.load(ROOT/'trained_models'/f'context_{mode}.pt',map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():outputs['context_'+mode]=model(xx).numpy()
    m=np.load(ROOT/'trained_models/numeric_meanloss_1.npz');outputs['numeric_meanloss_1']=((num[:,10:].numpy()-m['mean'])@m['coef']+m['center']).reshape(-1,4,2)
    current=a['recognition'][:,-1:,];future=outputs['localpool_ridge']
    for alpha in read(ROOT/'supplement_runplan.json')['alpha']:
        outputs[f'local_alpha_{alpha:g}']=np.repeat(current,4,1) if alpha==0 else future if alpha==1 else current+alpha*(future-current)
    offset=np.array(read(ROOT/'offset_model.json')['offset'])
    for name in read(ROOT/'supplement_runplan.json')['offset_forecasts']:outputs['offset_'+name]=outputs[name]+offset
    dense=np.load(ROOT/'dense_inputs.npz')
    for key in ['short','long']:
        features=summaries(dense[key]);m=np.load(ROOT/'trained_models'/f'summary_ridge_{key}.npz')
        outputs[f'summary_ridge_{key}']=(((features-m['mean'])/m['scale'])@m['coef']+m['center']).reshape(-1,4,2)
        t=np.arange(dense[key].shape[1])/10;t-=t.mean()
        outputs[f'linear_{key}']=dense[key].mean(1)[:,None,:]+features[:,-2:][:,None,:]*(np.array([.5,1,2,3])-t[0])[None,:,None]
    sys.path.insert(0,str(ASSET));from adapter import load_pipeline,forecast
    model=load_pipeline()
    for c in read(ROOT/'adapter_runplan.json')['chronos_configs']:
        outputs[c['method']]=forecast(model,dense['short' if c['length_s']==1 else 'long'],30,c['joint'])[:,[4,9,19,29]]
    assert len(outputs)==55
    differences={}
    for name,raw in outputs.items():
        assert raw.shape==(2740,4,2) and np.isfinite(raw).all()
        p=np.clip(raw,-1,1);npz(destination/(name+'.npz'),raw=raw,prediction=p)
        old=np.load(ROOT/'predictions'/(name+'.npz'))['prediction'];differences[name]=float(np.max(np.abs(old-p)))
    save(destination/'inference.json',{'methods':55,'samples_each':2740,'seconds':time.perf_counter()-started,'training_calls':0,'GT_label_access':False,
        'selected_unchanged':frozen['selected'],'torch':torch.__version__,'max_abs_diff_vs_original':differences,'all_frozen_sha256':digest(ROOT/'all_frozen.json')})
    print('fresh inference55x2740, no training, output='+str(destination),flush=True)

def score(destination):
    rows=read(ROOT/'manifest.json');table={r['sample_id']:r for s in ['train','dev','eval'] for r in read(ROOT/f'labels_{s}.json')}
    truth=np.array([table[r['sample_id']]['future'] for r in rows]);splits=np.array([r['split'] for r in rows]);results=[]
    assert len(list(destination.glob('*.npz')))==55
    for path in sorted(destination.glob('*.npz')):
        a=np.load(path)
        for split in ['train','dev','official_test','additional']:
            mask=splits==split;results.append({'method':path.stem,'split':split,'coverage':1.,**metric(truth[mask],a['prediction'][mask])})
    save(destination/'scores.json',results);print('independent replay scoring220rows',flush=True)

if __name__=='__main__':
    stage=sys.argv[1];destination=(ROOT/'replays'/sys.argv[2]).resolve();assert destination.is_relative_to(ROOT/'replays')
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;guard(stage)
    {'infer':infer,'score':score}[stage](destination)
