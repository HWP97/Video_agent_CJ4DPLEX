"""기존 C의 관측 rolling selector와 damping grid를 각 fold dev에서 선택한다."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from adapter_io import ROOT,load_role,sha,install_guard
from selector_core import METHODS,select_forecast
from trajectory_baselines import forecast


def observed(batch):
    return np.matmul(batch.ego[:,:,:2],batch.rotation.transpose(0,2,1))*batch.scale[:,None,None]+batch.origin[:,None]


def weights(length,steps,dt,method):
    matrix=[]
    for i in range(length):
        history=[{'time_s':j*dt,'x':float(i==j),'y':0.} for j in range(length)]
        matrix.append([p['x'] for p in forecast(history,np.arange(length,length+steps)*dt,method)])
    return np.array(matrix)


def rolling(batch,index,steps):
    xy=observed(batch);fps=np.array([r['fps'] for r in index]);errors=np.zeros((len(xy),3))
    origins=range(max(3,9-steps-2),9-steps+1)
    for rate in np.unique(fps):
        ids=np.flatnonzero(fps==rate);parts=xy[ids]
        for column,method in enumerate(METHODS):
            totals=[]
            for origin in origins:
                pred=np.einsum('nip,ij->njp',parts[:,:origin],weights(origin,steps,1/rate,method))
                totals.append(np.linalg.norm(pred-parts[:,origin:origin+steps],axis=-1))
            errors[ids,column]=np.concatenate(totals,axis=1).mean(1)
    chosen=np.zeros(len(xy),dtype=int);best=errors[:,0].copy()
    for column in (1,2):
        take=errors[:,column]<best-1e-12;chosen[take]=column;best[take]=errors[take,column]
    options=np.stack((np.repeat(batch.origin[:,None],12,axis=1),batch.cv,batch.kalman),axis=1)
    return options[np.arange(len(xy)),chosen],chosen


def damping(batch,index,decay):
    hold=np.repeat(batch.origin[:,None],12,axis=1)
    if decay==0:return hold
    if decay==1:return batch.cv.copy()
    delta=np.arange(1,13)[None,:]/np.array([r['fps'] for r in index])[:,None];rate=-np.log(decay)
    factor=-np.expm1(-rate*delta)/(rate*delta)
    return hold+(batch.cv-hold)*factor[:,:,None]


def main():
    stage=sys.argv[1];fold=int(sys.argv[2]);directory=ROOT/'adapters/selector'/f'fold{fold}';directory.mkdir(parents=True,exist_ok=True)
    install_guard('train' if stage=='train' else 'infer',fold)
    if stage=='train':
        batch,index,y=load_role(fold,'dev',True);start=time.monotonic();scores=[]
        for kind,grid in [('selector',[1,2,3]),('damping',[0,.25,.5,.75,1])]:
            for value in grid:
                pred=rolling(batch,index,value)[0] if kind=='selector' else damping(batch,index,value)
                scores.append({'kind':kind,'value':value,'dev_ADE':float(np.linalg.norm(pred-y,axis=-1).mean())})
        choice={kind:min((r for r in scores if r['kind']==kind),key=lambda r:r['dev_ADE'])['value'] for kind in ['selector','damping']}
        receipt={'choices':choice,'grid_scores':scores,'dev_n':len(y),'fit_samples':0,'seconds':time.monotonic()-start,'runplan':sha(ROOT/'runplan.json'),'code':sha(__file__),'original_core':sha(ROOT/'code/selector_core.py'),'eval_GT_read':False}
        (directory/'selection.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
    else:
        role=stage.removeprefix('predict-');assert role in ('dev','eval')
        selection=json.loads((directory/'selection.json').read_text());assert selection['runplan']==sha(ROOT/'runplan.json') and selection['code']==sha(__file__)
        batch,index,_=load_role(fold,role);out=directory/role;out.mkdir(exist_ok=True)
        values={'hold':np.repeat(batch.origin[:,None],12,axis=1),'constant_velocity':batch.cv,'kalman':batch.kalman,
                'selector_early':rolling(batch,index,1)[0],'selector_final':rolling(batch,index,selection['choices']['selector'])[0],'damped_velocity':damping(batch,index,selection['choices']['damping'])}
        for name,pred in values.items():assert pred.shape==(len(index),12,2) and np.isfinite(pred).all();np.save(out/(name+'.npy'),pred)
        (directory/(role+'_receipt.json')).write_text(json.dumps({'n':len(index),'GT_read':False,'outputs':{name:sha(out/(name+'.npy')) for name in values}},indent=2))


if __name__=='__main__':main()
