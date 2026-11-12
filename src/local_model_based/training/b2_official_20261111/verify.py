"""실제 manifest·학습 통계·동결 모델과 실패해야 하는 경계를 검증한다."""
import ast
import json
import subprocess
import sys
import numpy as np
import torch
from common import ROOT,SOURCE,read,save,digest,freeze_contract
from temporal import Temporal

def main():
    checks=[];rows=read(ROOT/'manifest.json');plan=read(ROOT/'runplan.json');labels={r['sample_id']:r for s in ['train','dev','eval'] for r in read(ROOT/f'labels_{s}.json')}
    assert len(rows)==2740 and len(labels)==len(rows) and len({r['sample_id'] for r in rows})==len(rows)
    for r in rows:
        target=labels[r['sample_id']];assert len(r['frame_indices'])==15
        assert min(r['frame_indices'])>=r['partition_start_frame'] and max(target['target_frames'])<r['partition_end_frame_exclusive']
        assert max(r['frame_pts_s'])<=r['cutoff_s']+1e-9 and min(target['target_pts_s'])>r['cutoff_s']
        assert max(abs(t-r['cutoff_s']-h) for t,h in zip(target['target_pts_s'],r['horizons_s']))<=.020001
        if r['split'] in ('train','dev'):assert r['group'] in plan[r['split']+'_groups']
    assert not set(plan['train_groups'])&set(plan['dev_groups'])
    checks.append({'scenario':'all2740 causal partition and group boundaries','passed':True})
    for stage,path in [('infer',ROOT/'labels_train.json'),('infer',ROOT/'labels_eval.json'),('extract',SOURCE/'data/VEATIC/rating_averaged/0_valence.csv'),('train',ROOT/'labels_eval.json')]:
        code=f"from common import guard;guard({stage!r});open({str(path)!r}).read()"
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode!=0 and 'PermissionError' in result.stderr
        checks.append({'scenario':stage+' rejects '+str(path),'returncode':result.returncode,'stderr':result.stderr,'passed':True})
    for code in ["import socket;socket.create_connection(('127.0.0.1',9))",f"open({str(ROOT.parent/'outside_probe.txt')!r},'w')",f"open('/mnt/RG2/api_based/probe.json')"]:
        result=subprocess.run([sys.executable,'-B','-c',"from common import guard;guard('infer');"+code],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode!=0 and 'PermissionError' in result.stderr
        checks.append({'scenario':code,'returncode':result.returncode,'stderr':result.stderr,'passed':True})
    for path in ROOT.glob('*.py'):ast.parse(path.read_text())
    checks.append({'scenario':'all owned Python files parse','passed':True})
    probe=ROOT/'runtime/resume_probe.json'
    if probe.exists():probe.unlink()
    contract={'input_sha':digest(ROOT/'manifest.json'),'config':{'penalty':1}}
    freeze_contract(probe,contract);freeze_contract(probe,contract)
    try:freeze_contract(probe,contract|{'config':{'penalty':10}})
    except ValueError as error:checks.append({'scenario':'changed resume config rejected while identical contract resumes','passed':True,'error':str(error)})
    else:raise AssertionError('changed resume config accepted')
    if (ROOT/'projection.npz').exists():
        a=np.load(ROOT/'inputs.npz');tr=np.array([r['split']=='train' for r in rows]);flat=a['x'][tr].reshape(-1,2048).astype(np.float64);p=np.load(ROOT/'projection.npz')
        assert np.allclose(flat.mean(0),p['p0'],atol=1e-12) and np.allclose(np.maximum(flat.std(0),1e-4),p['p1'],atol=1e-12)
        assert np.all(p['p3'][:10]==0) and np.all(p['p4'][:10]==1e-4)
        numeric=np.concatenate([np.zeros((int(tr.sum()),10),dtype=np.float32),a['numeric'][tr]],1)
        assert np.allclose(numeric.mean(0),p['p3']) and np.allclose(np.maximum(numeric.std(0),1e-4),p['p4'])
        z=(flat-p['p0'])/p['p1'];basis=p['p2'];projected=z@basis;variance=(projected**2).mean(0)
        assert np.allclose(basis.T@basis,np.eye(32),atol=1e-10)
        relative=float(np.linalg.norm(z.T@projected/len(z)-basis*variance)/np.linalg.norm(basis*variance))
        assert relative<1e-8 and np.all(np.diff(variance)<1e-8)
        checks.append({'scenario':'train means/scales recomputed; stored PCA basis orthogonality and covariance eigen residual checked (full SVD not rerun)',
                       'passed':True,'train_samples':int(tr.sum()),'eigen_relative_residual':relative})
    if (ROOT/'primary_frozen.json').exists():
        model=Temporal('direct',32);model.load_state_dict(torch.load(ROOT/'trained_models/direct_gru_h32_contrast0.pt',weights_only=True));model.eval()
        x=torch.randn(3,15,32);numeric=torch.randn(3,16);other=numeric.clone();other[:,:10]+=1000
        with torch.no_grad():assert torch.equal(model(x,numeric),model(x,other))
        checks.append({'scenario':'actual trained direct GRU ignores API placeholder columns exactly','passed':True})
    if (ROOT/'all_frozen.json').exists():
        frozen=read(ROOT/'all_frozen.json')
        for path,h in frozen['hashes'].items():assert digest(ROOT/path)==h,path
        assert len(list((ROOT/'predictions').glob('*.npz')))==55
        dev=np.array([r['split']=='dev' for r in rows]);truth=np.array([labels[r['sample_id']]['future'] for r in rows if r['split']=='dev']).reshape(-1,8);development={}
        for path in (ROOT/'predictions').glob('*.npz'):
            a=np.load(path);assert a['prediction'].shape==(2740,4,2) and np.isfinite(a['prediction']).all()
            pred=a['prediction'][dev].reshape(-1,8)
            ccc=2*((truth-truth.mean(0))*(pred-pred.mean(0))).mean(0)/np.maximum(truth.var(0)+pred.var(0)+(truth.mean(0)-pred.mean(0))**2,1e-15)
            development[path.stem]=(float(np.abs(truth-pred).mean()),float(ccc.mean()))
        reference=development['hold'];offset_gate=read(ROOT/'offset_model.json')['eligible']
        eligible=[name for name,(mae,ccc) in development.items() if mae<reference[0] and ccc>=reference[1] and (offset_gate or not name.startswith('offset_'))]
        selected=min(eligible,key=lambda name:development[name][0]) if eligible else 'hold'
        assert selected==frozen['selected']
        checks.append({'scenario':'all frozen models/predictions hashes and full2740 coverage','passed':True,'methods':len(list((ROOT/'predictions').glob('*.npz')))})
        checks.append({'scenario':'independent dev-only MAE/CCC selection recomputed over all55; offset OOF gate retained','passed':True,'selected':selected})
    if (ROOT/'equivalence.json').exists():
        assert read(ROOT/'equivalence.json')['passed'];checks.append({'scenario':'actual RGB direct vs cached recognizer','passed':True,'receipt':'equivalence.json'})
    if (ROOT/'departure_scores.json').exists():
        scores=read(ROOT/'departure_scores.json');model=read(ROOT/'departure_model.json');p=np.load(ROOT/'departure_predictions.npz')['probability']
        assert p.shape==(2740,4) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        for split in ['dev','official_test','additional']:
            for coverage in [.5,.7,.9,1.]:
                same=[r for r in scores if r['split']==split and r['requested_coverage']==coverage and r['method'] in ['confidence','hash_control']]
                assert len(same)==2 and same[0]['committed']==same[1]['committed']==int(np.ceil(coverage*same[0]['n']))
        for curve in model['crc_curves'].values():
            losses=np.array(curve['raw_group_losses']);envelope=np.maximum.accumulate(losses[:,::-1],axis=1)[:,::-1]
            assert np.allclose(envelope,curve['envelope_group_losses']) and np.allclose((envelope.sum(0)+1)/(len(envelope)+1),curve['corrected_bound'])
        assert all('alarm_recall' in r for r in scores)
        checks.append({'scenario':'departure finite probabilities, same-K coverage, group suffix envelope and alarm recall','passed':True})
    save(ROOT/'verification.json',{'checks':checks,'passed':True,'command':'existing Python -B verify.py'})
    print(json.dumps({'checks':len(checks),'passed':True}),flush=True)

if __name__=='__main__':main()
