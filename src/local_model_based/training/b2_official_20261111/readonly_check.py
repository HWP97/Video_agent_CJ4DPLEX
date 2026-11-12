"""파일을 쓰지 않고55개 전체 예측·재현·평가110행·동결SHA를 독립 확인한다."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8388608),b''):h.update(chunk)
    return h.hexdigest()

def main():
    frozen=read(ROOT/'all_frozen.json');source=read(ROOT/'source_manifest.json')
    for name,expected in frozen['hashes'].items():assert sha(ROOT/name)==expected,name
    for name,expected in source['code_snapshot_sha256'].items():assert sha(Path(name))==expected,name
    manifest=read(ROOT/'manifest.json');labels={r['sample_id']:r for r in read(ROOT/'labels_eval.json')}
    scores={(r['method'],r['split']):r for r in read(ROOT/'scores.json')};files=sorted((ROOT/'predictions').glob('*.npz'))
    assert len(files)==55 and len(scores)==220 and len(manifest)==2740
    replay=ROOT/'replays/verified_replay';maxdiff=0.;evaluated=0
    for path in files:
        p=np.load(path)['prediction'];q=np.load(replay/path.name)['prediction'];assert p.shape==q.shape==(2740,4,2)
        assert np.isfinite(p).all() and np.isfinite(q).all() and np.array_equal(p,q)
        for split in ['official_test','additional']:
            mask=np.array([r['split']==split for r in manifest]);y=np.array([labels[r['sample_id']]['future'] for r in manifest if r['split']==split]).reshape(-1,8)
            pred=p[mask].reshape(-1,8);mae=float(np.abs(y-pred).mean())
            ccc=float((2*((y-y.mean(0))*(pred-pred.mean(0))).mean(0)/np.maximum(y.var(0)+pred.var(0)+(y.mean(0)-pred.mean(0))**2,1e-15)).mean())
            stored=scores[path.stem,split];assert stored['n']==int(mask.sum()) and stored['coverage']==1
            difference=max(abs(stored['MAE']-mae),abs(stored['CCC_mean8']-ccc));maxdiff=max(maxdiff,difference);assert difference<1e-12;evaluated+=1
    for item in read(ROOT/'preservation_verification.json')['initial_frozen_inputs']:assert sha(Path(item['path']))==sha(Path(item['initial_copy']))==item['sha256']
    assert read(ROOT/'verification.json')['passed'] and read(ROOT/'replay_verification.json')['passed']
    print(json.dumps({'PASS':True,'methods':55,'predictions':55*2740,'independently_rescored_test_rows':evaluated,'max_score_difference':maxdiff,
                      'replay_prediction_difference':0,'selected':frozen['selected'],'writes':0}),flush=True)

if __name__=='__main__':main()
