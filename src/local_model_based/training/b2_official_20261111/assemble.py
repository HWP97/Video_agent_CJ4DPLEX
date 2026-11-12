"""정답을 열지 않고 관측 cache를 샘플별 입력으로 묶는다."""
import numpy as np
from common import ROOT,read,save,digest,guard

def main():
    guard('infer')
    assert (ROOT/'extraction_complete.json').exists()
    if (ROOT/'inputs.npz').exists() and (ROOT/'assembly_receipt.json').exists():
        receipt=read(ROOT/'assembly_receipt.json')
        assert receipt['inputs_sha256']==digest(ROOT/'inputs.npz') and receipt['manifest_sha256']==digest(ROOT/'manifest.json')
        assert receipt['feature_extraction_sha256']==digest(ROOT/'extraction_complete.json') and receipt['assembly_script_sha256']==digest(ROOT/'assemble.py')
        return
    rows=read(ROOT/'manifest.json'); videos=read(ROOT/'videos.json')
    byvideo={v['video_id']:v for v in videos}; caches={}
    visual=[]; recognition=[]; numeric=[]; linear=[]
    for row in rows:
        sha=byvideo[row['video_id']]['source_sha256']
        if sha not in caches:
            path=ROOT/'feature_cache'/f'{sha}.npz'
            assert digest(path)==read(path.with_suffix('.json'))['cache_sha256']
            with np.load(path) as stored:a={key:stored[key] for key in stored.files}
            caches[sha]=(a,{int(v):i for i,v in enumerate(a['indices'])},{str(v):i for i,v in enumerate(a['sample_ids'])})
        a,positions,samples=caches[sha]
        x=a['features'][[positions[f] for f in row['frame_indices']]]
        rec=a['recognition'][samples[row['sample_id']]].astype(np.float64)
        times=np.array(row['recognition_pts_s'])
        local=np.r_[rec[-1],(rec[-1]-rec[0])/(times[-1]-times[0]),rec.std(0)]
        design=np.column_stack([times-times[-1],np.ones(3)])
        coef=np.linalg.lstsq(design,rec,rcond=None)[0]
        forecast=np.column_stack([np.array(row['horizons_s'])+row['cutoff_s']-times[-1],np.ones(4)])@coef
        visual.append(x); recognition.append(rec); numeric.append(local);linear.append(forecast)
    with (ROOT/'inputs.npz').open('wb') as stream:
        np.savez(stream,x=np.stack(visual),recognition=np.stack(recognition),numeric=np.array(numeric,dtype=np.float32),linear=np.array(linear))
    save(ROOT/'assembly_receipt.json',{'samples':len(rows),'visual_shape':list(np.stack(visual).shape),
         'numeric_source':'raw estimated recognition last2 + endpoint slope2 + std2; no labels.current',
         'GT_read':False,'inputs_sha256':digest(ROOT/'inputs.npz'),'manifest_sha256':digest(ROOT/'manifest.json'),
         'feature_extraction_sha256':digest(ROOT/'extraction_complete.json'),'assembly_script_sha256':digest(ROOT/'assemble.py')})

if __name__=='__main__':main()
