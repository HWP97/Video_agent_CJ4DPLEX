"""기존 Chronos의 10Hz 관측 입력만 별도 추출한다. 기본15프레임 입력은 유지한다."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2
from extract import official,preprocess
from common import ROOT,SOURCE,read,save,digest,guard
from feature_guard import contract

def main():
    guard('extract');torch.set_num_threads(2);torch.manual_seed(42);cv2.setNumThreads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    contract()
    start=time.perf_counter();original=official.resnet50;official.resnet50=lambda **kwargs:original(weights=None)
    model=official.VEATIC_baseline();official.resnet50=original
    model.load_state_dict(torch.load(SOURCE/'data/one_stream_vit.pth',map_location='cpu',weights_only=True));model.eval().cuda()
    rows=read(ROOT/'manifest.json');videos=read(ROOT/'videos.json');metadata=read(SOURCE/'data_manifest.json')
    directory=ROOT/'dense_cache';directory.mkdir(exist_ok=True);groups={}
    for v in videos:groups.setdefault(v['source_sha256'],[]).append(v)
    receipts=[]
    for sha,versions in groups.items():
        selected=[r for r in rows if r['video_id'] in {v['video_id'] for v in versions}]
        if not selected:continue
        path=directory/(sha+'.npz');receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists():
            item=read(receipt);assert digest(path)==item['sha256'];receipts.append(item);continue
        started=time.perf_counter();pts=np.array(metadata[versions[0]['video_id']]['frame_pts_s'])
        endpoints=[]
        for r in selected:
            grid=r['cutoff_s']-np.arange(72,-1,-1)/10
            end=np.searchsorted(pts,grid+1e-9,side='right')-1
            assert np.all(pts[end]<=grid+1e-8) and end.min()-20>=r['partition_start_frame']
            assert pts[end.min()-20]>=r['observed_start_s']-1e-8
            endpoints.append(end)
        endpoints=np.stack(endpoints);ends=np.unique(endpoints)
        support=ends[:,None]+np.array([-20,-15,-10,-5,0]);needed=np.unique(support)
        basepath=ROOT/'feature_cache'/f'{sha}.npz';features={}
        if basepath.exists() and basepath.with_suffix('.json').exists():
            assert digest(basepath)==read(basepath.with_suffix('.json'))['cache_sha256']
            with np.load(basepath) as base:features={int(i):f for i,f in zip(base['indices'],base['features'])}
        missing=set(needed)-set(features);capture=cv2.VideoCapture(versions[0]['source']);pixels=[];indices=[]
        executor=ThreadPoolExecutor(max_workers=4)
        try:
            for i in range(int(needed[-1])+1):
                okay,bgr=capture.read();assert okay
                if i not in missing:continue
                pixels.append(bgr);indices.append(i)
                if len(pixels)==8:
                    prepared=np.stack(list(executor.map(preprocess,pixels)))
                    with torch.inference_mode():
                        out=model.pool(model.human(model.feature(torch.from_numpy(prepared).cuda()))).flatten(1).cpu().numpy()
                    features.update(zip(indices,out));pixels.clear();indices.clear()
            if pixels:
                prepared=np.stack(list(executor.map(preprocess,pixels)))
                with torch.inference_mode():out=model.pool(model.human(model.feature(torch.from_numpy(prepared).cuda()))).flatten(1).cpu().numpy()
                features.update(zip(indices,out))
        finally:capture.release();executor.shutdown()
        predictions=[]
        with torch.inference_mode():
            for i in range(0,len(support),64):
                x=np.array([[features[int(f)] for f in seq] for seq in support[i:i+64]])
                predictions.append(model.ViT(torch.from_numpy(x).cuda()).cpu().numpy())
        allrec=np.concatenate(predictions);index=np.searchsorted(ends,endpoints);long=allrec[index];short=long[:,-11:]
        assert np.isfinite(long).all()
        with path.with_suffix('.tmp').open('wb') as stream:np.savez(stream,long=long,short=short,sample_ids=np.array([r['sample_id'] for r in selected]),endpoints=endpoints)
        path.with_suffix('.tmp').replace(path)
        item={'video_ids':[v['video_id'] for v in versions],'samples':len(selected),'new_features':len(missing),'reused_features':len(needed)-len(missing),
              'seconds':time.perf_counter()-started,'sha256':digest(path),'GT_access':False,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'precision':'FP32 TF32 disabled','preprocessing_threads':4,'base_cache_read_only':True}
        save(receipt,item);receipts.append(item)
        progress={'groups_done':len(receipts),'groups_total':110,'seconds':time.perf_counter()-start,'last_video':item}
        save(ROOT/'dense_progress.json',progress);print(json.dumps(progress),flush=True)
    mapping={}
    for path in directory.glob('*.npz'):
        a=np.load(path)
        mapping.update({str(s):(a['short'][i],a['long'][i]) for i,s in enumerate(a['sample_ids'])})
    with (ROOT/'dense_inputs.npz').open('wb') as stream:
        np.savez(stream,short=np.stack([mapping[r['sample_id']][0] for r in rows]),long=np.stack([mapping[r['sample_id']][1] for r in rows]))
    save(ROOT/'dense_complete.json',{'receipts':receipts,'samples':len(rows),'sha256':digest(ROOT/'dense_inputs.npz'),'wall_s':time.perf_counter()-start})

if __name__=='__main__':main()
