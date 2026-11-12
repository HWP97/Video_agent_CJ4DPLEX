"""공식70% 경계 안에서 head train/dev를 고정하며 평가 정답을 분리한다."""
import hashlib
import json
from bisect import bisect_left,bisect_right
from collections import Counter

import numpy as np

from common import ROOT,SOURCE,digest,read,save

def main():
    metadata=read(SOURCE/'data_manifest.json')
    groups=sorted({v['original_group'] for v in metadata[:98]},key=lambda g:hashlib.sha256(('local-b2-official-train-v1:'+g).encode()).hexdigest())
    assert len(groups)==98
    dev=set(groups[:20])
    samples=[]
    labels={'train':[],'dev':[],'eval':[]}
    videos=[]
    exclusions=[]
    for info in metadata:
        vid=info['video_id'];pts=info['frame_pts_s'];n=info['frame_count'];boundary=int(.7*n)
        ratings=np.stack([np.loadtxt(SOURCE/f'data/VEATIC/rating_averaged/{vid}_{axis}.csv',delimiter=',')[:,1] for axis in ['valence','arousal']],axis=1)
        assert ratings.shape==(n,2) and np.isfinite(ratings).all()
        segments=[('dev' if info['original_group'] in dev else 'train',0,boundary),('official_test',boundary,n)] if vid<98 else [('additional',0,n)]
        for split,start,end in segments:
            cutoff=pts[start]+8.
            count=0
            while cutoff+3.<=pts[end-1]+1e-9:
                sequences=[]
                for relative in [-1.,-.5,0.]:
                    last=bisect_right(pts,cutoff+relative+1e-9)-1
                    seq=list(range(last-20,last+1,5))
                    assert seq[0]>=start and pts[seq[0]]>=cutoff-8.-1e-9 and pts[seq[-1]]<=cutoff+relative+1e-9
                    sequences.append(seq)
                frames=sorted({i for seq in sequences for i in seq})
                assert len(frames)==15 and pts[frames[-1]]<=cutoff+1e-9
                targets=[]
                for horizon in [.5,1.,2.,3.]:
                    near=bisect_left(pts,cutoff+horizon)
                    target=min([i for i in [near-1,near] if start<=i<end],key=lambda i:abs(pts[i]-cutoff-horizon))
                    assert abs(pts[target]-cutoff-horizon)<=.020001
                    targets.append(target)
                sid=f'v{vid:03d}_{split}_t{cutoff:010.4f}'
                row={'sample_id':sid,'video_id':vid,'group':info['original_group'],'split':split,'cutoff_s':cutoff,
                     'observed_start_s':cutoff-8.,'frame_indices':frames,'frame_pts_s':[pts[i] for i in frames],
                     'recognition_sequences':sequences,'recognition_pts_s':[pts[s[-1]] for s in sequences],
                     'partition_start_frame':start,'partition_end_frame_exclusive':end,'horizons_s':[.5,1.,2.,3.]}
                samples.append(row)
                labels[split if split in ('train','dev') else 'eval'].append({'sample_id':sid,'split':split,'future':ratings[targets].tolist(),
                    'current':ratings[sequences[-1][-1]].tolist(),'target_frames':targets,'target_pts_s':[pts[i] for i in targets]})
                count+=1;cutoff+=3.
            if count==0:
                exclusions.append({'video_id':vid,'split':split,'reason':'partition_shorter_than_8s_observation_plus_3s_future'})
        needed=sorted({f for row in samples if row['video_id']==vid for f in row['frame_indices']})
        videos.append({'video_id':vid,'source':str(SOURCE/info['video_path']),'source_sha256':info['sha256'],
                       'frame_count':n,'needed_frames':needed,'group':info['original_group']})
    configs=[{'method':'hold','family':'baseline'},{'method':'linear_trend','family':'baseline'}]
    configs += [{'method':f'numeric_ridge_{penalty}','family':'numeric_ridge','penalty':penalty} for penalty in [1,10,100]]
    configs += [{'method':'localpool_ridge','family':'meanpool','penalty':1}]
    configs += [{'method':f'direct_gru_h{hidden}_contrast{weight:g}','family':'gru','hidden':hidden,'weight':weight,'epochs':120,'lr':.001,'decay':.01,'batch':64} for hidden,weights in [(32,[0.,.03,.1,.3]),(8,[0.,.1])] for weight in weights]
    configs += [{'method':'direct_mean','family':'mean'}]
    configs += [{'method':f'retrieval_k{k}_visual{combined}_weighted{weighted}','family':'retrieval','k':k,'combined':combined,'weighted':weighted,'centered':False} for k in [5,20] for combined in [False,True] for weighted in [False,True]]
    configs += [{'method':'retrieval_k5_centered','family':'retrieval','k':5,'combined':False,'weighted':True,'centered':True}, {'method':'retrieval_k60','family':'retrieval','k':60,'combined':False,'weighted':True,'centered':False}]
    configs += [{'method':f'summary_h{hidden}_numeric{numeric}_weight{weight:g}','family':'summary','hidden':hidden,'numeric_only':numeric,'weight':weight,'epochs':200,'lr':.003,'decay':decay} for hidden,numeric,decay,weights in [(16,False,.05,[0.,.1,1.]),(8,True,.2,[0.,.03,.1])] for weight in weights]
    configs += [{'method':f'direct_linear_rank{rank}','family':'linear','rank':rank,'numeric_only':False} for rank in [2,4,8]]
    configs += [{'method':'direct_linear_numeric_rank8','family':'linear','rank':8,'numeric_only':True}]
    assert len(configs)==33
    plan={'seed':42,'configs':configs,'official_boundary':'first98 floor(0.7*frame_count)','train_groups':groups[20:],'dev_groups':groups[:20],
          'observation_window_s':8,'actual_visual':'union of three original 5-frame recognition sequences; original last approximately1.8seconds, not uniform8s',
          'stride_s':3,'horizons_s':[.5,1,2,3],'selection':'dev MAE decrease AND CCC nondecrease versus hold; lowest MAE eligible else hold',
          'PCA_components':32,'PCA_fit':'train samples frames only','API_calls':0,'backbone_training':False,
          'chronos':'existing shared chronos2 asset; four previous zero-shot configurations frozen separately before invocation',
          'sample_counts':dict(Counter(r['split'] for r in samples)),'requested_video_count':124,'official_train_source_frames':sum(int(.7*v['frame_count']) for v in metadata[:98]),
          'needed_unique_frames':sum(len(v['needed_frames']) for v in videos)}
    if (ROOT/'runplan.json').exists():
        assert read(ROOT/'runplan.json')==plan
    else:
        save(ROOT/'runplan.json',plan)
    save(ROOT/'manifest.json',samples);save(ROOT/'videos.json',videos);save(ROOT/'excluded_partitions.json',exclusions)
    for split,rows in labels.items():save(ROOT/f'labels_{split}.json',rows)
    save(ROOT/'manifest_receipt.json',{'samples':len(samples),'counts':plan['sample_counts'],'train_dev_groups_disjoint':not bool(set(groups[:20])&set(groups[20:])),
        'official_test_split':'within-video temporal holdout; shares original groups with train/dev',
        'manifest_sha256':digest(ROOT/'manifest.json'),'plan_sha256':digest(ROOT/'runplan.json'),'future_and_observation_inside_partition':True,'API_calls':0})
    print(json.dumps(plan),flush=True)

if __name__=='__main__':main()
