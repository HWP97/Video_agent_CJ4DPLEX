"""실제 원형 RGB forward와 cache 분해 추론을 한 실제 시퀀스로 비교한다."""
import numpy as np
import torch
import cv2
from extract import official,preprocess
from common import ROOT,SOURCE,read,save,guard,digest

def main():
    guard('extract');torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    row=read(ROOT/'manifest.json')[0];video=read(ROOT/'videos.json')[0];seq=row['recognition_sequences'][0]
    capture=cv2.VideoCapture(video['source']);pixels=[]
    try:
        for i in range(seq[-1]+1):
            okay,bgr=capture.read();assert okay
            if i in seq:pixels.append(preprocess(bgr))
    finally:capture.release()
    original=official.resnet50;official.resnet50=lambda **kwargs:original(weights=None)
    model=official.VEATIC_baseline();official.resnet50=original
    model.load_state_dict(torch.load(SOURCE/'data/one_stream_vit.pth',map_location='cpu',weights_only=True));model.eval().cuda()
    with torch.inference_mode():direct=model(torch.from_numpy(np.stack(pixels)[None]).cuda()).cpu().numpy()[0]
    a=np.load(ROOT/'feature_cache'/(video['source_sha256']+'.npz'));i=list(a['sample_ids']).index(row['sample_id']);cached=a['recognition'][i,0]
    difference=float(np.max(np.abs(direct-cached)));assert np.allclose(direct,cached,atol=2e-5,rtol=1e-5),(direct,cached,difference)
    save(ROOT/'equivalence.json',{'sample_id':row['sample_id'],'frame_indices':seq,'direct_RGB_VA':direct.tolist(),'cached_VA':cached.tolist(),
          'max_absolute_error':difference,'absolute_tolerance':2e-5,'relative_tolerance':1e-5,'passed':True,'GT_access':False,'source_video_sha256':digest(video['source']),
          'precision':'FP32 with cuDNN and matmul TF32 disabled'})
    densepath=ROOT/'dense_cache'/(video['source_sha256']+'.npz')
    if densepath.exists():
        dense=np.load(densepath);endpoint=int(dense['endpoints'][0,0]);frames=list(range(endpoint-20,endpoint+1,5))
        cap=cv2.VideoCapture(video['source']);observations=[]
        try:
            for i in range(endpoint+1):
                okay,bgr=cap.read();assert okay
                if i in frames:observations.append(preprocess(bgr))
        finally:cap.release()
        with torch.inference_mode():actual=model(torch.from_numpy(np.stack(observations)[None]).cuda()).cpu().numpy()[0]
        expected=dense['long'][0,0];difference=float(np.max(np.abs(actual-expected)))
        assert np.allclose(actual,expected,atol=2e-5,rtol=1e-5)
        save(ROOT/'dense_equivalence.json',{'frames':frames,'direct':actual.tolist(),'cached':expected.tolist(),'max_abs_difference':difference,
              'passed':True,'GT_access':False,'meaning':'actual dense RGB sequence, including parallel preprocessing, versus GPU1 cache'})

if __name__=='__main__':main()
