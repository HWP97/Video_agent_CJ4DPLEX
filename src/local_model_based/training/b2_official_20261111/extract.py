"""GPU0의 단일 동결 모델로 필요한 관측 프레임만 처리하고 영상별 재개한다."""
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from common import ROOT,SOURCE,digest,guard,read,save

sys.path[:0]=[str(SOURCE/'vendor'),str(SOURCE/'sources/VEATIC-2a9d5f1b3b4af3c8746c9a508ceb44f4b4891f95')]
import model as official

def preprocess(bgr):
    okay,encoded=cv2.imencode('.jpg',bgr)
    assert okay
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        rgb=np.asarray(image.convert('RGB'))/128.-1.
    return cv2.resize(rgb,(640,480)).transpose(2,0,1).astype(np.float32)

def main():
    guard('extract')
    torch.set_num_threads(2);torch.manual_seed(42);cv2.setNumThreads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    checkpoint=SOURCE/'data/one_stream_vit.pth'
    assert digest(checkpoint)=='8c5fedc813fcc4b96c4d541ea7dc859cf154f9d19a5fe3d47ffb6c43cb2ee006'
    original=official.resnet50
    official.resnet50=lambda **kwargs:original(weights=None)
    model=official.VEATIC_baseline()
    official.resnet50=original
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().to('cuda:0')
    manifest=read(ROOT/'manifest.json');videos=read(ROOT/'videos.json')
    cache=ROOT/'feature_cache';cache.mkdir(exist_ok=True)
    groups={}
    for video in videos:groups.setdefault(video['source_sha256'],[]).append(video)
    receipts=[]
    for group_index,(sha,versions) in enumerate(groups.items()):
        path=cache/(sha+'.npz');receipt_path=cache/(sha+'.json')
        if path.exists() and receipt_path.exists():
            receipt=read(receipt_path)
            assert digest(path)==receipt['cache_sha256']
            receipts.append(receipt)
            continue
        video=versions[0]
        before=time.perf_counter()
        assert digest(video['source'])==sha
        indices=sorted({i for v in versions for i in v['needed_frames']})
        if not indices:
            receipt={'video_ids':[v['video_id'] for v in versions],'source_sha256':sha,
                     'status':'excluded_no_observation_window','frames_extracted':0,'samples':0,
                     'seconds':time.perf_counter()-before,'reason':'all partitions shorter than 8s observation + 3s future'}
            save(receipt_path,receipt);receipts.append(receipt)
            continue
        wanted=set(indices);pixels=[];chunks=[];observed_indices=[];first_pixels={}
        pilot_sequence=manifest[0]['recognition_sequences'][0]
        capture=cv2.VideoCapture(video['source'])
        frame_index=0
        try:
            while frame_index<=indices[-1]:
                okay,bgr=capture.read()
                if not okay:break
                if frame_index in wanted:
                    pixel=preprocess(bgr);pixels.append(pixel);observed_indices.append(frame_index)
                    if group_index==0 and frame_index in pilot_sequence:first_pixels[frame_index]=pixel
                    if len(pixels)==8:
                        tensor=torch.from_numpy(np.stack(pixels)).to('cuda:0')
                        with torch.inference_mode():
                            chunks.append(model.pool(model.human(model.feature(tensor))).flatten(1).cpu().numpy())
                        pixels.clear()
                frame_index+=1
            if pixels:
                tensor=torch.from_numpy(np.stack(pixels)).to('cuda:0')
                with torch.inference_mode():chunks.append(model.pool(model.human(model.feature(tensor))).flatten(1).cpu().numpy())
        finally:capture.release()
        assert observed_indices==indices,(video['video_id'],frame_index,len(indices))
        features=np.concatenate(chunks)
        positions={index:i for i,index in enumerate(indices)}
        sample_rows=[r for r in manifest if r['video_id'] in {v['video_id'] for v in versions}]
        sequences=[seq for r in sample_rows for seq in r['recognition_sequences']]
        outputs=[]
        with torch.inference_mode():
            for offset in range(0,len(sequences),64):
                batch=np.array([[positions[i] for i in seq] for seq in sequences[offset:offset+64]])
                outputs.append(model.ViT(torch.from_numpy(features[batch]).to('cuda:0')).cpu().numpy())
        recognition=np.concatenate(outputs).reshape(len(sample_rows),3,2)
        assert np.isfinite(features).all() and np.isfinite(recognition).all()
        if group_index==0:
            with torch.inference_mode():
                direct=model(torch.from_numpy(np.stack([first_pixels[i] for i in pilot_sequence])[None]).cuda()).cpu().numpy()[0]
            cached=recognition[0,0];difference=float(np.max(np.abs(direct-cached)))
            passed=bool(np.allclose(direct,cached,atol=2e-5,rtol=1e-5))
            save(ROOT/'pilot_equivalence.json',{'direct':direct.tolist(),'cached':cached.tolist(),'max_abs_difference':difference,
                'atol':2e-5,'rtol':1e-5,'passed':passed,'frames':pilot_sequence,'precision':'FP32, TF32 disabled'})
            assert passed,('strict_RGB_cache_equivalence',difference)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream,indices=np.array(indices),features=features,sample_ids=np.array([r['sample_id'] for r in sample_rows]),recognition=recognition)
        temporary.replace(path)
        elapsed=time.perf_counter()-before
        receipt={'video_ids':[v['video_id'] for v in versions],'source_sha256':sha,'source':video['source'],
                 'frames_extracted':len(indices),'decoded_frames':frame_index,'samples':len(sample_rows),'seconds':elapsed,
                 'cache_sha256':digest(path),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'GPU':torch.cuda.get_device_name(),'CUDA_VISIBLE_DEVICES':'0','model_GT_access':False,'API_calls':0}
        receipt['precision']='float32, CUDA matmul TF32 false, cuDNN TF32 false'
        save(receipt_path,receipt);receipts.append(receipt)
        remaining=sum(len({i for v in group for i in v['needed_frames']}) for group in list(groups.values())[group_index+1:])
        completed_frames=sum(r['frames_extracted'] for r in receipts)
        rate=completed_frames/sum(r['seconds'] for r in receipts)
        progress={'groups_done':len(receipts),'groups_total':len(groups),'frames_done':completed_frames,
                  'seconds_so_far':time.perf_counter()-started,'estimated_remaining_seconds':remaining/rate,
                  'last_video':receipt}
        save(ROOT/'extract_progress.json',progress)
        print(json.dumps(progress),flush=True)
        if '--pilot' in sys.argv:return
    save(ROOT/'extraction_complete.json',{'groups':len(receipts),'requested_videos':len(videos),
         'samples':len(manifest),'frames':sum(r['frames_extracted'] for r in receipts),
         'wall_s_this_invocation':time.perf_counter()-started,'all_group_receipts':receipts,'API_calls':0})

if __name__=='__main__':main()
