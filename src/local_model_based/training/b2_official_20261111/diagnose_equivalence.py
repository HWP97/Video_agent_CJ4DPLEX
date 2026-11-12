"""실제 동일 픽셀에서 배치 크기와 TF32를 분리하는 실패 진단."""
import numpy as np
import torch
import cv2
from extract import official,preprocess
from common import ROOT,SOURCE,read,save,guard

def main():
    guard('extract');torch.set_num_threads(2);cv2.setNumThreads(2)
    row=read(ROOT/'manifest.json')[0];v=read(ROOT/'videos.json')[0];seq=row['recognition_sequences'][0]
    cap=cv2.VideoCapture(v['source']);pixels=[]
    for i in range(max(row['frame_indices'])+1):
        ok,bgr=cap.read();assert ok
        if i in row['frame_indices']:pixels.append(preprocess(bgr))
    cap.release();pixels=np.stack(pixels);indices=[row['frame_indices'].index(i) for i in seq]
    original=official.resnet50;official.resnet50=lambda **kwargs:original(weights=None)
    model=official.VEATIC_baseline();official.resnet50=original;model.load_state_dict(torch.load(SOURCE/'data/one_stream_vit.pth',map_location='cpu',weights_only=True));model.eval().cuda()
    a=np.load(ROOT/'feature_cache'/f"{v['source_sha256']}.npz");positions={int(f):i for i,f in enumerate(a['indices'])};cached=a['features'][[positions[i] for i in seq]]
    stored=a['recognition'][0,0];results=[]
    initial={'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'torch':torch.__version__}
    for tf32 in [True,False]:
        torch.backends.cudnn.allow_tf32=tf32;torch.backends.cuda.matmul.allow_tf32=False
        output={};features={}
        with torch.inference_mode():
            for batch in [1,8]:
                chunks=[]
                for start in range(0,len(pixels),batch):
                    t=torch.from_numpy(pixels[start:start+batch]).cuda();chunks.append(model.pool(model.human(model.feature(t))).flatten(1).cpu().numpy())
                f=np.concatenate(chunks)[indices];features[batch]=f
                output[batch]=model.ViT(torch.from_numpy(f[None]).cuda()).cpu().numpy()[0]
            direct=model(torch.from_numpy(pixels[indices][None]).cuda()).cpu().numpy()[0]
            cachehead=model.ViT(torch.from_numpy(cached[None]).cuda()).cpu().numpy()[0]
        results.append({'cudnn_tf32':tf32,'direct':direct.tolist(),'batch1':output[1].tolist(),'batch8':output[8].tolist(),
             'stored':stored.tolist(),'cachehead':cachehead.tolist(),'batch1_vs8_feature_max':float(np.max(np.abs(features[1]-features[8]))),
             'batch1_vs8_VA_max':float(np.max(np.abs(output[1]-output[8]))),'batch8_vs_cache_feature_max':float(np.max(np.abs(features[8]-cached))),
             'cachehead_vs_stored_VA_max':float(np.max(np.abs(cachehead-stored)))})
    save(ROOT/'equivalence_diagnosis.json',{'initial_settings':initial,'sample_id':row['sample_id'],'results':results})
    print(results,flush=True)

if __name__=='__main__':main()
