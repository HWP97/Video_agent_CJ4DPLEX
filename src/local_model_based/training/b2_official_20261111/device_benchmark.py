"""같은 합성 배치로 장치 시간만 비교한다. 결과 모델은 버리고 학습 seed를 다시 고정한다."""
import time
import numpy as np
import torch
from temporal import Temporal,views,contrast
from common import ROOT,read,save,digest

def choose():
    path=ROOT/'device_benchmark.json'
    sources={name:digest(ROOT/name) for name in ['device_benchmark.py','temporal.py']}
    if path.exists():
        prior=read(path);assert prior['source_hashes']==sources;return prior['selected']
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42);x=torch.randn(64,15,32);numeric=torch.randn(64,16);target=torch.randn(64,4,2)
    timing={}
    for device in ['cpu','cuda:0']:
        torch.manual_seed(42);generator=torch.Generator().manual_seed(42)
        model=Temporal('direct',32).to(device);xx=x.to(device);nn=numeric.to(device);yy=target.to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        for step in range(25):
            if step==5:
                if device!='cpu':torch.cuda.synchronize()
                start=time.perf_counter()
            main=(model(xx,nn)-yy).square().mean();aux=contrast(model,xx,views(xx,generator))
            loss=main+.1*aux;assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            float(main.detach());float(aux.detach())
        if device!='cpu':torch.cuda.synchronize()
        timing[device]=(time.perf_counter()-start)/20
        del model,opt,xx,nn,yy
    selected='cuda:0' if timing['cuda:0']<timing['cpu']/1.2 else 'cpu'
    save(path,{'selected':selected,'seconds_per_step':timing,'CPU_over_GPU_speedup':timing['cpu']/timing['cuda:0'],
        'warmup_steps':5,'timed_steps':20,'shape':[64,15,32],'contrast_weight':.1,'source_hashes':sources,
        'training_data_used':False,'retained_model':False,'selection_rule':'GPU if measured speedup >1.2x, else CPU',
        'precision':'FP32, TF32 disabled','torch':torch.__version__})
    return selected
