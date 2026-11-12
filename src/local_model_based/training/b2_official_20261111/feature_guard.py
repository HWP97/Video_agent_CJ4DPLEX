"""특징 추출의 재개 시 manifest·가중치·실행 코드를 최초 값과 비교한다."""
import sys
import runpy
from common import ROOT,SOURCE,freeze_contract,digest,read

def contract():
    paths=[ROOT/'manifest.json',ROOT/'videos.json',ROOT/'extract.py',ROOT/'runplan.json',
           SOURCE/'data/one_stream_vit.pth',SOURCE/'sources/VEATIC-2a9d5f1b3b4af3c8746c9a508ceb44f4b4891f95/model.py',
           SOURCE/'sources/VEATIC-2a9d5f1b3b4af3c8746c9a508ceb44f4b4891f95/ViT.py']
    return freeze_contract(ROOT/'feature_contract.json',{'hashes':{str(p):digest(p) for p in paths},
        'precision':'FP32, matmul and cuDNN TF32 disabled','backbone_batch':8,'temporal_head_batch':64,
        'preprocessing':'JPEG roundtrip /128-1 resize640x480 CHW','primary_inputs':'original15 union, approximately1.8s'})

if __name__=='__main__':
    contract()
    if len(sys.argv)>1 and sys.argv[1]=='extract':
        sys.argv=[str(ROOT/'extract.py')]+sys.argv[2:];runpy.run_path(str(ROOT/'extract.py'),run_name='__main__')
