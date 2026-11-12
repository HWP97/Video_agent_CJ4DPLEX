"""단계별 별도 프로세스 실행과 종료 상태 기록. 이미 실행 중인 추출은 기다린다."""
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from common import ROOT,save,read

def wait_for(path):
    while not path.exists():time.sleep(3)

def run(route):
    started=time.perf_counter();started_utc=datetime.now(timezone.utc).isoformat();receipts=[]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='1' if route=='dense' else '0',
         TMPDIR=str(ROOT/'runtime'),XDG_CACHE_HOME=str(ROOT/'runtime'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if route=='cpu':
        wait_for(ROOT/'extraction_complete.json')
        stages=[('assemble.py',[]),('learning.py',['train']),('learning.py',['infer']),('supplement.py',['train']),('supplement.py',['infer']),('departure.py',['train']),('departure.py',['infer'])]
    elif route=='gpu':
        wait_for(ROOT/'extraction_complete.json')
        wait_for(ROOT/'pipeline_cpu.json');assert read(ROOT/'pipeline_cpu.json')['status']=='completed'
        stages=[('equivalence.py',[]),('chronos_run.py',['forecast'])]
    elif route=='dense':
        stages=[('dense.py',[])]
    elif route=='final':
        for previous in ['cpu','gpu']:
            wait_for(ROOT/f'pipeline_{previous}.json');assert read(ROOT/f'pipeline_{previous}.json')['status']=='completed'
        stages=[('chronos_run.py',['controls']),('supplement.py',['select']),('learning.py',['score']),('departure.py',['score']),('input_audit.py',[]),('verify.py',[])]
    else:raise ValueError(route)
    for script,args in stages:
        if script=='chronos_run.py' and args==['forecast']:wait_for(ROOT/'dense_complete.json')
        before=time.perf_counter();utc=datetime.now(timezone.utc).isoformat();python=sys.executable
        if (script=='chronos_run.py' and args==['forecast']) or script=='departure.py':
            python='/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/assets/chronos2/env/bin/python'
        command=[python,'-B','-u',str(ROOT/script)]+args;log=ROOT/'logs'/(script[:-3]+'_'+('_'.join(args) or 'run')+'.log')
        with log.open('a') as stream:result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        item={'command':command,'started_utc':utc,'seconds':time.perf_counter()-before,'exit_code':result.returncode,'log':str(log)}
        receipts.append(item);save(ROOT/f'pipeline_{route}_progress.json',receipts);print(item,flush=True)
        if result.returncode:
            save(ROOT/f'pipeline_{route}.json',{'status':'failed','stages':receipts,'started_utc':started_utc,'wall_s':time.perf_counter()-started});raise SystemExit(result.returncode)
    save(ROOT/f'pipeline_{route}.json',{'status':'completed','stages':receipts,'started_utc':started_utc,'wall_s':time.perf_counter()-started})

if __name__=='__main__':run(sys.argv[1])
