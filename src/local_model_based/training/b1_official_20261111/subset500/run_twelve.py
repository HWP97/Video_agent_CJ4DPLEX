# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
# 실행: 기존 AG Python -B run_twelve.py
"""동결 평가 코드를 바꾸지 않고 승인된 CPU worker 수만 12로 지정한다."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
import time
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
import full_test


def approved_pool(**kwargs):
    return ProcessPoolExecutor(max_workers=12)


if __name__ == '__main__':
    records=list((HERE/'results/test_records').glob('*.json'))
    for path in records:
        row=json.loads(path.read_text())
        if row['status']=='success':
            prediction=HERE/'results/test_predictions'/path.with_suffix('.pt').name
            assert hashlib.sha256(prediction.read_bytes()).hexdigest()==row['prediction_sha256']
    receipt={'pid':os.getpid(),'workers':12,'resumed_verified_records':len(records),'started_unix':time.time(),
             'change':'pool execution configuration only; frozen model/GT/evaluator/selection unchanged'}
    (HERE/'evidence/worker_restart.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
    full_test.ProcessPoolExecutor=approved_pool
    full_test.main()
