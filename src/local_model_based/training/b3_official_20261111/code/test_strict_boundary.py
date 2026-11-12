"""원시 archive·평가 cache·외부 cache는 닫고 정상 관측 cache는 허용한다."""
import json
import socket
from pathlib import Path
import numpy as np
from adapter_io import ROOT
from run_guarded import install

install('infer',1);records=[]
for name,path in [('raw_train_zip',ROOT/'provenance/official_v4_train.zip'),('eval_only',ROOT/'evaluation_only/biwi_hotel.npz'),('labels',ROOT/'data/targets/biwi_hotel_labels.npz'),('external_cache',Path('/mnt/RG2/llm_transfer_20261110/b3/cache/validation.npz'))]:
    rejected=False
    try:
        with path.open('rb') as stream:stream.read(1)
    except PermissionError:rejected=True
    assert rejected,name;records.append({'case':name,'blocked':True})
with socket.socket() as connection:
    try:connection.connect(('127.0.0.1',9))
    except PermissionError:records.append({'case':'network','blocked':True})
    else:raise AssertionError('network_not_blocked')
with np.load(ROOT/'public/inputs.npz') as values:assert len(values['ego'])==3146
result={'passed':True,'cases':records,'observed_input_allowed':3146};(ROOT/'pilot/strict_boundary_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
