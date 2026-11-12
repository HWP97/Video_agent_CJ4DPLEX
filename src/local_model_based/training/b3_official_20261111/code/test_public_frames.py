"""실제 public metadata를 변조하여 불규칙 frame와 잘못된 종료값을 거절한다."""
import json
import shutil
import numpy as np
import public_submission as writer
from adapter_io import ROOT

fixture=ROOT/'pilot/public_frame_boundary';(fixture/'public').mkdir(parents=True,exist_ok=True);(fixture/'provenance').mkdir(exist_ok=True)
shutil.copyfile(ROOT/'provenance/official_v4_test.zip',fixture/'provenance/official_v4_test.zip');original=json.loads((ROOT/'public/index.json').read_text());writer.ROOT=fixture;records=[]
for case in ('nonuniform_frames','wrong_endpoint'):
    rows=json.loads(json.dumps(original))
    if case=='nonuniform_frames':rows[0]['observed_frames'][3]+=1
    else:rows[0]['observed_frames']=[2*i for i in range(9)]
    (fixture/'public/index.json').write_text(json.dumps(rows));rejected=False
    try:writer.write_submission(case,np.zeros((3146,12,2)))
    except AssertionError:rejected=True
    assert rejected,case;records.append({'case':case,'rejected':rejected})
result={'passed':True,'cases':records,'GT_coordinates_used':False};(ROOT/'pilot/public_frame_boundary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
