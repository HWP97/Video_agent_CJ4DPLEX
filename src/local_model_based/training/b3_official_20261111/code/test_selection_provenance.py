"""실제 dev 채점 영수증을 대조하고 새 선택/재개 변조를 격리해서 거절한다."""
import copy
import json
import shutil
import sys
import select_methods as selection
from adapter_io import ROOT,sha

summary=ROOT/'scores/fold1/dev/summary.json';receipt=json.loads((summary.parent/'receipt.json').read_text());selection.validate_scores(summary,receipt,1);records=[{'case':'actual_complete_dev_receipt','accepted':True}]
for case in ('wrong_summary_sha','wrong_role','wrong_runplan'):
    changed=copy.deepcopy(receipt)
    if case=='wrong_summary_sha':changed['summary_sha256']='wrong'
    elif case=='wrong_role':changed['role']='eval'
    else:changed['runplan']='wrong'
    rejected=False
    try:selection.validate_scores(summary,changed,1)
    except AssertionError:rejected=True
    assert rejected,case;records.append({'case':'fresh_'+case,'rejected':True})
fixture=ROOT/'pilot/selection_mutation';(fixture/'code').mkdir(parents=True,exist_ok=True);(fixture/'selections').mkdir(exist_ok=True)
for name in ('runplan.json','ours_candidate_catalog.json','attention_transfer_amendment.json','code/select_methods.py'):shutil.copyfile(ROOT/name,fixture/name)
snapshot=fixture/'selections/fold1_dev_scores.json';shutil.copyfile(summary,snapshot)
base={'source':sha(fixture/'code/select_methods.py'),'candidate_catalog':sha(fixture/'ours_candidate_catalog.json'),'attention_transfer_amendment':sha(fixture/'attention_transfer_amendment.json'),'runplan':sha(fixture/'runplan.json'),'dev_scores':sha(snapshot),'score_receipt':receipt}
selection.ROOT=fixture;sys.argv=['select_methods.py','1']
for case in ('runplan','dev_scores'):
    changed={**base,case:'wrong'};(fixture/'selections/fold1.json').write_text(json.dumps(changed));rejected=False
    try:selection.main()
    except AssertionError:rejected=True
    assert rejected,case;records.append({'case':'resume_wrong_'+case,'rejected':True})
selection.ROOT=ROOT
result={'passed':True,'cases':records,'actual_prediction_or_selection_modified':False};(ROOT/'pilot/selection_provenance_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
