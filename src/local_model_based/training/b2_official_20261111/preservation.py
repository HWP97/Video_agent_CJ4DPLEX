"""최초 동결 입력 복사본과 원본372개 데이터의 공개 manifest SHA를 대조한다."""
import time
from common import ROOT,SOURCE,read,save,digest
from feature_guard import contract

def main():
    start=time.perf_counter();checks=[]
    for name in ['manifest.json','videos.json','labels_train.json','labels_dev.json','labels_eval.json','runplan.json','adapter_runplan.json','supplement_runplan.json','objective_runplan.json']:
        current=digest(ROOT/name);initial=digest(ROOT.parent/name);assert current==initial,name
        checks.append({'path':str(ROOT/name),'initial_copy':str(ROOT.parent/name),'sha256':current,'unchanged':True})
    original=[]
    for row in read(SOURCE/'data_manifest.json'):
        assets=[(SOURCE/row['video_path'],row['sha256'])]+[(SOURCE/f"data/VEATIC/rating_averaged/{row['video_id']}_{axis}.csv",h) for axis,h in row['GT_sha256'].items()]
        for path,expected in assets:
            actual=digest(path);assert actual==expected,path
            original.append({'path':str(path),'sha256':actual,'matches_source_manifest':True})
    assert len(original)==372
    for path,expected in read(ROOT/'adapter_runplan.json')['source_hashes'].items():assert digest(path)==expected,path
    contract()
    save(ROOT/'preservation_verification.json',{'initial_frozen_inputs':checks,'original_data':original,'original_data_count':372,
        'original_recipe_source_hashes_match':True,'feature_contract_verified':True,'seconds':time.perf_counter()-start,
        'HWP_writes_by_this_task':0,'note':'all source data match the source manifest; scientific inputs match the initial pre-training frozen copies; no claim about unrelated actors writing shared HWP'})
    print('original372 SHA PASS; initial frozen inputs9 SHA PASS; recipe and feature contracts PASS',flush=True)

if __name__=='__main__':main()
