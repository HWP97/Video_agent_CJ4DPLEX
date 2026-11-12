"""모델에 보존된 입력필드와 실제 dense endpoint를 전체 표본에서 검증한다."""
import ast
import numpy as np
from common import ROOT,SOURCE,read,save,digest

def main():
    rows=read(ROOT/'manifest.json');videos=read(ROOT/'videos.json');metadata=read(SOURCE/'data_manifest.json')
    forbidden={'GT_min','GT_max','GT_sha256','GT_alignment','target_reference','future','current','truth','labels'}
    assert all(not forbidden.intersection(r) for r in rows+videos)
    # Audit the actual metadata subscript expression, not a claim based only on a report.
    tree=ast.parse((ROOT/'dense.py').read_text());reads=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name) and node.value.value.id=='metadata':
            reads.append(ast.literal_eval(node.slice))
    assert reads==['frame_pts_s'],reads
    checked=0
    byid={r['sample_id']:r for r in rows}
    for path in (ROOT/'dense_cache').glob('*.npz'):
        a=np.load(path)
        for sid,ends in zip(a['sample_ids'],a['endpoints']):
            r=byid[str(sid)];pts=np.array(metadata[r['video_id']]['frame_pts_s']);grid=r['cutoff_s']-np.arange(72,-1,-1)/10
            expected=np.searchsorted(pts,grid+1e-9,side='right')-1
            assert np.array_equal(ends,expected) and np.all(pts[ends]<=grid+1e-8)
            assert pts[ends.min()-20]>=r['observed_start_s']-1e-8 and ends.min()-20>=r['partition_start_frame']
            checked+=1
    assert checked==2740
    save(ROOT/'input_field_audit.json',{'manifest_samples':len(rows),'dense_samples_checked':checked,'forbidden_fields_absent_in_model_manifests':True,
         'actual_dense_source_metadata_keys_accessed':reads,'source_metadata_contains_GT_summaries':True,
         'model_uses_GT_summary_or_target_reference':False,'actual_rating_arrays_or_labels_blocked_during_inference':True,
         'source_code_sha256':digest(ROOT/'dense.py'),'interpretation':'GT_access=false receipts mean no actual GT label arrays consumed by model; metadata container itself includes GT summary fields'})
    print('all2740 dense endpoint boundaries and metadata field whitelist PASS',flush=True)

if __name__=='__main__':main()
