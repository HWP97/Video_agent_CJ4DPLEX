"""작성 중 파일·변조 SHA·순서·shape·NaN이 최종 채점에 들어가지 않는지 검사한다."""
import json
import numpy as np
import freeze_predictions as controller
from adapter_io import ROOT,sha

fixture=ROOT/'pilot/manifest_mutation';(fixture/'attention/fold1').mkdir(parents=True,exist_ok=True);(fixture/'data/indices').mkdir(parents=True,exist_ok=True)
(fixture/'runplan.json').write_text(json.dumps({'folds':[{'development':'group'}],'groups':{'group':[{'file':'scene.ndjson'}]}}))
(fixture/'data/indices/scene_index.json').write_text(json.dumps([{'sample_id':'scene_0'}]))
(fixture/'method_registry.json').write_bytes((ROOT/'method_registry.json').read_bytes())
path=fixture/'attention/fold1/direct_unshared4_dev.npy';receipt=path.with_suffix('.json')
checkpoint=fixture/'attention/fold1/direct_unshared4.pt';checkpoint.write_bytes(b'fixture_checkpoint')
seal=fixture/'attention/fold1/dev_inference_receipt.json'
if seal.exists():seal.unlink()
(fixture/'attention/fold1/incomplete_dev_d1.npy').write_bytes(b'partial_npy')
controller.ROOT=fixture
records=[]
for case in ('valid','hash_corruption','wrong_order','wrong_shape','nonfinite','wrong_runplan'):
    value=np.zeros((1,12,2))
    if case=='wrong_shape':value=np.zeros((1,11,2))
    if case=='nonfinite':value[0,0,0]=np.nan
    np.save(path,value);row={'sample_ids':['scene_0'],'sha256':sha(path),'runplan_sha256':sha(fixture/'runplan.json'),'checkpoint_sha256':sha(checkpoint)}
    if case=='wrong_order':row['sample_ids']=['another_scene']
    if case=='wrong_runplan':row['runplan_sha256']='wrong'
    receipt.write_text(json.dumps(row))
    if case=='hash_corruption':
        with path.open('ab') as stream:stream.write(b'corrupt')
    rejected=False
    try:found=controller.freeze(1,'dev')
    except AssertionError:rejected=True
    assert rejected==(case!='valid'),case
    if case=='valid':assert list(found)==['loop/direct_unshared4_d4']
    records.append({'case':case,'rejected':rejected,'expected':case!='valid'})
np.save(path,np.zeros((1,12,2)));receipt.write_text(json.dumps({'sample_ids':['scene_0'],'sha256':sha(path),'runplan_sha256':sha(fixture/'runplan.json'),'checkpoint_sha256':sha(checkpoint)}))
producer=fixture/'attention/fold1/direct_single.json';producer.write_text('{}');model=producer.with_suffix('.pt');model.write_bytes(b'another_checkpoint');prediction=fixture/'attention/fold1/direct_single_dev_d1.npy';np.save(prediction,np.ones((1,12,2)))
metadata=json.loads((fixture/'data/indices/scene_index.json').read_text());order=__import__('hashlib').sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
value={'fold':1,'role':'dev','runplan':sha(fixture/'runplan.json'),'order_sha256':order,'n':1,'models':{'direct_single':{'checkpoint':sha(model),'receipt':sha(producer)}},'outputs':{'direct_single_d1':{'path':prediction.name,'sha256':sha(prediction)}}};seal.write_text(json.dumps(value))
assert 'loop/direct_single_d1' in controller.freeze(1,'dev')
for case in ('changed_sealed_prediction','changed_sealed_runplan'):
    if case=='changed_sealed_prediction':np.save(prediction,np.zeros((1,12,2)))
    else:np.save(prediction,np.ones((1,12,2)));value['runplan']='wrong';seal.write_text(json.dumps(value))
    rejected=False
    try:controller.freeze(1,'dev')
    except AssertionError:rejected=True
    assert rejected,case;records.append({'case':case,'rejected':True,'expected':True})
out=ROOT/'pilot/prediction_manifest_boundary.json';out.write_text(json.dumps({'passed':True,'unreceipted_partial_npy_ignored':True,'cases':records},indent=2));print(out.read_text())
