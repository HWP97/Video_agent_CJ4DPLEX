"""실제 검증된 코드만 별도 snapshot으로 보존하고 통합용 자산·명령을 기록한다."""
import shutil
import importlib.metadata
from pathlib import Path
from common import ROOT,SOURCE,read,save,digest

def main():
    assert (ROOT/'completion_receipt.json').exists()
    snapshot=ROOT/'code_snapshot';snapshot.mkdir(exist_ok=True);code={}
    for path in sorted(ROOT.glob('*.py')):
        target=snapshot/path.name
        if target.exists():assert digest(target)==digest(path),path
        else:shutil.copy2(path,target)
        code[str(target)]=digest(target)
    note=snapshot/'README.md'
    text='# 검증된 b2 공식 train 실험 소스\n\n이 폴더는 Python 소스와 이 안내만 포함한다. 가중치·라벨·영상·특징·실험 결과·환경은 포함하지 않는다.\n\n과학 코드의 ROOT는 실행 파일 위치를 기준으로 한다. 이 snapshot을 HWP에 복사한 뒤 그대로 학습 실행하면 결과 경로가 HWP 내부가 되므로 실행하지 않는다. 통합 시 별도 support artifact 경로를 명시하도록 진입점을 연결해야 한다. 기존 검증 실행 위치와 자산 SHA·재현 명령은 상위 source_manifest.json을 따른다.\n'
    if note.exists():assert note.read_text()==text
    else:note.write_text(text)
    code[str(note)]=digest(note)
    source=read(ROOT/'training_sources.json')
    assert all(digest(ROOT/name)==sha for name,sha in source['script_hashes'].items())
    assets=[SOURCE/'data/one_stream_vit.pth',Path('/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/assets/chronos2/model/model.safetensors'),
            ROOT/'projection.npz',ROOT/'memory.npz',ROOT/'inputs.npz',ROOT/'dense_inputs.npz',ROOT/'offset_model.json',ROOT/'departure_model.json']
    assets+=list((ROOT/'trained_models').glob('*.npz'))+list((ROOT/'trained_models').glob('*.pt'))
    versions={}
    for name in ['numpy','torch','torchvision','Pillow']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable in export environment'
    commands={
       'inference_without_training':'CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 '+
          '/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/assets/chronos2/env/bin/python -B '+str(ROOT/'replay.py')+' infer NEW_RUN_NAME',
       'rescore_without_training':'/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B '+str(ROOT/'replay.py')+' score NEW_RUN_NAME',
       'resume_primary_features':'CUDA_VISIBLE_DEVICES=0 /mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B '+str(ROOT/'feature_guard.py')+' extract',
       'resume_dense':'/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B '+str(ROOT/'pipeline.py')+' dense',
       'resume_failed_head_stage':'/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B '+str(ROOT/'learning.py')+' train',
       'verify':'/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B '+str(ROOT/'verify.py')}
    commands['readonly_full_check']='/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python -B '+str(ROOT/'readonly_check.py')
    save(ROOT/'source_manifest.json',{'code_snapshot_sha256':code,'assets_sha256':{str(p):digest(p) for p in assets},'commands':commands,
         'environment_versions':versions,'existing_recognition_python':'/mnt/RG2/my_works/b1/envs/qwen3vl/bin/python',
         'existing_Chronos_and_CRC_python':'/mnt/RG2/llm_transfer_expansion_20261110/additional_candidates/assets/chronos2/env/bin/python',
         'reusable_functions':['learning.predict_one','learning.representation','temporal.Temporal(arm=direct)','supplement.Context','chronos2.adapter.forecast'],
         'integration_note':'This is the verified scientific snapshot. ROOT currently resolves to this support attempt. HWP integration must pass a separate artifact directory; do not directly run copied training entrypoints inside HWP.',
         'weights_and_data_must_remain_outside_HWP':True,'snapshot_contains_no_weights_or_environment':True,'training_fingerprint_sha256':digest(ROOT/'training_sources.json')})
    print('verified code snapshot files='+str(len(code)),flush=True)

if __name__=='__main__':main()
