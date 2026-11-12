# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "regex"]
# ///
# 실행: 기존 AG Python -B tools/local_demo/relations_verify.py
"""최종 관계 Demo의 디코딩·시간·점수·원본 보존을 독립 확인한다."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import regex

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'demo/local_model_based/scene_graph_anticipation'
DATA = ROOT/'src/local_model_based/data/b1/latest'
EVIDENCE = ROOT.parent/'.omo/evidence/local_relations_latest_20261112'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = json.loads((OUT/'catalog.json').read_text())
    assert len(rows)==20 and len({r['sample_id'].split(':')[1].split('.mp4')[0] for r in rows})==20
    assert all(r['category']=='demo' for r in rows)
    results = []
    for row in rows:
        video = ROOT/row['video']
        meta = json.loads((ROOT/row['prediction']).read_text())
        evaluation = json.loads(video.with_suffix('.evaluation.json').read_text())
        assert abs(sum(r['baseline_R20'] for r in evaluation)/len(evaluation)*100-row['baseline_metric'])<1e-8
        assert abs(sum(r['ours_R20'] for r in evaluation)/len(evaluation)*100-row['ours_metric'])<1e-8
        expected = meta['cutoff_s']+2*(meta['future_end_s']-meta['cutoff_s'])
        probe = json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-of','json',str(video)],text=True))['streams'][0]
        assert probe['width']==1600 and probe['height']==900 and probe['r_frame_rate']=='15/1'
        assert int(probe['nb_frames'])==math.floor(expected*15)+1
        assert abs(float(probe['duration'])-expected)<=1/15+.001
        decoded = subprocess.run(['/usr/bin/ffmpeg','-v','error','-threads','2','-i',str(video),'-f','null','-'],capture_output=True,text=True)
        assert decoded.returncode==0 and not decoded.stderr,(video,decoded.stderr)
        assert (ROOT/row['poster']).is_file()
        results.append({'sample_id':row['sample_id'],'mp4_sha256':digest(video),'probe':probe,'full_decode_mp4':True,'score_match':True})
    assets = json.loads((DATA/'provenance.json').read_text()) + json.loads((DATA/'training_sources.json').read_text())
    for asset in assets:
        assert digest(DATA/asset['destination'])==asset['sha256']
        assert digest(Path(asset['source']))==asset['sha256']
    by_source = {item['source']:item for item in assets}
    for mode, method in (('sgcls','c04_gate_positive_cap3'),('predcls','c06_gate_positive_cap15')):
        training = json.loads((DATA/f'{mode}_{method}.json').read_text())
        for name, expected in training['signature']['sources'].items():
            assert by_source[name]['sha256']==expected
    sources = [*Path(__file__).parent.glob('relations*.py'),* (ROOT/'src/local_model_based/ours/b1').glob('*.py')]
    for source in sources:
        ast.parse(source.read_text())
    pattern = regex.compile(r'[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]')
    for doc in (OUT/'CATALOG.md',ROOT/'src/local_model_based/ours/b1/README.md'):
        assert not pattern.search(doc.read_text())
    sys.path.insert(0,str(ROOT/'src/local_model_based/ours/b1'))
    import latest_inference
    latest_inference.ACTIVE_FORWARD[0] = True
    denied = False
    try:
        with (DATA/'summary.json').open():
            raise AssertionError('forward permitted file access')
    except PermissionError as error:
        denied = str(error)=='model_forward_file_access_forbidden'
    finally:
        latest_inference.ACTIVE_FORWARD[0] = False
    assert denied
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/'verification.json').write_text(json.dumps({'videos':results,'asset_hashes_preserved':len(assets),'forward_file_access_denied':denied,'source_files_ast':len(sources),'mixed_scripts':[]},indent=2))
    print(json.dumps({'verified_demos':len(results),'full_decoded_media':len(results)*2,'asset_hashes_preserved':len(assets),'evidence':str(EVIDENCE)}))


if __name__=='__main__':
    main()
