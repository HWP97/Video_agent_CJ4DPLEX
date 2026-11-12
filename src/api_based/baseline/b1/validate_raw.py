# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: entry.py verify에서 호출
"""전 Sol 원문·parsed·정확 전송 프레임의 연결을 대조한다."""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qsl, urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parent / 'vendor'))
from sol_schema import parse
from bundle_paths import REPORTS


def reject_credentials(document) -> None:
    blocked = {'apikey','authorization','accesstoken','refreshtoken','clientsecret','secretkey','password','token'}
    signed = blocked | {'key','signature','sig','credential','xamzcredential','xamzsignature','xgoogsignature'}
    queue = [document]
    while queue:
        value = queue.pop()
        if isinstance(value,dict):
            for key,child in value.items():
                normalized = key.lower().replace('-','').replace('_','')
                if normalized in blocked and child not in (None,'','REDACTED'):
                    raise PermissionError('credential field: '+key)
                queue.append(child)
        if isinstance(value,list):
            queue.extend(value)
        if isinstance(value,str):
            if re.search(r'(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])|(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{30,}(?![A-Za-z0-9_-])|\bBearer [A-Za-z0-9._-]{20,}',value):
                raise PermissionError('credential-shaped string')
            for url in re.findall(r'https?://[^\s"<>]+',value):
                keys = {key.lower().replace('-','').replace('_','') for key,_ in parse_qsl(urlsplit(url).query)}
                if keys & signed:
                    raise PermissionError('credential-bearing URL query')


def validate(root: Path):
    hashes = {row['destination']:row['destination_sha256'] for row in map(json.loads,(REPORTS / 'b1/copy_manifest.jsonl').open())}
    inputs = {row['sample_id']:row for row in map(json.loads,(root / 'data/b1/requests.original.jsonl').open())}
    for request in inputs.values():
        reject_credentials(request)
    responses = 0
    excluded = 0
    frame_links = 0
    error_documents = 0
    for row in map(json.loads,(root / 'data/b1/sol/index.final.jsonl').open()):
        if not row.get('valid'):
            excluded += 1
            continue
        signature = row['provider_signature']
        folder = root / 'data/b1/sol/raw' / signature
        metadata = json.loads((folder / 'request.json').read_text())
        raw_input_bytes = (folder / 'input.json').read_bytes()
        assert hashlib.sha256(raw_input_bytes).hexdigest() == metadata['input_record_sha256']
        reject_credentials(json.loads(raw_input_bytes))
        reject_credentials(metadata)
        for error_path in folder.glob('error*.json'):
            reject_credentials(json.loads(error_path.read_text()))
            error_documents += 1
        response_bytes = (folder / 'response.json').read_bytes()
        assert hashlib.sha256(response_bytes).hexdigest() == metadata['response_sha256']
        response = json.loads(response_bytes)
        reject_credentials(response)
        text = ''.join(part.get('text','') for message in response.get('output',[]) for part in message.get('content',[]) if part.get('type')=='output_text')
        request = inputs[row['sample_id']]
        assert parse(text,request) == row['prediction']
        assert row['input_signature'] == request['input_signature']
        for frame in request['frames']:
            assert hashes['data/b1/images/'+frame['source_frame_id']] == frame['sha256']
            frame_links += 1
        responses += 1
    return {'raw_parsed_matches':responses,'excluded':excluded,'exact_transmitted_frame_links':frame_links,'credential_documents_checked':responses*3+len(inputs)+error_documents,'error_documents_checked':error_documents,'credential_fields_or_token_urls':0}


if __name__ == '__main__':
    from bundle_paths import ROOT, RESULTS
    result = validate(ROOT)
    target = RESULTS / 'b1/qa_raw_integrity'
    target.mkdir(exist_ok=False)
    (target / 'receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
