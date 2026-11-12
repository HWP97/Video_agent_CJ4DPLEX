# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest"]
# ///
# 실행: python -m pytest tests/test_b1_credentials.py
"""인증정보 검사에서 실제 자격정보와 암호화 payload를 구분한다."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / 'baseline/b1'))
from validate_raw import reject_credentials


@pytest.mark.parametrize('document',[
    {'Authorization':'Bearer example-secret'},
    {'nested':{'api_key':'example-secret'}},
    {'text':'request failed: https://example.test/path?token=example-secret'},
    {'text':'sk-'+'x'*30},
])
def test_authentication_material_is_rejected(document) -> None:
    with pytest.raises(PermissionError):
        reject_credentials(document)


def test_base64url_internal_sk_sequence_is_not_a_key() -> None:
    document = {'encrypted_content':'AAAA-sk-'+'x'*80,'usage':{'input_tokens':10,'output_tokens':20}}
    assert reject_credentials(document) is None
