# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown"]
# ///
# 실행: python3 tools/build_submission.py
"""docs의 Markdown에서 인쇄용 HTML을 만든다. 생성된 HTML은 Chromium headless로 PDF로 변환한다."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import markdown

ROOT: Final = Path(__file__).resolve().parents[1]


def build_print() -> None:
    destination = ROOT / "docs/print"
    destination.mkdir(parents=True, exist_ok=True)
    style = '''@page{size:A4;margin:17mm}body{font:10pt/1.6 "Noto Sans CJK KR",sans-serif;color:#202623}h1{font-size:23pt;margin:0 0 16pt}h2{font-size:14pt;margin:20pt 0 8pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}p{orphans:3;widows:3}table{border-collapse:collapse;width:100%;font-size:8.5pt;break-inside:avoid}th,td{border-bottom:1px solid #c8cfca;text-align:left;padding:6pt;word-break:normal}pre{white-space:pre-wrap;font-size:8pt}a{color:#1f5f47}code{overflow-wrap:anywhere}'''
    style += 'body,th,td{word-break:keep-all}img{max-width:100%;height:auto;break-inside:avoid}'
    for name in ("SYSTEM_ARCHITECTURE", "TECHNICAL_REPORT", "EXECUTIVE_SUMMARY", "USAGE_AND_SCOPE"):
        text = (ROOT / "docs" / f"{name}.md").read_text()
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
        body = re.sub(r'src="(?!https?://)([^"]+)"', r'src="../\1"', body)
        body = re.sub(r'<a href="(?!https?://)[^"]+">([^<]*)</a>', r'\1', body)
        document_style = style
        if name == "EXECUTIVE_SUMMARY":
            document_style += 'th:first-child,td:first-child{white-space:nowrap}'
        if name == "SYSTEM_ARCHITECTURE":
            document_style += 'body{line-height:1.5}p:last-child{break-inside:avoid}'
        (destination / f"{name}.html").write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>{name}</title><style>{document_style}</style><body>{body}</body></html>')


if __name__ == "__main__":
    build_print()
