"""로컬 모델 Demo의 첫 5개를 각 head별로 GIF로 만든다.
API와 동일한 ffmpeg 설정 사용(fps=6, scale=800, 7.5초 꼬리 발췌).
입력 MP4는 support/submission_cleanup_20261112/demo_media/ 에 있다."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 로컬 MP4 원본 위치. 기본은 원 작업 서버 경로지만 --media 또는 VFA_LOCAL_DEMO_MEDIA 로 교체 가능.
DEFAULT_MEDIA = Path('/mnt/RG2/HWP_CJ4DPLEX_AI_support/submission_cleanup_20261112/demo_media')
HEADS = ('scene_graph_anticipation', 'affect_forecasting', 'trajectory_forecasting')


def duration(source: Path) -> float:
    out = subprocess.run(['/usr/bin/ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'default=noprint_wrappers=1:nokey=1', str(source)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def convert(source: Path, output: Path, start: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = ('fps=6,scale=800:-1:flags=lanczos,split[a][b];'
               '[a]palettegen=max_colors=96:stats_mode=diff[p];'
               '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
    subprocess.run(['/usr/bin/ffmpeg', '-v', 'error', '-y',
                    '-ss', str(start), '-t', '7.5',
                    '-i', str(source), '-filter_complex', filters,
                    '-an', '-loop', '0', str(output)], check=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', help='로컬 Demo MP4 원본 루트. 기본: $VFA_LOCAL_DEMO_MEDIA 또는 코드 상수.')
    args = parser.parse_args()
    media = Path(args.media or os.environ.get('VFA_LOCAL_DEMO_MEDIA') or DEFAULT_MEDIA)
    if not media.is_dir():
        raise SystemExit(f'MP4 루트 없음: {media} — --media 또는 VFA_LOCAL_DEMO_MEDIA 로 지정하세요.')
    for head in HEADS:
        catalog = json.loads((ROOT / f'demo/local_model_based/{head}/catalog.json').read_text())
        for case in catalog[:5]:
            rel_mp4 = case['video']
            source = media / rel_mp4
            if not source.is_file():
                print(f'MISS: {source}')
                continue
            rank = case['rank']
            output = ROOT / f'assets/demos/local_{head}_demo_{rank:02d}.gif'
            total = duration(source)
            start = max(0.0, total - 9.0)
            convert(source, output, start)
            print(f'OK: {output.relative_to(ROOT)} ({total:.1f}s → start {start:.1f})')


if __name__ == '__main__':
    main()
