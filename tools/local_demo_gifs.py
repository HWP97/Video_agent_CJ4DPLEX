"""로컬 모델 Demo의 첫 5개를 각 head별로 GIF로 만든다.
API와 동일한 ffmpeg 설정 사용(fps=6, scale=800, 7.5초 꼬리 발췌).
입력 MP4는 support/submission_cleanup_20261112/demo_media/ 에 있다."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = Path('/mnt/RG2/HWP_CJ4DPLEX_AI_support/submission_cleanup_20261112/demo_media')
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
    for head in HEADS:
        catalog = json.loads((ROOT / f'demo/local_model_based/{head}/catalog.json').read_text())
        for case in catalog[:5]:
            rel_mp4 = case['video']
            source = MEDIA / rel_mp4
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
