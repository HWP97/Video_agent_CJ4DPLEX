"""확정 API Demo의 전체 GIF와 README용 짧은 GIF를 생성한다."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {
    'affect_forecasting': [1, 2, 3, 4, 5],
    'scene_graph_anticipation': [1, 2, 3, 4, 5],
    'trajectory_forecasting_reselected': [1, 2, 3, 4, 5],
}


def convert(source: Path, output: Path, start: float | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['/usr/bin/ffmpeg', '-v', 'error', '-y']
    if start is not None:
        command += ['-ss', str(start), '-t', '7.5']
    filters = ('fps=6,scale=800:-1:flags=lanczos,split[a][b];'
               '[a]palettegen=max_colors=96:stats_mode=diff[p];'
               '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
    subprocess.run(command + ['-i', str(source), '-filter_complex', filters,
                             '-an', '-loop', '0', str(output)], check=True)
    subprocess.run(['/usr/bin/ffmpeg', '-v', 'error', '-i', str(output),
                    '-f', 'null', '-'], check=True, capture_output=True)


def main() -> None:
    records = []
    for folder, ranks in SELECTION.items():
        task = folder.removesuffix('_reselected')
        catalog = json.loads((ROOT/'demo/api_based'/folder/'catalog.json').read_text())
        for index, rank in enumerate(ranks):
            case = next(row for row in catalog if row['rank'] == rank)
            source = ROOT/case['video']
            name = f'{task}_demo_{rank:02d}.gif'
            full = ROOT/'website/assets/demos'/name
            convert(source, full, None)
            short = None
            if index < 2:
                short = ROOT/'assets/readme'/name
                convert(source, short, max(0., case['observation_seconds']-1.5))
            records.append({'task': task, 'rank': rank, 'sample_id': case['sample_id'],
                            'source': case['video'], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                            'gif': str(full.relative_to(ROOT)), 'bytes': full.stat().st_size,
                            'readme_gif': str(short.relative_to(ROOT)) if short else None,
                            'metric': case['metric'], 'baseline': case['baseline_metric'],
                            'ours': case['ours_metric']})
            print(f'{task} Demo {rank}: {full.stat().st_size/1048576:.2f} MiB', flush=True)
    (ROOT/'assets/final_demo_manifest.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
