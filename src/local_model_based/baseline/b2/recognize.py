"""관측 JPEG만 사용하는 동결 VEATIC 실추론."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/b2'

def recognize(sample_id):
    import cv2
    from PIL import Image
    import torch
    sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
    sys.path.insert(0, str(Path(__file__).parent / 'official'))
    import model as official
    manifest = json.loads((DATA / 'inputs/forecast.json').read_text())
    sample = next(row for row in manifest if row['sample_id'] == sample_id)
    last = max(frame['frame_index'] for frame in sample['frames'])
    wanted = [last - 20 + i * 5 for i in range(5)]
    frames = {frame['frame_index']: frame for frame in sample['frames']}
    pixels = []
    paths = []
    for index in wanted:
        frame = frames[index]
        assert frame['pts_s'] <= sample['cutoff_s']
        relative = Path(frame['path']).parts[-2:]
        path = DATA / 'inputs/observations' / relative[0] / relative[1]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB')) / 128.0 - 1.0
        pixels.append(cv2.resize(rgb, (640, 480)).transpose(2,0,1).astype(np.float32))
        paths.append(str(path.relative_to(ROOT)))
    torch.set_num_threads(2)
    original = official.resnet50
    official.resnet50 = lambda **kwargs: original(weights=None)
    model = official.VEATIC_baseline()
    official.resnet50 = original
    model.load_state_dict(torch.load(DATA / 'weights/one_stream_vit.pth', map_location='cpu', weights_only=True), strict=True)
    model.eval().to('cuda:0')
    tensor = torch.from_numpy(np.stack(pixels)).unsqueeze(0).to('cuda:0')
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        value = model(tensor).cpu().numpy()[0]
    torch.cuda.synchronize()
    cached = json.loads((DATA / f'local_predictions/recognition/{sample["video_id"]:03d}.json').read_text())
    expected = np.array(cached['raw_VA'][cached['frames'].index(last)])
    difference = float(np.abs(value - expected).max())
    assert difference < 1e-4, difference
    return {'sample_id': sample_id, 'raw_current_VA': value.tolist(),
            'prediction': np.broadcast_to(np.clip(value,-1,1), (4,2)).tolist(),
            'frames': paths, 'frame_indices': wanted, 'maximum_cache_difference': difference,
            'forward_seconds': time.perf_counter()-started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'paid_calls': 0, 'GT_access': False}
