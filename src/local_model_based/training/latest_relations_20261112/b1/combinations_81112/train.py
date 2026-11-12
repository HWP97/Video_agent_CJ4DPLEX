"""원본 영상 분할을 유지한 관계 head 학습과 재개."""
from dataclasses import asdict
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from prepare import ROOT, sha
from models import CONFIGS, RelationHead, retrieve


def main() -> None:
    stage = sys.argv[1]
    assert stage in ('inner', 'full')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    directory = ROOT / stage
    directory.mkdir(exist_ok=True)
    if stage == 'full':
        assert (ROOT / 'selection_lock.json').exists()
    data_receipt = json.loads((ROOT / 'data/receipt.json').read_text())
    assert all(sha(ROOT / 'data' / p) == h for p, h in data_receipt['outputs'].items())
    for mode in ('sgcls', 'predcls'):
        with np.load(ROOT / 'data' / f'{mode}.npz') as data, np.load(ROOT / 'data' / f'{mode}_memory.npz') as memory:
            mask = data['split'] == 'inner_train' if stage == 'inner' else np.ones(len(data['base']), dtype=bool)
            bank_mask = memory['split'] == 'inner_train' if stage == 'inner' else np.ones(len(memory['context']), dtype=bool)
            bank = {key: memory[key][bank_mask] for key in memory.files}
            groups = data['groups'].tolist()
            tensors = {key: torch.as_tensor(data[key], device='cuda', dtype=torch.float32) for key in ('history', 'scene', 'delta', 'base', 'target')}
            bins = torch.as_tensor(data['bin'], device='cuda', dtype=torch.long)
        recalled, confidence = retrieve(tensors['history'], bins, groups, bank)
        ids = np.flatnonzero(mask)
        inputs = [tensors[key] for key in ('history', 'scene', 'delta', 'base')] + [recalled, confidence]
        support = tensors['target'][ids].sum(0).clamp_min(1)
        group_counts = Counter(groups[i] for i in ids)
        group_weight = torch.tensor([1 / group_counts[groups[i]] for i in ids])
        for config in CONFIGS:
            weights = (support.max() / support).pow(config.balance_power).clamp(1, 10)
            prefix = mode + '_' + config.name
            checkpoint, resume = directory / (prefix + '.pt'), directory / (prefix + '.resume.pt')
            receipt_path = directory / (prefix + '.json')
            signature = {'config': asdict(config), 'mode': mode, 'stage': stage,
                'data': sha(ROOT / 'data/receipt.json'),
                'sources': {str(p): sha(p) for p in (Path(__file__), ROOT / 'models.py', ROOT.parent / 'models.py', ROOT.parent.parent / 'b2/models.py')}}
            if receipt_path.exists():
                old = json.loads(receipt_path.read_text())
                assert old['signature'] == signature and sha(checkpoint) == old['checkpoint_sha256']
                continue
            torch.manual_seed(20261111)
            torch.cuda.manual_seed_all(20261111)
            generator = torch.Generator().manual_seed(20261111)
            model = RelationHead(config).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            first, trace = 0, []
            if resume.exists():
                previous = torch.load(resume, map_location='cpu', weights_only=False)
                assert previous['signature'] == signature
                model.load_state_dict(previous['state'])
                optimizer.load_state_dict(previous['optimizer'])
                generator.set_state(previous['generator'])
                torch.set_rng_state(previous['cpu_rng'])
                torch.cuda.set_rng_state(previous['cuda_rng'])
                first, trace = previous['epoch'], previous['trace']
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(first, config.epochs):
                order = torch.multinomial(group_weight, len(ids), replacement=True, generator=generator) if config.group_balance else torch.randperm(len(ids), generator=generator)
                losses = []
                for chunk in order.split(256):
                    index = ids[chunk.numpy()]
                    prediction = model(*(value[index] for value in inputs))
                    target = tensors['target'][index]
                    loss = -(weights * target * prediction.log() + (1 - target) * torch.log1p(-prediction)).mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                trace.append(float(np.mean(losses)))
                temporary = resume.with_suffix('.tmp')
                torch.save({'signature': signature, 'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'generator': generator.get_state(), 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                    'epoch': epoch + 1, 'trace': trace}, temporary)
                temporary.replace(resume)
            torch.save({'state': model.state_dict(), 'signature': signature}, checkpoint)
            receipt_path.write_text(json.dumps({'signature': signature, 'checkpoint_sha256': sha(checkpoint),
                'epochs': config.epochs, 'rows': len(ids), 'groups': len({groups[i] for i in ids}),
                'losses': trace, 'seconds_this_process': time.time() - started,
                'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'API_calls': 0}, indent=2))
            print(json.dumps({'stage': stage, 'mode': mode, 'method': config.name, 'loss': trace[-1]}), flush=True)


if __name__ == '__main__':
    main()
