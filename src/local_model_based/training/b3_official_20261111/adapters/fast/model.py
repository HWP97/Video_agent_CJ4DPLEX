"""기존 FAST/TRM 순수 모델을 분리한 코드. 원본 모델 연산은 유지한다."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.fft import dct, idct
import torch
from torch import nn
from tokenizers import Tokenizer, models, trainers
class BoundaryError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class Codec:
    low: np.ndarray
    high: np.ndarray
    scale: float
    frequency: bool
    tokenizer: Tokenizer

    def quantized(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        norm = 2 * (values - self.low) / np.maximum(self.high - self.low, 1e-5) - 1
        transformed = dct(norm, axis=1, norm='ortho') if self.frequency else norm
        integers = np.rint(transformed * self.scale)
        clipped = float(np.mean((integers < -128) | (integers > 127)))
        return np.clip(integers, -128, 127).astype(int), clipped

    def encode(self, values: np.ndarray) -> list[list[int]]:
        integers, _ = self.quantized(values)
        texts = [''.join(chr(int(q) + 384) for q in row.ravel()) for row in integers]
        return [r.ids for r in self.tokenizer.encode_batch(texts)]

    def decode(self, tokens: list[list[int]]) -> np.ndarray:
        coefficients = []
        for row in tokens:
            text = ''.join(self.tokenizer.id_to_token(int(t)) for t in row if t >= 4)
            if len(text) != 24:
                raise BoundaryError('invalid_coefficient_count', str(len(text)))
            coefficients.append(np.array([ord(ch) - 384 for ch in text]).reshape(12, 2) / self.scale)
        values = np.stack(coefficients)
        norm = idct(values, axis=1, norm='ortho') if self.frequency else values
        return (norm + 1) / 2 * np.maximum(self.high - self.low, 1e-5) + self.low

    def persist(self, stem: Path) -> None:
        self.tokenizer.save(str(stem.with_suffix('.json')))
        np.savez(stem.with_suffix('.npz'), low=self.low, high=self.high, scale=self.scale, frequency=self.frequency)

    @staticmethod
    def restore(stem: Path) -> Codec:
        with np.load(stem.with_suffix('.npz')) as values:
            return Codec(values['low'], values['high'], float(values['scale']), bool(values['frequency']), Tokenizer.from_file(str(stem.with_suffix('.json'))))


def fit_codec(values: np.ndarray, scale: float, frequency: bool) -> Codec:
    low, high = np.quantile(values, [.01, .99], axis=(0, 1))
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    codec = Codec(low, high, scale, frequency, tokenizer)
    integers, _ = codec.quantized(values)
    texts = [''.join(chr(int(q) + 384) for q in row.ravel()) for row in integers]
    trainer = trainers.BpeTrainer(vocab_size=384, min_frequency=2, show_progress=False, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'], initial_alphabet=[chr(i + 256) for i in range(256)])
    tokenizer.train_from_iterator(texts, trainer)
    return codec


class Predictor(nn.Module):
    def __init__(self, inputs: int, kind: str, vocabulary: int = 384) -> None:
        super().__init__()
        self.kind = kind
        self.encoder = nn.Sequential(nn.Linear(inputs, 128), nn.Tanh())
        if kind.endswith('bpe'):
            self.embedding = nn.Embedding(vocabulary, 64)
            self.gru = nn.GRU(64, 128, batch_first=True)
            self.tokens = nn.Linear(128, vocabulary)
        else:
            self.block = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.LayerNorm(128))
            self.head = nn.Linear(128, 24)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def continuous(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        x = self.encoder(inputs)
        match self.kind:
            case 'continuous':
                return [self.head(self.block(x))]
            case 'one_iteration':
                return [self.head(self.block(x))]
            case 'single_state':
                z = torch.zeros_like(x)
                for _ in range(12):
                    z = self.block(x + z)
                return [self.head(z)]
            case 'two_state_deep' | 'two_state_last':
                y, z = torch.zeros_like(x), torch.zeros_like(x)
                outputs = []
                for _ in range(2):
                    with torch.no_grad():
                        for _ in range(2):
                            z = self.block(x + y + z)
                        y = self.block(y + z)
                    for _ in range(2):
                        z = self.block(x + y + z)
                    y = self.block(y + z)
                    outputs.append(self.head(y))
                    y, z = y.detach(), z.detach()
                return outputs
            case unexpected:
                raise BoundaryError('not_continuous_kind', unexpected)

    def teacher(self, inputs: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(inputs)[None]
        states, _ = self.gru(self.embedding(prefix), hidden)
        return self.tokens(states)

    def greedy(self, inputs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        hidden = self.encoder(inputs)[None]
        current = torch.ones((len(inputs), 1), dtype=torch.long, device=inputs.device)
        remaining = torch.full((len(inputs),), 24, device=inputs.device, dtype=torch.long)
        output = [[] for _ in range(len(inputs))]
        for _ in range(24):
            states, hidden = self.gru(self.embedding(current), hidden)
            logits = self.tokens(states[:, 0])
            valid = (lengths[None] > 0) & (lengths[None] <= remaining[:, None])
            logits = logits.masked_fill(~valid, -torch.inf)
            selected = logits.argmax(-1)
            selected = torch.where(remaining > 0, selected, torch.zeros_like(selected))
            for row, token, left in zip(output, selected.tolist(), remaining.tolist()):
                if left:
                    row.append(token)
            remaining -= lengths[selected]
            current = selected[:, None]
            if not torch.any(remaining):
                break
        assert not torch.any(remaining)
        return output


def predict(net: Predictor, x: np.ndarray, codec: Codec | None) -> np.ndarray:
    net.eval()
    outputs = []
    if codec is not None:
        lengths = torch.tensor([len(codec.tokenizer.id_to_token(i)) if i >= 4 else 0 for i in range(codec.tokenizer.get_vocab_size())], device='cuda')
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            tensor = torch.as_tensor(x[start:start + 256], dtype=torch.float32, device='cuda')
            if codec is None:
                outputs.append(net.continuous(tensor)[-1].cpu().numpy().reshape(-1, 12, 2))
            else:
                outputs.append(codec.decode(net.greedy(tensor, lengths)))
    result = np.concatenate(outputs)
    assert np.isfinite(result).all()
    return result
