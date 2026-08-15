"""Pretraining loop for KhmerGPT (README section 14, stages 1-5).

Next-token prediction: given a flat stream of token ids, each training
example is a window of `seq_len` tokens as input and that same window
shifted one position as the target. Every position in the window
contributes a prediction, which is what makes causal masking worth
having.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.from_scratch.gpt import KhmerGPT
from ..models.from_scratch.optimizer import Adam, clip_grad_norm
from ..tokenizer.base import BaseTokenizer


def encode_corpus(tokenizer: BaseTokenizer, corpus: list[str]) -> np.ndarray:
    """Flatten a corpus into one stream of token ids."""
    ids: list[int] = []
    for text in corpus:
        ids.extend(tokenizer.encode(text))
    return np.array(ids, dtype=np.int64)


def make_batch(
    data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random (input, target) windows. Targets are inputs shifted
    by one, so position t predicts token t+1."""
    max_start = len(data) - seq_len - 1
    if max_start < 1:
        raise ValueError(
            f"corpus has {len(data)} tokens, too short for seq_len={seq_len}; "
            "need at least seq_len + 2"
        )
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


@dataclass
class TrainingReport:
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def improved(self) -> bool:
        return len(self.losses) > 1 and self.losses[-1] < self.losses[0]


def train(
    model: KhmerGPT,
    data: np.ndarray,
    steps: int = 200,
    batch_size: int = 8,
    seq_len: int | None = None,
    lr: float = 3e-3,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    log_every: int | None = None,
) -> TrainingReport:
    """Train `model` on a flat id stream. Returns per-step loss history."""
    seq_len = seq_len or min(model.config.max_seq_len, 32)
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), lr=lr)
    report = TrainingReport()

    for step in range(steps):
        x, y = make_batch(data, batch_size, seq_len, rng)

        optimizer.zero_grad()
        loss, dlogits = model.loss(x, y)
        model.backward(dlogits)
        grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)
        optimizer.step()

        report.losses.append(loss)
        report.grad_norms.append(grad_norm)

        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}/{steps}  loss {loss:.4f}  grad_norm {grad_norm:.3f}")

    return report
