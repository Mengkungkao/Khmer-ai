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
from ..models.from_scratch.optimizer import Adam, clip_grad_norm, cosine_lr
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
    # (step, validation loss) pairs - held-out, so unlike `losses` these
    # measure generalization rather than memorization.
    validation_losses: list[tuple[int, float]] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def final_validation_loss(self) -> float:
        return self.validation_losses[-1][1] if self.validation_losses else float("nan")

    @property
    def improved(self) -> bool:
        return len(self.losses) > 1 and self.losses[-1] < self.losses[0]

    @property
    def overfitting(self) -> bool:
        """Whether validation loss has started rising while training loss
        keeps falling - the signature of memorizing rather than learning."""
        if len(self.validation_losses) < 2:
            return False
        best = min(v for _, v in self.validation_losses)
        return self.validation_losses[-1][1] > best * 1.05


def evaluate(
    model: KhmerGPT,
    data: np.ndarray,
    seq_len: int,
    batches: int = 8,
    batch_size: int = 8,
    seed: int = 12345,
) -> float:
    """Mean loss on held-out data.

    Uses a fixed seed so the same windows are sampled every time it is
    called: otherwise the validation curve would move because of sampling
    noise rather than because the model changed.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(batches):
        x, y = make_batch(data, batch_size, seq_len, rng)
        loss, _ = model.loss(x, y)
        total += loss
    return total / batches


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
    validation_data: np.ndarray | None = None,
    eval_every: int | None = None,
    schedule: bool = True,
    warmup_steps: int | None = None,
) -> TrainingReport:
    """Train `model` on a flat id stream. Returns per-step loss history.

    Pass `validation_data` (a held-out id stream from documents the model
    never trains on - see `corpus/split.py`) to track generalization
    alongside training loss.
    """
    seq_len = seq_len or min(model.config.max_seq_len, 32)
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), lr=lr)
    report = TrainingReport()
    eval_every = eval_every or max(1, steps // 10)
    warmup = warmup_steps if warmup_steps is not None else max(1, int(steps * 0.05))

    for step in range(steps):
        if schedule:
            optimizer.lr = cosine_lr(step, steps, lr, warmup=warmup)
        x, y = make_batch(data, batch_size, seq_len, rng)

        optimizer.zero_grad()
        loss, dlogits = model.loss(x, y)
        model.backward(dlogits)
        grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)
        optimizer.step()

        report.losses.append(loss)
        report.grad_norms.append(grad_norm)

        is_last = step + 1 == steps
        if validation_data is not None and ((step + 1) % eval_every == 0 or is_last):
            report.validation_losses.append(
                (step + 1, evaluate(model, validation_data, seq_len, batch_size=batch_size))
            )

        if log_every and ((step + 1) % log_every == 0 or is_last):
            message = f"step {step + 1:5d}/{steps}  loss {loss:.4f}  grad_norm {grad_norm:.3f}"
            if report.validation_losses:
                message += f"  val_loss {report.validation_losses[-1][1]:.4f}"
            print(message, flush=True)

    return report
