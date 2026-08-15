"""KhmerGPT - a GPT-style decoder-only language model, from scratch.

README section 12 (KhmerGPT-0) and 13. Architecture:

    token ids
        -> token embedding + learned positional embedding
        -> N x TransformerBlock (pre-norm)
        -> final LayerNorm
        -> linear head -> logits over the vocabulary

Trained on the single objective P(next token | previous tokens), which is
all a language model is: causal masking (see `attention.py`) means every
position in a sequence can be trained in parallel while never seeing its
own answer.

Positional embeddings are *learned* (a plain parameter matrix) rather
than the fixed sinusoids of the 2017 paper - simpler, and what GPT-2 does.
The cost is a hard context limit: the model cannot be run on a sequence
longer than `max_seq_len`, because no position vector exists for it. That
is enforced explicitly rather than silently misbehaving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layers import Embedding, Layer, LayerNorm, Linear, Parameter, cross_entropy_loss, softmax
from .transformer import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    max_seq_len: int = 64
    expansion: int = 4

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")


class KhmerGPT(Layer):
    def __init__(self, config: GPTConfig, seed: int = 0):
        self.config = config
        rng = np.random.default_rng(seed)

        self.token_emb = Embedding(config.vocab_size, config.dim, rng)
        # Positional embeddings are indexed by a slice, not by arbitrary
        # ids, so a plain Parameter is a better fit than an Embedding.
        self.pos_emb = Parameter(
            rng.normal(0.0, 1.0 / np.sqrt(config.dim), size=(config.max_seq_len, config.dim))
        )
        self.blocks = [
            TransformerBlock(config.dim, config.num_heads, rng, expansion=config.expansion)
            for _ in range(config.num_layers)
        ]
        self.ln_f = LayerNorm(config.dim)
        self.head = Linear(config.dim, config.vocab_size, rng, bias=False)
        self._seq_len: int | None = None

    def parameters(self) -> list[Parameter]:
        params = self.token_emb.parameters() + [self.pos_emb]
        for block in self.blocks:
            params += block.parameters()
        return params + self.ln_f.parameters() + self.head.parameters()

    def num_parameters(self) -> int:
        return sum(int(np.prod(p.shape)) for p in self.parameters())

    def forward(self, ids: np.ndarray) -> np.ndarray:
        """(B, T) token ids -> (B, T, vocab) logits."""
        if ids.ndim == 1:
            ids = ids[None, :]
        _, T = ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.config.max_seq_len}; "
                "no positional embedding exists for those positions"
            )
        self._seq_len = T

        x = self.token_emb.forward(ids) + self.pos_emb.value[:T]
        for block in self.blocks:
            x = block.forward(x)
        return self.head.forward(self.ln_f.forward(x))

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        assert self._seq_len is not None, "forward() must be called before backward()"
        d = self.ln_f.backward(self.head.backward(dlogits))
        for block in reversed(self.blocks):
            d = block.backward(d)

        # The embedding sum splits the gradient: positional embeddings are
        # shared across the batch, so their gradient sums over that axis.
        self.pos_emb.grad[: self._seq_len] += d.sum(axis=0)
        self.token_emb.backward(d)
        return np.zeros(0)  # ids are integers; no gradient flows to them

    def loss(self, ids: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        logits = self.forward(ids)
        return cross_entropy_loss(logits, targets)

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Autoregressively sample continuation ids.

        `temperature` < 1 sharpens the distribution (more predictable),
        > 1 flattens it. `top_k` restricts sampling to the k most likely
        tokens, which mainly suppresses rare-token noise from an
        undertrained model.
        """
        rng = rng or np.random.default_rng()
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Keep only the last max_seq_len tokens: beyond that there is
            # no positional embedding to use.
            context = ids[-self.config.max_seq_len :]
            logits = self.forward(np.array([context]))[0, -1]

            if temperature <= 0:
                ids.append(int(np.argmax(logits)))
                continue

            logits = logits / temperature
            if top_k is not None and top_k < len(logits):
                cutoff = np.partition(logits, -top_k)[-top_k]
                logits = np.where(logits < cutoff, -np.inf, logits)

            probs = softmax(logits)
            ids.append(int(rng.choice(len(probs), p=probs)))

        return ids
