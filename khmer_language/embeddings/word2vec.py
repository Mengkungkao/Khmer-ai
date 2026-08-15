"""Word2Vec Skip-Gram with negative sampling, from scratch (Project 5 /
README section 11).

Implemented directly with NumPy rather than a library, since the point of
this project is understanding each component. The whole model is two
matrices:

    W_in  (vocab_size x dim)  - the "input"/center embeddings, the ones
                                you actually keep and use afterwards
    W_out (vocab_size x dim)  - the "output"/context embeddings, used
                                only during training and then discarded

Training objective (skip-gram with negative sampling, Mikolov et al.
2013): for a real (center, context) pair, push their dot product up; for
`negative_samples` fake pairs drawn from the noise distribution, push
theirs down. Concretely, with sigma = logistic sigmoid:

    loss = -log sigma(v_c . v_o) - sum_k log sigma(-v_c . v_nk)

Negative samples are drawn from the unigram distribution raised to the
3/4 power, the standard choice from the paper: it damps very frequent
tokens without ignoring them the way a uniform distribution would.

Tokenization is pluggable - pass any tokenizer from
`khmer_language.tokenizer`, so this can train over Khmer graphemes,
syllables, or BPE subwords rather than assuming space-separated words
(which Khmer text does not have).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..tokenizer.base import BaseTokenizer, Vocabulary


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clip before exp: without this, large-magnitude dot products early in
    # training overflow float64 and produce nan gradients.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class Word2Vec:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        dim: int = 64,
        window: int = 2,
        negative_samples: int = 5,
        min_count: int = 1,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.dim = dim
        self.window = window
        self.negative_samples = negative_samples
        self.min_count = min_count
        self.rng = np.random.default_rng(seed)

        self.vocab = Vocabulary()
        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None
        self._noise_probs: np.ndarray | None = None

    def build_vocab(self, corpus: list[str]) -> None:
        counts: Counter[str] = Counter()
        for text in corpus:
            counts.update(self.tokenizer.tokenize(text))

        kept = [(t, c) for t, c in counts.most_common() if c >= self.min_count]
        self.vocab = Vocabulary([t for t, _ in kept])

        # Noise distribution over the full vocab (special tokens get 0
        # probability so they are never drawn as negative samples).
        freqs = np.zeros(len(self.vocab), dtype=np.float64)
        for token, count in kept:
            freqs[self.vocab.token_to_id[token]] = count
        smoothed = freqs**0.75
        total = smoothed.sum()
        self._noise_probs = smoothed / total if total > 0 else None

        scale = 1.0 / np.sqrt(self.dim)
        self.W_in = self.rng.normal(0.0, scale, size=(len(self.vocab), self.dim))
        self.W_out = np.zeros((len(self.vocab), self.dim))

    def _training_pairs(self, corpus: list[str]) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        unk_id = self.vocab.token_to_id["<UNK>"]
        for text in corpus:
            ids = [self.vocab.encode_token(t) for t in self.tokenizer.tokenize(text)]
            ids = [i for i in ids if i != unk_id]
            for pos, center in enumerate(ids):
                start = max(0, pos - self.window)
                end = min(len(ids), pos + self.window + 1)
                for ctx_pos in range(start, end):
                    if ctx_pos != pos:
                        pairs.append((center, ids[ctx_pos]))
        return pairs

    def train(
        self,
        corpus: list[str],
        epochs: int = 5,
        learning_rate: float = 0.05,
    ) -> list[float]:
        """Train and return the mean loss per epoch (should decrease)."""
        if self.W_in is None:
            self.build_vocab(corpus)
        assert self.W_in is not None and self.W_out is not None

        pairs = self._training_pairs(corpus)
        if not pairs or self._noise_probs is None:
            return []

        losses = []
        for _ in range(epochs):
            self.rng.shuffle(pairs)
            epoch_loss = 0.0
            for center, context in pairs:
                negatives = self.rng.choice(
                    len(self.vocab), size=self.negative_samples, p=self._noise_probs
                )
                targets = np.concatenate(([context], negatives))
                labels = np.zeros(len(targets))
                labels[0] = 1.0

                v_center = self.W_in[center]
                v_targets = self.W_out[targets]

                scores = v_targets @ v_center
                preds = _sigmoid(scores)
                epoch_loss += float(
                    -np.log(np.clip(np.where(labels == 1, preds, 1 - preds), 1e-10, None)).sum()
                )

                grad = preds - labels
                grad_center = grad @ v_targets
                grad_targets = np.outer(grad, v_center)

                self.W_out[targets] -= learning_rate * grad_targets
                self.W_in[center] -= learning_rate * grad_center

            losses.append(epoch_loss / len(pairs))
        return losses

    def get_vector(self, token: str) -> np.ndarray | None:
        if self.W_in is None or token not in self.vocab:
            return None
        return self.W_in[self.vocab.token_to_id[token]]

    def similarity(self, a: str, b: str) -> float | None:
        va, vb = self.get_vector(a), self.get_vector(b)
        if va is None or vb is None:
            return None
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(va @ vb / denom) if denom else 0.0

    def most_similar(self, token: str, top_n: int = 5) -> list[tuple[str, float]]:
        vec = self.get_vector(token)
        if vec is None or self.W_in is None:
            return []
        norms = np.linalg.norm(self.W_in, axis=1) * np.linalg.norm(vec)
        with np.errstate(invalid="ignore", divide="ignore"):
            sims = np.where(norms > 0, self.W_in @ vec / np.where(norms > 0, norms, 1), 0.0)

        skip = {self.vocab.token_to_id[token]} | {
            self.vocab.token_to_id[t] for t in ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
        }
        ranked = sorted(
            ((i, s) for i, s in enumerate(sims) if i not in skip),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(self.vocab.id_to_token[i], float(s)) for i, s in ranked[:top_n]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self.W_in is not None
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "vocab": self.vocab.to_dict(),
                    "W_in": self.W_in.tolist(),
                },
                f,
                ensure_ascii=False,
            )
