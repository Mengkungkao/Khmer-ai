"""Shared vocabulary and tokenizer base class for the Khmer Tokenizer Lab
(README.md Project 4 / section 10).

Every tokenizer variant (character, grapheme, syllable, BPE) shares the
same vocabulary bookkeeping, `encode`/`decode`, and frequency-based
`train`; they only differ in how `tokenize()` splits raw text into
string pieces before those pieces get mapped to ids.
"""

from __future__ import annotations

from collections import Counter

PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, BOS, EOS)


class Vocabulary:
    def __init__(self, tokens: list[str] | None = None):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        for t in SPECIAL_TOKENS:
            self._add(t)
        for t in tokens or []:
            self._add(t)

    def _add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def add(self, token: str) -> int:
        return self._add(token)

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[UNK])

    def decode_id(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return UNK

    def __len__(self) -> int:
        return len(self.id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def to_dict(self) -> dict:
        return {"id_to_token": self.id_to_token}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        vocab = cls()
        for t in data["id_to_token"]:
            vocab._add(t)
        return vocab


class BaseTokenizer:
    """Subclasses implement `tokenize(text) -> list[str]`."""

    def __init__(self) -> None:
        self.vocab = Vocabulary()

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def train(self, corpus: list[str], vocab_size: int | None = None) -> None:
        """Build the vocabulary from the most frequent tokens in `corpus`."""
        counts: Counter[str] = Counter()
        for text in corpus:
            counts.update(self.tokenize(text))
        limit = None if vocab_size is None else max(vocab_size - len(SPECIAL_TOKENS), 0)
        most_common = counts.most_common(limit)
        self.vocab = Vocabulary([token for token, _ in most_common])

    def encode(self, text: str) -> list[int]:
        return [self.vocab.encode_token(t) for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        tokens = (self.vocab.decode_id(i) for i in ids)
        return "".join(t for t in tokens if t not in SPECIAL_TOKENS)

    def unknown_rate(self, text: str) -> float:
        """Fraction of tokens in `text` that fall outside the trained vocab."""
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        unk_id = self.vocab.encode_token(UNK)
        unknown = sum(1 for t in tokens if self.vocab.encode_token(t) == unk_id and t != UNK)
        return unknown / len(tokens)
