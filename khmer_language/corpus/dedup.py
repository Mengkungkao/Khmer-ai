"""Deduplication (README section 6, "Deduplication").

Two levels, because they catch different things:

**Exact** - hash the normalized text. Catches byte-identical reposts,
which are extremely common when scraping (mirrors, syndicated news).

**Near-duplicate** - MinHash over grapheme n-gram shingles. Catches the
same article with a different header, a changed date, or one edited
paragraph. Exact hashing misses all of those, and they matter: training
on many near-copies of one document over-weights it and wastes compute.

MinHash estimates Jaccard similarity |A ∩ B| / |A ∪ B| without comparing
every pair. The trick: for a random hash function h, the probability that
min(h(A)) == min(h(B)) is exactly the Jaccard similarity. Using
`num_hashes` independent functions and counting agreements estimates it,
turning an O(n^2) all-pairs comparison into a signature comparison, and
allowing bucketing so most pairs are never compared at all.

Shingles are **grapheme** n-grams, not codepoint n-grams, so the units
match what a Khmer reader perceives (the same reasoning as the Grapheme
Error Rate in `evaluation/metrics.py`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..unicode.grapheme import grapheme_strings
from ..unicode.normalizer import normalize, strip_zero_width
from .document import Document

_MERSENNE_PRIME = (1 << 61) - 1  # large prime for the hash family
_MAX_HASH = (1 << 32) - 1


def content_hash(text: str) -> str:
    """Stable hash of normalized text, for exact-duplicate detection.

    Normalizing first (NFC, whitespace collapse, zero-width stripped)
    means documents differing only in invisible characters or spacing are
    correctly treated as identical - a real issue in Khmer text, where
    ZWSP placement varies between sources.
    """
    canonical = strip_zero_width(normalize(text))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> set[str]:
    """Grapheme n-grams of `text`."""
    units = grapheme_strings(strip_zero_width(normalize(text)))
    if len(units) < n:
        return {"".join(units)} if units else set()
    return {"".join(units[i : i + n]) for i in range(len(units) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MinHasher:
    """MinHash signatures using a family of random affine hash functions."""

    def __init__(self, num_hashes: int = 64, seed: int = 0):
        self.num_hashes = num_hashes
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _MERSENNE_PRIME, size=num_hashes, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE_PRIME, size=num_hashes, dtype=np.uint64)

    def signature(self, text: str, n: int = 5) -> np.ndarray:
        items = shingles(text, n)
        if not items:
            return np.full(self.num_hashes, _MAX_HASH, dtype=np.uint64)

        base = np.array(
            [int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) for s in items],
            dtype=np.uint64,
        )
        # (a*x + b) mod p, minimum over all shingles, for each hash function
        hashed = (self.a[:, None] * base[None, :] + self.b[:, None]) % _MERSENNE_PRIME
        return hashed.min(axis=1)

    def similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))

    def bands(self, signature: np.ndarray, num_bands: int) -> list[bytes]:
        """Split a signature into `num_bands` band keys for LSH bucketing.

        Two documents are treated as *candidates* if any band matches
        exactly. Probability of at least one band colliding is
        1 - (1 - s^r)^b for similarity s with b bands of r rows, an
        S-curve that rises sharply near s = (1/b)^(1/r). Choosing bands so
        that knee sits just below the similarity threshold means genuine
        near-duplicates almost always become candidates, while unrelated
        pairs almost never do.
        """
        rows = len(signature) // num_bands
        return [signature[i * rows : (i + 1) * rows].tobytes() for i in range(num_bands)]


@dataclass(frozen=True)
class DedupResult:
    kept: list[Document]
    exact_duplicates: int
    near_duplicates: int

    @property
    def removed(self) -> int:
        return self.exact_duplicates + self.near_duplicates


def deduplicate(
    documents: list[Document],
    near_duplicate_threshold: float = 0.8,
    num_hashes: int = 64,
    shingle_size: int = 5,
    num_bands: int = 8,
    seed: int = 0,
) -> DedupResult:
    """Remove exact and near-duplicate documents, keeping the first seen.

    Near-duplicate search uses LSH banding rather than comparing every
    document against every other. The naive all-pairs version is O(n^2)
    and becomes unusable on a real corpus - at 10k documents it is 50M
    signature comparisons. Banding buckets documents by band key and only
    compares within a bucket, which is effectively linear while still
    confirming every candidate with the real similarity, so results stay
    exact rather than approximate for the pairs it does examine.

    Set `near_duplicate_threshold` to 1.0 to skip near-duplicate detection
    entirely (exact hashing only), which is faster still.
    """
    kept: list[Document] = []
    seen_hashes: set[str] = set()
    exact = near = 0

    do_near = near_duplicate_threshold < 1.0
    hasher = MinHasher(num_hashes=num_hashes, seed=seed) if do_near else None
    buckets: dict[tuple[int, bytes], list[np.ndarray]] = {}

    for doc in documents:
        digest = content_hash(doc.text)
        if digest in seen_hashes:
            exact += 1
            continue

        if do_near:
            assert hasher is not None
            signature = hasher.signature(doc.text, n=shingle_size)
            band_keys = [(i, b) for i, b in enumerate(hasher.bands(signature, num_bands))]

            candidates: list[np.ndarray] = []
            seen_ids: set[int] = set()
            for key in band_keys:
                for other in buckets.get(key, ()):
                    if id(other) not in seen_ids:
                        seen_ids.add(id(other))
                        candidates.append(other)

            if any(hasher.similarity(signature, c) >= near_duplicate_threshold for c in candidates):
                near += 1
                continue

            for key in band_keys:
                buckets.setdefault(key, []).append(signature)

        seen_hashes.add(digest)
        kept.append(doc)

    return DedupResult(kept=kept, exact_duplicates=exact, near_duplicates=near)
