#!/usr/bin/env python3
"""Choose decoding settings by measurement rather than by taste.

    python3 scripts/tune_sampling.py --checkpoint data/checkpoints/khmergpt0-kmwiki.npz

Scores each candidate on two metrics that deliberately pull in opposite
directions:

  grounding  - fraction of generated grapheme 3-grams attested in the
               reference corpus. Rewards real Khmer.
  diversity  - fraction of distinct 2-grams. Punishes degenerate
               repetition.

Neither works alone. Grounding by itself is maximized by greedy decoding,
which on this model scores 88.6% while emitting one repeated zero-width
space. Diversity by itself is maximized by sampling at random.

The target is not "maximize both" either - it is to **match real Khmer**,
which is measured here as the reference point. Real text repeats common
words, so diversity ABOVE the real-Khmer level is a symptom of randomness,
not a better sample. Candidates are therefore ranked by Euclidean
distance to the real-corpus point.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khmer_language.corpus import read_jsonl  # noqa: E402
from khmer_language.evaluation.corpus_grounding import CorpusGrounding, distinct_n  # noqa: E402
from khmer_language.models.from_scratch.checkpoint import load_checkpoint  # noqa: E402

CANDIDATES: list[tuple[str, dict]] = [
    ("greedy", dict(temperature=0.0)),
    ("t=0.5 k=10", dict(temperature=0.5, top_k=10)),
    ("t=0.6 k=20", dict(temperature=0.6, top_k=20)),
    ("t=0.6 p=0.9", dict(temperature=0.6, top_p=0.9)),
    ("t=0.7 k=40", dict(temperature=0.7, top_k=40)),
    ("t=0.7 p=0.95", dict(temperature=0.7, top_p=0.95)),
    ("t=0.8 p=0.9", dict(temperature=0.8, top_p=0.9)),
    ("t=0.8 p=0.9 rep=1.15", dict(temperature=0.8, top_p=0.9, repetition_penalty=1.15)),
    ("t=1.0 k=40", dict(temperature=1.0, top_k=40)),
]

PROMPTS = ["ភ្នំពេញ", "ប្រទេសកម្ពុជា", "ប្រវត្តិសាស្ត្រ"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/checkpoints/khmergpt0-kmwiki.npz")
    parser.add_argument("--corpus", default="data/cleaned/kmwiki.jsonl")
    parser.add_argument("--reference-docs", type=int, default=400)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    corpus = [d.text for _, d in zip(range(args.reference_docs), read_jsonl(args.corpus))]
    grounding = CorpusGrounding(corpus)
    model, tokenizer = load_checkpoint(args.checkpoint)

    real_g = float(np.mean([grounding.score(t[:200], 3).ratio for t in corpus[:40]]))
    real_d = float(np.mean([distinct_n(t[:200], 2) for t in corpus[:40]]))
    print(f"real Khmer reference:  grounding {real_g:.1%}   diversity {real_d:.1%}\n")

    results = []
    for name, options in CANDIDATES:
        grounded, diverse = [], []
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt)
            for seed in range(args.seeds):
                out = model.generate(
                    ids, max_new_tokens=args.tokens, rng=np.random.default_rng(seed), **options
                )
                text = tokenizer.decode(out[len(ids):])
                grounded.append(grounding.score(text, 3).ratio)
                diverse.append(distinct_n(text, 2))
        g, d = float(np.mean(grounded)), float(np.mean(diverse))
        results.append((name, g, d, math.dist((g, d), (real_g, real_d))))

    results.sort(key=lambda row: row[3])
    print(f"{'settings':<24}{'grounding':>10}{'diversity':>11}{'distance':>10}")
    print("-" * 55)
    for name, g, d, distance in results:
        print(f"{name:<24}{g:>9.1%}{d:>11.1%}{distance:>10.3f}")

    print(f"\nclosest to real Khmer: {results[0][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
