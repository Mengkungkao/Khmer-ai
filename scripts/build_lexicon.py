#!/usr/bin/env python3
"""Build the Khmer segmentation lexicon (README section 8).

    python3 scripts/build_lexicon.py

Extracts Khmer headwords and parts of speech from the Kaikki machine-
readable Wiktionary export, then attaches corpus frequencies so the
segmenter can prefer common words over rare ones.

**Licence.** Wiktionary is CC BY-SA, the same licence as the Khmer
Wikipedia corpus this project already uses, so it introduces no new
restriction and the result stays redistributable. Attribution is written
into the lexicon file itself rather than left to a README that can drift
away from the data.

Only the word form and part of speech are taken - the lexical inventory a
segmenter needs. Definitions, etymologies, usage notes and example
sentences are deliberately not extracted: they are the substance of the
dictionary, they are not needed to find word boundaries, and copying them
wholesale would be redistributing someone else's work rather than
building on it.

A note on what was NOT used: khPOS is a manually word-segmented Khmer
corpus and would be excellent training data, but it is CC BY-NC-SA. The
NonCommercial and ShareAlike terms would propagate to this project, which
is a decision for the project owner rather than something to adopt
silently.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khmer_language.corpus import read_jsonl  # noqa: E402

KHMER_BLOCK = range(0x1780, 0x1800)

ATTRIBUTION = {
    "_license": "CC BY-SA 4.0",
    "_source": "Wiktionary, via the Kaikki.org machine-readable export",
    "_source_url": "https://kaikki.org/dictionary/Khmer/",
    "_extracted": "headwords and parts of speech only; no definitions or examples",
    "_note": "Redistribution permitted under CC BY-SA with attribution.",
}


def is_khmer(word: str) -> bool:
    return any(ord(c) in KHMER_BLOCK for c in word)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiktionary", default="data/raw/kaikki-khmer.jsonl")
    parser.add_argument("--corpus", default="data/cleaned/kmwiki.jsonl")
    parser.add_argument("--corpus-docs", type=int, default=2000)
    parser.add_argument("--out", default="data/lexicon/khmer_lexicon.jsonl")
    args = parser.parse_args()

    source = Path(args.wiktionary)
    if not source.exists():
        print(f"missing {source}", file=sys.stderr)
        print("download it from https://kaikki.org/dictionary/Khmer/", file=sys.stderr)
        return 1

    entries: dict[str, str] = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = record.get("word", "").strip()
            if word and is_khmer(word):
                # Keep the first part of speech seen; a word listed as both
                # noun and verb is segmented identically either way.
                entries.setdefault(word, record.get("pos", "unknown"))
    print(f"{len(entries):,} Khmer headwords from Wiktionary")

    # Corpus frequency. A dictionary alone cannot rank candidate splits:
    # segmentation is ambiguous, and the deciding evidence is which words
    # actually occur often in real text.
    counts: collections.Counter[str] = collections.Counter()
    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        vocabulary = set(entries)
        scanned = 0
        for _, doc in zip(range(args.corpus_docs), read_jsonl(corpus_path)):
            scanned += 1
            text = doc.text
            for word in vocabulary:
                found = text.count(word)
                if found:
                    counts[word] += found
        print(f"counted occurrences across {scanned:,} corpus documents")
        print(f"  {sum(1 for w in entries if counts[w]):,} headwords attested in the corpus")
    else:
        print(f"no corpus at {corpus_path}; frequencies will all be zero")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(ATTRIBUTION, ensure_ascii=False) + "\n")
        for word, pos in sorted(entries.items()):
            handle.write(
                json.dumps(
                    {"word": word, "pos": pos, "count": counts.get(word, 0)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\nwrote {out} ({len(entries):,} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
