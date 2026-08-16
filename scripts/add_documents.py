#!/usr/bin/env python3
"""Add your own Khmer text to the corpus (README section 6, "Collection").

    python3 scripts/add_documents.py mytext.txt \
        --source "Ministry of Agriculture Facebook" \
        --license "public-communication" \
        --domain news

The Wikipedia importer covers one source with one known licence. This is
for everything else: text you have collected, written, or been given.

`--license` is REQUIRED and has no default. That is deliberate. The
pipeline refuses documents without a licence, because provenance cannot
be reconstructed after the fact - once a model has trained on a document,
"were we allowed to use this?" is unanswerable unless it was recorded at
ingestion. A default of "unknown" would make it trivially easy to build a
corpus nobody can vouch for.

Blank lines separate documents by default, so a file of pasted articles
becomes one document per article.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khmer_language.corpus import (  # noqa: E402
    Document,
    read_jsonl,
    run_pipeline,
    write_jsonl,
)
from khmer_language.corpus.language_id import identify  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="UTF-8 text files to ingest")
    parser.add_argument("--source", required=True, help="where this text came from")
    parser.add_argument("--license", required=True, help="licence or usage basis (no default)")
    parser.add_argument("--domain", default="general", help="news, education, culture, ...")
    parser.add_argument("--corpus", default="data/cleaned/user_corpus.jsonl")
    parser.add_argument(
        "--split-on-blank-lines",
        action="store_true",
        default=True,
        help="treat blank lines as document separators (default)",
    )
    parser.add_argument("--whole-file", dest="split_on_blank_lines", action="store_false")
    parser.add_argument("--min-graphemes", type=int, default=30)
    args = parser.parse_args()

    incoming: list[Document] = []
    for path_name in args.files:
        path = Path(path_name)
        if not path.exists():
            print(f"no such file: {path}", file=sys.stderr)
            return 1
        raw = path.read_text(encoding="utf-8")
        chunks = (
            [c.strip() for c in raw.split("\n\n") if c.strip()]
            if args.split_on_blank_lines
            else [raw.strip()]
        )
        for i, chunk in enumerate(chunks):
            incoming.append(
                Document(
                    id=f"{path.stem}_{i:04d}",
                    text=chunk,
                    source=args.source,
                    license=args.license,
                    domain=args.domain,
                    metadata={"file": path.name},
                )
            )

    print(f"read {len(incoming)} document(s) from {len(args.files)} file(s)")
    for doc in incoming[:3]:
        print(f"  {doc.id}: {identify(doc.text).khmer_ratio:.0%} Khmer, {len(doc.text)} chars")

    result = run_pipeline(incoming, min_graphemes=args.min_graphemes)
    print(f"\n{result.stats}")
    if not result.documents:
        print("\nnothing passed the filters - see the counts above for why", file=sys.stderr)
        return 1

    # Merge with anything already collected, re-running dedup across the
    # combined set so re-adding the same text cannot duplicate it.
    corpus_path = Path(args.corpus)
    existing = list(read_jsonl(corpus_path)) if corpus_path.exists() else []
    merged = run_pipeline(existing + result.documents, min_graphemes=args.min_graphemes)

    write_jsonl(merged.documents, corpus_path)
    print(f"\ncorpus now holds {len(merged.documents)} documents at {corpus_path}")
    if existing:
        print(f"  ({len(existing)} were already there; duplicates were removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
