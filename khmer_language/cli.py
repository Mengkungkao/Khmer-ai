"""Command-line demo for the Khmer Unicode Explorer.

Usage:
    python -m khmer_language "កម្ពុជា"
    echo "កម្ពុជា" | python -m khmer_language
"""

from __future__ import annotations

import argparse
import sys

from .analyzer import analyze, format_analysis
from .unicode.codepoints import DEFAULT_EXPORT_PATH, export_json


def _run_tokenizer_comparison(vocab_size: int) -> int:
    from .tokenizer import (
        SAMPLE_CORPUS,
        BPETokenizer,
        CharacterTokenizer,
        GraphemeTokenizer,
        SyllableTokenizer,
        compare,
        format_comparison,
    )

    print(f"Tokenizer comparison on {len(SAMPLE_CORPUS)}-sentence SAMPLE_CORPUS")
    print("(placeholder corpus - no real Khmer corpus exists yet, see README Project 3)\n")
    tokenizers = {
        "character": CharacterTokenizer(),
        "grapheme": GraphemeTokenizer(),
        "syllable": SyllableTokenizer(),
        "bpe": BPETokenizer(),
    }
    stats = compare(tokenizers, list(SAMPLE_CORPUS), vocab_size=vocab_size)
    print(format_comparison(stats))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(prog="khmer_language", description=__doc__)
    parser.add_argument("text", nargs="*", help="Khmer text to analyze (reads stdin if omitted)")
    parser.add_argument(
        "--export-db",
        nargs="?",
        const=str(DEFAULT_EXPORT_PATH),
        default=None,
        metavar="PATH",
        help="write the character database to PATH as JSON and exit",
    )
    parser.add_argument(
        "--compare-tokenizers",
        nargs="?",
        const=100,
        type=int,
        default=None,
        metavar="VOCAB_SIZE",
        help="train character/grapheme/syllable/BPE tokenizers on the sample corpus and compare them",
    )
    args = parser.parse_args(argv)

    if args.export_db is not None:
        export_json(args.export_db)
        print(f"wrote {args.export_db}")
        return 0

    if args.compare_tokenizers is not None:
        return _run_tokenizer_comparison(args.compare_tokenizers)

    text = " ".join(args.text) if args.text else sys.stdin.read().strip()
    if not text:
        print("usage: python -m khmer_language <khmer text>", file=sys.stderr)
        return 1

    print(format_analysis(analyze(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
