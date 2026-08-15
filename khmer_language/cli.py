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
    args = parser.parse_args(argv)

    if args.export_db is not None:
        export_json(args.export_db)
        print(f"wrote {args.export_db}")
        return 0

    text = " ".join(args.text) if args.text else sys.stdin.read().strip()
    if not text:
        print("usage: python -m khmer_language <khmer text>", file=sys.stderr)
        return 1

    print(format_analysis(analyze(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
