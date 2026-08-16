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
        UnigramTokenizer,
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
        "unigram": UnigramTokenizer(),
    }
    stats = compare(tokenizers, list(SAMPLE_CORPUS), vocab_size=vocab_size)
    print(format_comparison(stats))
    return 0


def _run_train_demo(steps: int) -> int:
    """Train KhmerGPT-0 on the sample corpus and show what it generates."""
    import numpy as np

    from .models.from_scratch.gpt import GPTConfig, KhmerGPT
    from .tokenizer import SAMPLE_CORPUS, GraphemeTokenizer
    from .training import encode_corpus, train
    from .unicode.validator import is_valid

    tokenizer = GraphemeTokenizer()
    tokenizer.train(list(SAMPLE_CORPUS))
    data = encode_corpus(tokenizer, list(SAMPLE_CORPUS))

    config = GPTConfig(vocab_size=len(tokenizer.vocab), dim=64, num_layers=2, num_heads=2, max_seq_len=32)
    model = KhmerGPT(config, seed=0)

    print(f"corpus:  {len(data)} tokens, vocab {len(tokenizer.vocab)} (placeholder SAMPLE_CORPUS)")
    print(f"model:   KhmerGPT-0, {model.num_parameters():,} parameters")
    print(f"a random-init model should start near ln(vocab) = {np.log(len(tokenizer.vocab)):.3f}\n")

    report = train(model, data, steps=steps, batch_size=8, seq_len=16, lr=3e-3, seed=0,
                   log_every=max(1, steps // 5))

    print(f"\nloss {report.losses[0]:.3f} -> {report.final_loss:.3f}")

    from .evaluation import analyze_output, format_report, perplexity

    ppl = perplexity(model, data, seq_len=16)
    print(f"perplexity on training data: {ppl.perplexity:.2f} over {ppl.num_tokens} tokens")
    print("(this is training data, so it measures memorization, not generalization)")

    sample = tokenizer.decode(
        model.generate(list(data[:3]), max_new_tokens=25, temperature=0.8,
                       rng=np.random.default_rng(7))
    )
    print()
    print(format_report(analyze_output(sample)))
    return 0


def _run_benchmark(cases_path: str | None) -> int:
    """Score the repo's own components against the authored benchmark."""
    from .evaluation import (
        DEFAULT_CASES_PATH,
        STRUCTURAL_CASES,
        case_counts,
        format_benchmark,
        load_cases,
        run_benchmark,
    )
    from .unicode.normalizer import normalize
    from .unicode.transliterator import transliterate

    authored = load_cases(cases_path)
    path = cases_path or DEFAULT_CASES_PATH
    print(f"authored cases: {len(authored)} from {path}")
    if authored:
        print(f"  by category: {case_counts(authored)}")
    else:
        print("  (none yet - add them to data/benchmark/cases.jsonl)")

    structural = {c.input: c for c in STRUCTURAL_CASES}

    def predict(text: str) -> str:
        case = structural.get(text)
        if case is None:
            # No model is wired in yet; an untrained pipeline has nothing
            # meaningful to answer with, so echo and let the score show it.
            return text
        return normalize(text) if case.category == "normalization" else transliterate(text)

    print("\nStructural cases (this repo's normalizer/transliterator):")
    print(format_benchmark(run_benchmark(list(STRUCTURAL_CASES), predict)))

    if authored:
        print("\nAuthored cases (echo baseline - no trained model wired in yet):")
        print(format_benchmark(run_benchmark(authored, predict)))
        print("\nThese scores are a floor, not a model evaluation: they show what")
        print("a do-nothing baseline achieves, so a real model has something to beat.")
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
    parser.add_argument(
        "--train-demo",
        nargs="?",
        const=150,
        type=int,
        default=None,
        metavar="STEPS",
        help="train KhmerGPT-0 on the sample corpus and generate a sample",
    )
    parser.add_argument(
        "--benchmark",
        nargs="?",
        const="",
        default=None,
        metavar="CASES_PATH",
        help="run the KhmerAI benchmark (defaults to data/benchmark/cases.jsonl)",
    )
    args = parser.parse_args(argv)

    if args.benchmark is not None:
        return _run_benchmark(args.benchmark or None)

    if args.train_demo is not None:
        return _run_train_demo(args.train_demo)

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
