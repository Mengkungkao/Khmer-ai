"""Tokenizer comparison harness (README.md section 10: "compare vocabulary
size, sequence length, compression ratio, unknown token rate...").

`compare()` trains each given tokenizer on the same corpus and reports
those metrics side by side. There is no real Khmer corpus in this repo
yet (README Project 3), so `SAMPLE_CORPUS` is a small hand-written set of
sentences for demonstration/testing only - swap in a real corpus once
one exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseTokenizer

SAMPLE_CORPUS: tuple[str, ...] = (
    "កម្ពុជាស្ថិតនៅអាស៊ីអាគ្នេយ៍។",
    "រាជធានីរបស់ប្រទេសកម្ពុជាគឺទីក្រុងភ្នំពេញ។",
    "ខ្ញុំចង់ទៅភ្នំពេញនៅថ្ងៃស្អែក។",
    "តើអ្នកសុខសប្បាយជាទេ?",
    "អង្គរវត្តជាកេរ្តិ៍ដំណែលពិភពលោក។",
    "ភាសាខ្មែរមានអក្សរច្រើនណាស់។",
    "ខ្ញុំចូលចិត្តញ៉ាំបាយនិងសម្លរម្ជូរ។",
    "ប្រទេសថៃ វៀតណាម និងឡាវ នៅជិតកម្ពុជា។",
)


@dataclass(frozen=True)
class TokenizerStats:
    name: str
    vocab_size: int
    avg_sequence_length: float
    compression_ratio: float  # avg chars per token; higher = shorter sequences
    unknown_rate: float


def compare(
    tokenizers: dict[str, BaseTokenizer],
    train_corpus: list[str],
    eval_corpus: list[str] | None = None,
    vocab_size: int | None = None,
) -> list[TokenizerStats]:
    eval_corpus = list(eval_corpus or train_corpus)
    stats = []
    for name, tokenizer in tokenizers.items():
        if vocab_size is not None:
            tokenizer.train(list(train_corpus), vocab_size=vocab_size)
        else:
            tokenizer.train(list(train_corpus))

        lengths = [len(tokenizer.tokenize(text)) for text in eval_corpus]
        total_tokens = sum(lengths)
        total_chars = sum(len(text) for text in eval_corpus)
        unk_rates = [tokenizer.unknown_rate(text) for text in eval_corpus]

        stats.append(
            TokenizerStats(
                name=name,
                vocab_size=len(tokenizer.vocab),
                avg_sequence_length=total_tokens / len(eval_corpus) if eval_corpus else 0.0,
                compression_ratio=total_chars / total_tokens if total_tokens else 0.0,
                unknown_rate=sum(unk_rates) / len(unk_rates) if unk_rates else 0.0,
            )
        )
    return stats


def format_comparison(stats: list[TokenizerStats]) -> str:
    header = f"{'tokenizer':<12} {'vocab':>7} {'avg_len':>9} {'chars/tok':>10} {'unk_rate':>9}"
    lines = [header, "-" * len(header)]
    for s in stats:
        lines.append(
            f"{s.name:<12} {s.vocab_size:>7} {s.avg_sequence_length:>9.2f} "
            f"{s.compression_ratio:>10.2f} {s.unknown_rate:>9.2%}"
        )
    return "\n".join(lines)
