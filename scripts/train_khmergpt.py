#!/usr/bin/env python3
"""Train KhmerGPT on a real cleaned corpus.

    python3 scripts/train_khmergpt.py --corpus data/cleaned/kmwiki.jsonl

Unlike `python -m khmer_language --train-demo`, which trains on eight
hand-written sentences to prove the pipeline runs, this trains on real
Khmer and reports **held-out** numbers, so the results say something
about learning rather than memorization.

One methodological detail that is easy to get wrong: the tokenizer is
fitted on the TRAINING split only. Fitting it on the whole corpus would
let validation text influence the vocabulary, so validation perplexity
would be measuring a tokenizer that had already seen the answers. It is a
small leak, but it is exactly the kind that quietly flatters results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khmer_language.corpus import read_jsonl, split_documents  # noqa: E402
from khmer_language.evaluation import analyze_output, format_report, perplexity  # noqa: E402
from khmer_language.models.from_scratch.checkpoint import save_checkpoint  # noqa: E402
from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT  # noqa: E402
from khmer_language.tokenizer import BPETokenizer, GraphemeTokenizer  # noqa: E402
from khmer_language.training import encode_corpus, train  # noqa: E402

TOKENIZERS = {"grapheme": GraphemeTokenizer, "bpe": BPETokenizer}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/cleaned/kmwiki.jsonl")
    parser.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default="grapheme")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="save trained weights to this .npz path")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"corpus not found: {corpus_path}", file=sys.stderr)
        print("build it first, e.g. from a Wikipedia dump via khmer_language.corpus.wikipedia",
              file=sys.stderr)
        return 1

    print(f"loading {corpus_path} ...", flush=True)
    documents = list(read_jsonl(corpus_path))
    if args.max_documents:
        documents = documents[: args.max_documents]
    print(f"  {len(documents):,} documents, {sum(len(d.text) for d in documents):,} chars")

    split = split_documents(documents, validation_fraction=0.05, test_fraction=0.05, seed=args.seed)
    print(f"  {split}")

    # Tokenizer sees TRAINING text only - see the module docstring.
    print(f"\nfitting {args.tokenizer} tokenizer on the training split only ...", flush=True)
    tokenizer = TOKENIZERS[args.tokenizer]()
    t0 = time.time()
    tokenizer.train([d.text for d in split.train], vocab_size=args.vocab_size)
    print(f"  vocab {len(tokenizer.vocab):,} in {time.time() - t0:.1f}s")

    print("\nencoding ...", flush=True)
    t0 = time.time()
    train_ids = encode_corpus(tokenizer, [d.text for d in split.train])
    validation_ids = encode_corpus(tokenizer, [d.text for d in split.validation])
    print(f"  train {len(train_ids):,} tokens, validation {len(validation_ids):,} tokens "
          f"({time.time() - t0:.1f}s)")

    config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        max_seq_len=args.context,
    )
    model = KhmerGPT(config, seed=args.seed)
    print(f"\nKhmerGPT: {model.num_parameters():,} parameters")
    print(f"  a random model starts near ln(vocab) = {np.log(config.vocab_size):.3f}")

    t0 = time.time()
    report = train(
        model,
        train_ids,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        seed=args.seed,
        log_every=max(1, args.steps // 20),
        validation_data=validation_ids if len(validation_ids) > args.seq_len + 2 else None,
        eval_every=max(1, args.steps // 20),
    )
    print(f"\ntrained in {time.time() - t0:.1f}s")
    print(f"  train loss      {report.losses[0]:.3f} -> {report.final_loss:.3f}")
    if report.validation_losses:
        first = report.validation_losses[0][1]
        print(f"  validation loss {first:.3f} -> {report.final_validation_loss:.3f}")
        print(f"  overfitting: {report.overfitting}")

    if len(validation_ids) > args.seq_len + 2:
        result = perplexity(model, validation_ids, seq_len=args.seq_len)
        print(f"\nheld-out perplexity: {result.perplexity:.2f} over {result.num_tokens:,} tokens")
        print(f"  (uniform guessing over {config.vocab_size:,} tokens would be "
              f"{config.vocab_size:,})")

    print("\nsamples:")
    for temperature in (0.7, 1.0):
        ids = model.generate(
            list(train_ids[:5]), max_new_tokens=60, temperature=temperature,
            rng=np.random.default_rng(args.seed),
        )
        text = tokenizer.decode(ids)
        print(f"\n  [temperature {temperature}] {text}")
        print("  " + format_report(analyze_output(text)).replace("\n", "\n  "))

    if args.save:
        # Saves the tokenizer alongside the weights - without it the token
        # ids the model was trained on cannot be reproduced, so the
        # checkpoint would load but be unusable.
        saved = save_checkpoint(args.save, model, tokenizer)
        print(f"\nsaved model + tokenizer to {saved}")
        print(f"  chat with it:  python3 scripts/chat.py --checkpoint {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
