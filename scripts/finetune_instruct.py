#!/usr/bin/env python3
"""Fine-tune a pretrained KhmerGPT to answer questions (Project 8).

    python3 scripts/finetune_instruct.py \
        --checkpoint data/checkpoints/khmergpt0-kmwiki.npz \
        --instructions data/instructions/khmer_instructions.jsonl \
        --save data/checkpoints/khmergpt0-instruct.npz

Starts from a model that already knows Khmer and teaches it the SHAPE of
answering: given a question, produce an answer and then stop.

Two things this cannot do, worth knowing before reading the output:

  It does not add knowledge. A model that never learned a fact during
  pretraining will now state a wrong one fluently and confidently, which
  is harder to spot than rambling.

  It does not generalize from a handful of examples. With ten pairs the
  model reproduces those ten answers; asked anything else it returns the
  nearest one it memorized. Generalizing the shape of answering takes
  roughly a thousand examples, and useful breadth many more.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khmer_language.models.from_scratch.checkpoint import (  # noqa: E402
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from khmer_language.training.instruction import (  # noqa: E402
    answer,
    finetune,
    load_instructions,
)

DEFAULT_INSTRUCTIONS = "data/instructions/khmer_instructions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/checkpoints/khmergpt0-kmwiki.npz")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--save", default="data/checkpoints/khmergpt0-instruct.npz")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = load_instructions(args.instructions)
    if not examples:
        print(f"no instruction examples found in {args.instructions}", file=sys.stderr)
        print("add some - see the template in that file for the format.", file=sys.stderr)
        return 1

    print(f"{len(examples)} instruction examples from {args.instructions}")
    if len(examples) < 100:
        print(
            f"  warning: {len(examples)} examples will be MEMORIZED, not generalized.\n"
            "  Expect exact answers to these questions and nonsense for anything else."
        )

    try:
        model, tokenizer = load_checkpoint(args.checkpoint)
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"model: {model.num_parameters():,} parameters, vocab {len(tokenizer.vocab):,}\n")

    print("BEFORE fine-tuning:")
    for example in examples[:2]:
        before = answer(model, tokenizer, example.instruction, example.input, 20,
                        rng=np.random.default_rng(0))
        print(f"  Q {example.instruction}")
        print(f"  A {before!r}")

    losses = finetune(
        model, examples, tokenizer,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        log_every=max(1, args.steps // 10),
    )
    print(f"\nloss {losses[0]:.4f} -> {losses[-1]:.4f}\n")

    print("AFTER fine-tuning:")
    correct = 0
    for example in examples:
        got = answer(model, tokenizer, example.instruction, example.input, 60, temperature=0.0)
        exact = got == example.output
        correct += exact
        print(f"  Q {example.instruction}")
        print(f"  A {got!r}{'  [exact]' if exact else ''}")
    print(f"\nreproduced {correct}/{len(examples)} training answers exactly")
    print("(on TRAINING questions - this measures memorization, not ability)")

    saved = save_checkpoint(args.save, model, tokenizer)
    print(f"\nsaved to {saved}")
    print(f"  chat with it:  python3 scripts/chat.py --checkpoint {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
