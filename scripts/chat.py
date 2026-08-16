#!/usr/bin/env python3
"""Interactive session with a trained KhmerGPT (README Project 9, first step).

    python3 scripts/chat.py --checkpoint data/checkpoints/khmergpt0-kmwiki.npz

**This completes text; it does not answer questions.** The distinction
matters and is not a limitation of this script:

A pretrained language model has learned exactly one thing - P(next token
| previous tokens) over Khmer Wikipedia. Type a question and it will
continue it the way Wikipedia text tends to continue, which is usually
*more question-like text*, not an answer. Answering requires instruction
tuning (README section 15 / Project 8): thousands of Khmer
instruction-and-response pairs teaching the model that a question should
be followed by an answer. That dataset does not exist yet, so neither
does that behaviour.

So use this to see what the model actually learned about Khmer: give it
the start of a sentence and watch how it continues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khmer_language.evaluation import analyze_output, format_report  # noqa: E402
from khmer_language.evaluation.reference_translation import ReferenceTranslator  # noqa: E402
from khmer_language.models.from_scratch.checkpoint import (  # noqa: E402
    CheckpointError,
    load_checkpoint,
)

BANNER = """\
KhmerGPT interactive session
────────────────────────────────────────────────────────────────────
This model COMPLETES Khmer text. It does not answer questions - it was
never instruction-tuned (that is Project 8, and needs a Khmer
instruction dataset that does not exist yet).

Try giving it the beginning of a sentence, for example:
  ប្រទេសកម្ពុជា
  ភ្នំពេញជា
  ប្រវត្តិសាស្ត្រ

Commands:  /temp <n>   sampling temperature (lower = safer, higher = wilder)
           /tokens <n> how many tokens to generate
           /analyze    toggle the structural quality report
           /translate  toggle a rough English gloss (see caveat below)
           /quit       exit
────────────────────────────────────────────────────────────────────"""

TRANSLATE_CAVEAT = """\
  note: machine translation always returns something plausible, so a
  readable gloss does NOT mean the Khmer was correct. Measured on this
  project, real Khmer, model output and pure nonsense all came back with
  the same 0.85 confidence score. Use it to get the gist, not as a grader."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/checkpoints/khmergpt0-kmwiki.npz")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--analyze", action="store_true", help="show the quality report each turn")
    parser.add_argument("--translate", action="store_true", help="show a rough English gloss")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--prompt", default=None, help="run one prompt and exit (non-interactive)")
    args = parser.parse_args()

    try:
        model, tokenizer = load_checkpoint(args.checkpoint)
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("\nTrain one first:", file=sys.stderr)
        print("  python3 scripts/train_khmergpt.py --corpus data/cleaned/kmwiki.jsonl "
              "--save data/checkpoints/khmergpt0-kmwiki.npz", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    temperature = args.temperature
    tokens = args.tokens
    show_analysis = args.analyze
    show_translation = args.translate
    translator = ReferenceTranslator()

    def respond(prompt: str) -> str:
        ids = tokenizer.encode(prompt)
        if not ids:
            return ""
        generated = model.generate(
            ids,
            max_new_tokens=tokens,
            temperature=temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            rng=rng,
        )
        # Return only the continuation, so it is obvious what the model added.
        return tokenizer.decode(generated[len(ids):])

    def gloss(text: str) -> None:
        result = translator.translate(text)
        if result is None:
            print("  [gloss unavailable - translation service unreachable]")
        else:
            print(f"  [gloss] {result.translated}")
        print(TRANSLATE_CAVEAT)

    if args.prompt is not None:
        continuation = respond(args.prompt)
        print(f"{args.prompt}{continuation}")
        if show_analysis:
            print(format_report(analyze_output(continuation)))
        if show_translation:
            gloss(args.prompt + continuation)
        return 0

    print(BANNER)
    print(f"model: {model.num_parameters():,} parameters, vocab {len(tokenizer.vocab):,}")
    print(f"temperature {temperature}, {tokens} tokens\n")

    while True:
        try:
            line = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line == "/analyze":
            show_analysis = not show_analysis
            print(f"      analysis {'on' if show_analysis else 'off'}\n")
            continue
        if line == "/translate":
            show_translation = not show_translation
            print(f"      gloss {'on' if show_translation else 'off'}\n")
            continue
        if line.startswith("/temp"):
            try:
                temperature = float(line.split()[1])
                print(f"      temperature = {temperature}\n")
            except (IndexError, ValueError):
                print("      usage: /temp 0.8\n")
            continue
        if line.startswith("/tokens"):
            try:
                tokens = int(line.split()[1])
                print(f"      tokens = {tokens}\n")
            except (IndexError, ValueError):
                print("      usage: /tokens 60\n")
            continue

        continuation = respond(line)
        print(f"gpt ▸ {line}\033[1m{continuation}\033[0m\n")
        if show_analysis:
            print(format_report(analyze_output(continuation)))
            print()
        if show_translation:
            gloss(line + continuation)
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
