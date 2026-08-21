#!/usr/bin/env python3
"""Interactive session with a trained KhmerGPT (README Project 9).

    python3 scripts/chat.py --checkpoint data/checkpoints/khmergpt0-kmwiki.npz
    python3 scripts/chat.py --checkpoint data/checkpoints/khmergpt0-instruct.npz

The script runs in one of two modes, because there are two kinds of
checkpoint and feeding one to the other's code path fails silently.

**Completion mode** (a pretrained checkpoint). The model has learned
exactly one thing - P(next token | previous tokens) over Khmer
Wikipedia. Type a question and it will continue it the way Wikipedia
text tends to continue, which is usually *more question-like text*, not
an answer. Give it the start of a sentence instead and watch what it
knows about Khmer.

**Instruct mode** (a checkpoint from scripts/finetune_instruct.py). The
line you type is wrapped in the ### សំណួរ: / ### ចម្លើយ: format the
model was fine-tuned on, and generation stops at the end-of-answer
token. Skipping that wrapping is not a small loss of quality: the model
never saw bare text after tuning and responds to it with degenerate
output, so the mode has to be right.

The mode is read from the checkpoint (`prompt_format` in its metadata)
and can be forced with --instruct / --no-instruct, which is what older
instruct checkpoints - saved before that field existed - need.
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
    read_metadata,
)
from khmer_language.training.instruction import PROMPT_FORMAT, answer  # noqa: E402

COMMANDS = """\
Commands:  /temp <n>   sampling temperature (lower = safer, higher = wilder)
           /tokens <n> how many tokens to generate
           /analyze    toggle the structural quality report
           /translate  toggle a rough English gloss (see caveat below)
           /quit       exit"""

COMPLETION_BANNER = f"""\
KhmerGPT interactive session - completion mode
────────────────────────────────────────────────────────────────────
This checkpoint COMPLETES Khmer text. It does not answer questions: it
was pretrained on Khmer Wikipedia and nothing else, so a question comes
back continued rather than answered.

To ask questions, load an instruction-tuned checkpoint instead:
  python3 scripts/chat.py --checkpoint data/checkpoints/khmergpt0-instruct.npz

Try giving it the beginning of a sentence, for example:
  ប្រទេសកម្ពុជា
  ភ្នំពេញជា
  ប្រវត្តិសាស្ត្រ

{COMMANDS}
────────────────────────────────────────────────────────────────────"""

INSTRUCT_BANNER = f"""\
KhmerGPT interactive session - instruct mode
────────────────────────────────────────────────────────────────────
This checkpoint was instruction-tuned, so what you type is wrapped in
the question/answer format it was trained on and generation stops at
the end-of-answer token.

It was tuned on a handful of examples, which is enough to MEMORIZE
those answers and nowhere near enough to generalize. Expect the tuned
questions to come back right and most other input - English especially
- to return the nearest memorized answer or nonsense. That is the
dataset size talking, not a bug.

Try one of the questions it was tuned on:
  តើរាជធានីរបស់ប្រទេសកម្ពុជាគឺជាអ្វី?
  តើប្រាសាទអង្គរវត្តស្ថិតនៅខេត្តណា?
  តើរូបិយប័ណ្ណរបស់កម្ពុជាមានឈ្មោះថាអ្វី?

{COMMANDS}
           /input <t>  extra context for tasks that need it, e.g.
                       the word to translate; /input alone clears it
────────────────────────────────────────────────────────────────────"""

TRANSLATE_CAVEAT = """\
  note: machine translation always returns something plausible, so a
  readable gloss does NOT mean the Khmer was correct. Measured on this
  project, real Khmer, model output and pure nonsense all came back with
  the same 0.85 confidence score. Use it to get the gist, not as a grader."""

# Completion defaults chosen by measurement, not taste - see
# scripts/tune_sampling.py. Across nine candidates, t=0.6/k=20 landed
# closest to real Khmer on grounding and diversity together (93.0%/79.8%
# against a real-corpus 99.8%/82.2%). A repetition penalty measurably
# HURT here, dropping grounding to 75.4%, so it defaults to off.
COMPLETION_TEMPERATURE = 0.6

# Instruct defaults to greedy. Sampling is there to keep completions from
# going flat, but a tuned answer is a specific string the model already
# knows: temperature only gives it chances to wander off that string.
# This is what finetune_instruct.py and the benchmark both decode with.
INSTRUCT_TEMPERATURE = 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/checkpoints/khmergpt0-kmwiki.npz")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"default {COMPLETION_TEMPERATURE} completing, "
             f"{INSTRUCT_TEMPERATURE} (greedy) in instruct mode",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--analyze", action="store_true", help="show the quality report each turn")
    parser.add_argument("--translate", action="store_true", help="show a rough English gloss")
    parser.add_argument("--top-p", type=float, default=None, help="completion mode only")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="completion mode only")
    parser.add_argument("--prompt", default=None, help="run one prompt and exit (non-interactive)")
    parser.add_argument("--input", default="", help="instruct mode: extra context for the task")
    parser.add_argument(
        "--instruct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force instruction mode on/off instead of reading it from the checkpoint",
    )
    args = parser.parse_args()

    try:
        metadata = read_metadata(args.checkpoint)
        model, tokenizer = load_checkpoint(args.checkpoint)
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("\nTrain one first:", file=sys.stderr)
        print("  python3 scripts/train_khmergpt.py --corpus data/cleaned/kmwiki.jsonl "
              "--save data/checkpoints/khmergpt0-kmwiki.npz", file=sys.stderr)
        return 1

    instruct = (
        metadata.get("prompt_format") == PROMPT_FORMAT
        if args.instruct is None
        else args.instruct
    )

    rng = np.random.default_rng(args.seed)
    temperature = args.temperature
    if temperature is None:
        temperature = INSTRUCT_TEMPERATURE if instruct else COMPLETION_TEMPERATURE
    tokens = args.tokens
    show_analysis = args.analyze
    show_translation = args.translate
    task_input = args.input
    translator = ReferenceTranslator()

    if instruct and (args.top_p is not None or args.repetition_penalty != 1.0):
        print("note: --top-p and --repetition-penalty apply to completion mode only "
              "and are ignored here.\n", file=sys.stderr)

    def respond(prompt: str) -> str:
        """The model's reply: an answer in instruct mode, a continuation otherwise."""
        if instruct:
            return answer(
                model,
                tokenizer,
                prompt,
                input=task_input,
                max_new_tokens=tokens,
                temperature=temperature,
                top_k=args.top_k,
                rng=rng,
            )

        ids = tokenizer.encode(prompt)
        if not ids:
            return ""
        generated = model.generate(
            ids,
            max_new_tokens=tokens,
            temperature=temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            rng=rng,
        )
        # Return only the continuation, so it is obvious what the model added.
        return tokenizer.decode(generated[len(ids):])

    def full_text(prompt: str, reply: str) -> str:
        """What to read as the model's output.

        In completion mode the prompt is part of the sentence the model
        is building, so the two belong together. An answer is not a
        continuation of the question and stands on its own.
        """
        return reply if instruct else f"{prompt}{reply}"

    def gloss(text: str) -> None:
        result = translator.translate(text)
        if result is None:
            print("  [gloss unavailable - translation service unreachable]")
        else:
            print(f"  [gloss] {result.translated}")
        print(TRANSLATE_CAVEAT)

    if args.prompt is not None:
        reply = respond(args.prompt)
        print(full_text(args.prompt, reply))
        if show_analysis:
            print(format_report(analyze_output(reply)))
        if show_translation:
            gloss(full_text(args.prompt, reply))
        return 0

    print(INSTRUCT_BANNER if instruct else COMPLETION_BANNER)
    print(f"model: {model.num_parameters():,} parameters, vocab {len(tokenizer.vocab):,}")
    print(f"temperature {temperature}, {tokens} tokens")
    if instruct and task_input:
        print(f"input: {task_input}")
    print()

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
        if line.startswith("/input"):
            if not instruct:
                print("      /input applies to instruct mode only\n")
                continue
            task_input = line[len("/input"):].strip()
            print(f"      input = {task_input!r}\n" if task_input else "      input cleared\n")
            continue

        reply = respond(line)
        if instruct:
            print(f"gpt ▸ \033[1m{reply}\033[0m\n")
        else:
            print(f"gpt ▸ {line}\033[1m{reply}\033[0m\n")
        if show_analysis:
            print(format_report(analyze_output(reply)))
            print()
        if show_translation:
            gloss(full_text(line, reply))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
