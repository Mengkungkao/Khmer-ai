"""Instruction tuning (README sections 15 and 19, Project 8).

This is the stage that turns a text predictor into something that
answers. A pretrained model has learned only P(next token | previous
tokens) over Wikipedia, so given a Khmer question it continues the text
the way Wikipedia continues text - typically with more question-like
prose. It is not refusing to answer; answering is simply not what it was
trained to do.

Instruction tuning fixes that by training on formatted pairs:

    ### សំណួរ:
    តើរាជធានីរបស់ប្រទេសកម្ពុជាគឺជាអ្វី?

    ### ចម្លើយ:
    ភ្នំពេញ

**The essential detail is loss masking.** The model is shown the whole
sequence, but loss is computed only over the response tokens. Train on
everything and the model learns to generate questions as readily as
answers, and will happily reply to a question with another question.
Masking teaches it one thing: given this prompt shape, produce the part
after the answer marker.

Two honest limitations. Instruction tuning teaches the *form* of
answering, not knowledge - a model that never learned Cambodia's capital
during pretraining will now confidently produce a fluent, wrong answer,
which is arguably worse than rambling. And it needs on the order of
thousands of examples to generalize; a handful teaches it to reproduce
those specific answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models.from_scratch.gpt import KhmerGPT
from ..models.from_scratch.layers import cross_entropy_loss
from ..models.from_scratch.optimizer import Adam, clip_grad_norm, cosine_lr
from ..tokenizer.base import EOS, BaseTokenizer

# Khmer markers, so the model is not asked to learn English scaffolding
# in the middle of Khmer text. សំណួរ = question, ចម្លើយ = answer.
PROMPT_MARKER = "### សំណួរ:"
RESPONSE_MARKER = "### ចម្លើយ:"

# Recorded in the checkpoint by `save_checkpoint(prompt_format=...)` so a
# loader can tell that this model must be prompted through `answer()`
# rather than fed raw text.
PROMPT_FORMAT = "instruction"


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    output: str
    input: str = ""

    def format_prompt(self) -> str:
        """Everything the model is given, up to where its answer begins."""
        body = self.instruction if not self.input else f"{self.instruction}\n{self.input}"
        return f"{PROMPT_MARKER}\n{body}\n\n{RESPONSE_MARKER}\n"

    def format_full(self) -> str:
        return self.format_prompt() + self.output


class InstructionDatasetError(ValueError):
    """Raised when an instruction file is malformed."""


def load_instructions(path: str | Path) -> list[InstructionExample]:
    """Load instruction/input/output records from JSONL.

    Validates rather than trusts: an example with an empty output teaches
    the model to answer with nothing, which is worse than not training on
    it at all.
    """
    path = Path(path)
    if not path.exists():
        return []

    examples: list[InstructionExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InstructionDatasetError(f"line {line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(payload, dict):
                raise InstructionDatasetError(f"line {line_number}: expected a JSON object")

            for field in ("instruction", "output"):
                if field not in payload:
                    raise InstructionDatasetError(f"line {line_number}: missing '{field}'")
                if not isinstance(payload[field], str) or not payload[field].strip():
                    raise InstructionDatasetError(
                        f"line {line_number}: '{field}' must be a non-empty string"
                    )

            examples.append(
                InstructionExample(
                    instruction=payload["instruction"],
                    output=payload["output"],
                    input=payload.get("input", ""),
                )
            )
    return examples


def encode_example(
    example: InstructionExample, tokenizer: BaseTokenizer, max_len: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Encode one example into (ids, loss_mask), or None if it cannot fit.

    The mask marks only response positions. Because next-token prediction
    shifts by one, position t predicts token t+1 - so the mask is set on
    the positions whose *target* falls in the response.
    """
    prompt_ids = tokenizer.encode(example.format_prompt())
    # Append EOS so the model is explicitly taught where an answer ENDS.
    # Without it there is no way to signal completion, and a correct
    # answer gets padded with repeated fragments until the token budget
    # runs out ("ភ្នំពេញពេញពេញ" rather than "ភ្នំពេញ").
    full_ids = tokenizer.encode(example.format_full()) + [tokenizer.vocab.encode_token(EOS)]

    if len(full_ids) < 2 or len(prompt_ids) >= len(full_ids):
        return None
    if len(full_ids) > max_len:
        return None

    ids = np.array(full_ids, dtype=np.int64)
    mask = np.zeros(len(ids) - 1, dtype=np.int64)
    # Target at index i is ids[i+1]; count it when that lands in the response.
    mask[max(prompt_ids and len(prompt_ids) - 1, 0) :] = 1
    return ids, mask


def build_batches(
    examples: list[InstructionExample],
    tokenizer: BaseTokenizer,
    max_len: int,
    pad_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode all examples into padded (inputs, targets, mask) arrays.

    Padding is masked out, so shorter examples contribute nothing beyond
    their real tokens.
    """
    encoded = [e for e in (encode_example(x, tokenizer, max_len) for x in examples) if e]
    if not encoded:
        raise InstructionDatasetError(
            "no examples could be encoded - they may all exceed max_len, "
            "or the tokenizer may not cover the text"
        )

    width = max(len(ids) for ids, _ in encoded) - 1
    inputs = np.full((len(encoded), width), pad_id, dtype=np.int64)
    targets = np.full((len(encoded), width), pad_id, dtype=np.int64)
    masks = np.zeros((len(encoded), width), dtype=np.int64)

    for row, (ids, mask) in enumerate(encoded):
        length = len(ids) - 1
        inputs[row, :length] = ids[:-1]
        targets[row, :length] = ids[1:]
        masks[row, :length] = mask

    return inputs, targets, masks


def finetune(
    model: KhmerGPT,
    examples: list[InstructionExample],
    tokenizer: BaseTokenizer,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    log_every: int | None = None,
) -> list[float]:
    """Fine-tune a pretrained model on instruction pairs.

    The learning rate defaults far below pretraining's. Fine-tuning
    starts from a model that already knows the language, so large steps
    overwrite that knowledge faster than they teach the new format - the
    usual cause of a fine-tuned model that answers in the right shape but
    has forgotten how to write.
    """
    max_len = model.config.max_seq_len
    inputs, targets, masks = build_batches(examples, tokenizer, max_len + 1)

    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), lr=lr)
    warmup = max(1, int(steps * 0.1))
    losses: list[float] = []

    for step in range(steps):
        optimizer.lr = cosine_lr(step, steps, lr, warmup=warmup)
        rows = rng.integers(0, len(inputs), size=min(batch_size, len(inputs)))

        optimizer.zero_grad()
        logits = model.forward(inputs[rows])
        loss, dlogits = cross_entropy_loss(logits, targets[rows], mask=masks[rows])
        model.backward(dlogits)
        clip_grad_norm(model.parameters(), max_grad_norm)
        optimizer.step()

        losses.append(loss)
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:4d}/{steps}  loss {loss:.4f}", flush=True)

    return losses


def answer(
    model: KhmerGPT,
    tokenizer: BaseTokenizer,
    question: str,
    input: str = "",
    max_new_tokens: int = 40,
    temperature: float = 0.6,
    top_k: int = 20,
    rng: np.random.Generator | None = None,
) -> str:
    """Ask an instruction-tuned model a question and return its answer.

    `input` supplies the extra context some tasks need - the text to
    translate or summarize. It must be passed whenever the training
    example had one, because the prompt is reconstructed here and has to
    match the training format exactly. Omitting it asks the model to
    translate without showing it what to translate, and the model does
    not fail cleanly: it produces a fluent answer to the question it
    imagines was asked.
    """
    example = InstructionExample(instruction=question, output="", input=input)
    prompt_ids = tokenizer.encode(example.format_prompt())[-model.config.max_seq_len :]
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.vocab.encode_token(EOS),
        rng=rng,
    )
    # decode() already drops special tokens such as EOS.
    text = tokenizer.decode(generated[len(prompt_ids) :])
    # Stop at the next prompt marker if the model starts a new turn.
    return text.split(PROMPT_MARKER)[0].strip()
