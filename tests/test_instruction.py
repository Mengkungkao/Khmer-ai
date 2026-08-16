import numpy as np
import pytest

from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT
from khmer_language.tokenizer import GraphemeTokenizer
from khmer_language.tokenizer.base import EOS
from khmer_language.training.instruction import (
    PROMPT_MARKER,
    RESPONSE_MARKER,
    InstructionDatasetError,
    InstructionExample,
    answer,
    build_batches,
    encode_example,
    finetune,
    load_instructions,
)

EXAMPLES = [
    InstructionExample("តើរាជធានីរបស់ប្រទេសកម្ពុជាគឺជាអ្វី?", "ភ្នំពេញ"),
    InstructionExample("តើប្រាសាទអង្គរវត្តស្ថិតនៅខេត្តណា?", "សៀមរាប"),
    InstructionExample("តើរូបិយប័ណ្ណរបស់កម្ពុជាមានឈ្មោះថាអ្វី?", "រៀល"),
]


def _setup():
    tokenizer = GraphemeTokenizer()
    tokenizer.train([e.format_full() for e in EXAMPLES])
    model = KhmerGPT(
        GPTConfig(vocab_size=len(tokenizer.vocab), dim=64, num_layers=2, num_heads=2,
                  max_seq_len=96),
        seed=0,
    )
    return model, tokenizer


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def test_prompt_contains_both_markers():
    prompt = EXAMPLES[0].format_prompt()
    assert PROMPT_MARKER in prompt
    assert RESPONSE_MARKER in prompt


def test_prompt_stops_before_the_answer():
    assert EXAMPLES[0].output not in EXAMPLES[0].format_prompt()
    assert EXAMPLES[0].output in EXAMPLES[0].format_full()


def test_optional_input_is_included():
    example = InstructionExample("បកប្រែ", "Cambodia", input="កម្ពុជា")
    assert "កម្ពុជា" in example.format_prompt()


# --------------------------------------------------------------------------
# Loss masking - the essential mechanism
# --------------------------------------------------------------------------
def test_loss_mask_covers_only_the_response():
    """Training on the prompt too would teach the model to generate
    questions, and it would answer a question with another question."""
    _, tokenizer = _setup()
    example = EXAMPLES[0]
    ids, mask = encode_example(example, tokenizer, 200)

    trained_on = "".join(
        tokenizer.vocab.decode_id(int(ids[i + 1])) for i in range(len(mask)) if mask[i]
    )
    assert example.output in trained_on
    assert example.instruction not in trained_on


def test_mask_excludes_every_prompt_position():
    _, tokenizer = _setup()
    example = EXAMPLES[0]
    prompt_len = len(tokenizer.encode(example.format_prompt()))
    _, mask = encode_example(example, tokenizer, 200)
    assert mask[: prompt_len - 1].sum() == 0


def test_encoding_appends_an_end_of_sequence_token():
    """Without EOS the model cannot signal that an answer is finished,
    and pads correct answers with repeated fragments."""
    _, tokenizer = _setup()
    ids, _ = encode_example(EXAMPLES[0], tokenizer, 200)
    assert int(ids[-1]) == tokenizer.vocab.encode_token(EOS)


def test_eos_is_included_in_the_trained_positions():
    _, tokenizer = _setup()
    ids, mask = encode_example(EXAMPLES[0], tokenizer, 200)
    assert mask[-1] == 1  # the position whose target is EOS


def test_overlong_example_is_skipped():
    _, tokenizer = _setup()
    assert encode_example(EXAMPLES[0], tokenizer, max_len=5) is None


def test_batches_are_padded_and_padding_is_masked_out():
    _, tokenizer = _setup()
    inputs, targets, masks = build_batches(EXAMPLES, tokenizer, 200)
    assert inputs.shape == targets.shape == masks.shape
    assert len(inputs) == len(EXAMPLES)
    # every row must have at least one counted position, and fewer than all
    for row in range(len(EXAMPLES)):
        assert 0 < masks[row].sum() < masks.shape[1]


def test_build_batches_rejects_a_corpus_it_cannot_encode():
    _, tokenizer = _setup()
    with pytest.raises(InstructionDatasetError, match="no examples"):
        build_batches(EXAMPLES, tokenizer, max_len=3)


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def test_load_instructions(tmp_path):
    path = tmp_path / "i.jsonl"
    path.write_text(
        '# comment\n{"instruction": "សួស្តី", "output": "ជំរាបសួរ"}\n', encoding="utf-8"
    )
    loaded = load_instructions(path)
    assert len(loaded) == 1
    assert loaded[0].output == "ជំរាបសួរ"


def test_missing_file_returns_empty(tmp_path):
    assert load_instructions(tmp_path / "absent.jsonl") == []


def test_empty_output_is_rejected(tmp_path):
    """An example with no answer teaches the model to answer with nothing."""
    path = tmp_path / "i.jsonl"
    path.write_text('{"instruction": "x", "output": "  "}\n', encoding="utf-8")
    with pytest.raises(InstructionDatasetError, match="non-empty"):
        load_instructions(path)


def test_invalid_json_reports_the_line(tmp_path):
    path = tmp_path / "i.jsonl"
    path.write_text('{"instruction": "x", "output": "y"}\n{bad\n', encoding="utf-8")
    with pytest.raises(InstructionDatasetError, match="line 2"):
        load_instructions(path)


# --------------------------------------------------------------------------
# Fine-tuning end to end
# --------------------------------------------------------------------------
def test_finetuning_teaches_the_model_to_answer():
    """The whole point of Project 8: a model that continues text learns to
    respond to a question instead."""
    model, tokenizer = _setup()
    losses = finetune(model, EXAMPLES, tokenizer, steps=250, batch_size=3, lr=3e-3, seed=0)
    assert losses[-1] < losses[0]

    for example in EXAMPLES:
        got = answer(model, tokenizer, example.instruction, max_new_tokens=20, temperature=0.0)
        assert got == example.output


def test_answers_terminate_instead_of_rambling():
    """Regression: before EOS was trained, a correct answer came back as
    'ភ្នំពេញពេញពេញពេញ' because nothing told the model to stop."""
    model, tokenizer = _setup()
    finetune(model, EXAMPLES, tokenizer, steps=250, batch_size=3, lr=3e-3, seed=0)
    got = answer(model, tokenizer, EXAMPLES[0].instruction, max_new_tokens=40, temperature=0.0)
    assert len(got) <= len(EXAMPLES[0].output) + 2


def test_generation_stops_at_eos():
    model, tokenizer = _setup()
    eos = tokenizer.vocab.encode_token(EOS)
    out = model.generate([1, 2], max_new_tokens=50, temperature=0.0, eos_id=eos)
    if eos in out:
        assert out.index(eos) == len(out) - 1  # nothing generated after EOS
