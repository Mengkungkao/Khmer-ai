"""Saving and loading trained models.

A saved model is only useful if it can be reloaded and used, and that
needs three things together:

  1. the weights
  2. the architecture (`GPTConfig`) that gives those weights meaning
  3. the **tokenizer vocabulary**

The third is the one that is easy to forget and fatal to omit. Token id
417 means nothing on its own - it is only "ជា" because a particular
vocabulary said so. Saving weights alone produces a file that can be
loaded but not used, since there is no way to turn text into the ids the
model was trained on, or ids back into Khmer.

Everything is stored in a single `.npz`: weight arrays plus a JSON
metadata blob. Loading uses `allow_pickle=False`, so a checkpoint file
can never execute code on load.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ...tokenizer.base import BaseTokenizer, Vocabulary
from ...tokenizer.character import CharacterTokenizer
from ...tokenizer.grapheme import GraphemeTokenizer
from ...tokenizer.syllable import SyllableTokenizer
from .gpt import GPTConfig, KhmerGPT

CHECKPOINT_VERSION = 1

# Tokenizers whose behaviour is fully determined by their vocabulary, so
# restoring the vocabulary restores the tokenizer exactly. BPE and Unigram
# also carry learned merges/probabilities and are handled separately.
_VOCAB_ONLY_TOKENIZERS: dict[str, type[BaseTokenizer]] = {
    "CharacterTokenizer": CharacterTokenizer,
    "GraphemeTokenizer": GraphemeTokenizer,
    "SyllableTokenizer": SyllableTokenizer,
}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded or used as intended."""


def save_checkpoint(
    path: str | Path,
    model: KhmerGPT,
    tokenizer: BaseTokenizer,
    *,
    prompt_format: str | None = None,
) -> Path:
    """Write model weights, architecture and tokenizer to one .npz file.

    `prompt_format` records how the model expects to be *prompted* - for
    an instruction-tuned model, "instruction". This belongs with the
    weights for the same reason the vocabulary does: a model fine-tuned
    on the question/answer format produces degenerate text when fed raw
    prompts, and nothing about the weights reveals that. Omitted means a
    plain completion model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    name = type(tokenizer).__name__
    metadata = {
        "version": CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "tokenizer": name,
        "vocab": tokenizer.vocab.id_to_token,
    }
    if prompt_format is not None:
        metadata["prompt_format"] = prompt_format

    if name == "BPETokenizer":
        metadata["merges"] = [list(pair) for pair in tokenizer.merges]
    elif name == "UnigramTokenizer":
        metadata["piece_logprobs"] = tokenizer.piece_logprobs
        metadata["max_piece_graphemes"] = tokenizer.max_piece_graphemes
    elif name not in _VOCAB_ONLY_TOKENIZERS:
        raise CheckpointError(f"don't know how to serialize tokenizer {name!r}")

    arrays = {f"p{i}": p.value for i, p in enumerate(model.parameters())}
    np.savez(path, metadata=np.array(json.dumps(metadata), dtype=object), **arrays)
    return path


def _restore_tokenizer(metadata: dict) -> BaseTokenizer:
    name = metadata["tokenizer"]
    vocab = Vocabulary.from_dict({"id_to_token": metadata["vocab"]})

    if name in _VOCAB_ONLY_TOKENIZERS:
        tokenizer = _VOCAB_ONLY_TOKENIZERS[name]()
    elif name == "BPETokenizer":
        from ...tokenizer.bpe import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.merges = [tuple(pair) for pair in metadata["merges"]]
    elif name == "UnigramTokenizer":
        from ...tokenizer.unigram import UnigramTokenizer

        tokenizer = UnigramTokenizer(max_piece_graphemes=metadata["max_piece_graphemes"])
        tokenizer.piece_logprobs = dict(metadata["piece_logprobs"])
    else:
        raise CheckpointError(f"unknown tokenizer in checkpoint: {name!r}")

    tokenizer.vocab = vocab
    return tokenizer


def read_metadata(path: str | Path) -> dict:
    """Read a checkpoint's metadata without building the model.

    Callers that only need to know *how to use* a checkpoint - which
    prompt format it expects, what vocabulary it carries - should not
    have to allocate its weights to find out.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")

    with np.load(path, allow_pickle=True) as data:
        if "metadata" not in data:
            raise CheckpointError(
                f"{path} has no metadata - it was written by an older save that stored "
                "weights only, so the architecture and tokenizer it needs are unknown"
            )
        metadata = json.loads(str(data["metadata"]))

    if metadata.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"checkpoint version {metadata.get('version')} "
            f"!= supported {CHECKPOINT_VERSION}"
        )
    return metadata


def load_checkpoint(path: str | Path) -> tuple[KhmerGPT, BaseTokenizer]:
    """Load a model and its tokenizer back from a checkpoint."""
    path = Path(path)
    metadata = read_metadata(path)

    with np.load(path, allow_pickle=True) as data:
        config_fields = dict(metadata["config"])
        saved_arrays = sum(1 for name in data.files if name.startswith("p"))

        # Backward compatibility. `tie_embeddings` was added after some
        # checkpoints were written, and it changes the parameter LIST -
        # a tied model has no separate head matrix. Defaulting it for an
        # older file silently builds the wrong architecture and loads
        # weights into mismatched slots, which does not crash: it just
        # produces a model that generates garbage. Infer it from the
        # array count instead of assuming.
        if "tie_embeddings" not in config_fields:
            for tie in (False, True):
                candidate = GPTConfig(**config_fields, tie_embeddings=tie)
                if len(KhmerGPT(candidate).parameters()) == saved_arrays:
                    config_fields["tie_embeddings"] = tie
                    break

        model = KhmerGPT(GPTConfig(**config_fields))
        parameters = model.parameters()

        # Guard the general case: any mismatch between what the file holds
        # and what the architecture expects means the weights cannot be
        # placed correctly, so fail loudly rather than load them wrongly.
        if saved_arrays != len(parameters):
            raise CheckpointError(
                f"{path} holds {saved_arrays} weight arrays but this architecture "
                f"expects {len(parameters)}; the saved config does not describe these "
                "weights, so loading them would silently produce a broken model"
            )

        for i, parameter in enumerate(parameters):
            key = f"p{i}"
            if key not in data:
                raise CheckpointError(f"{path} is missing weight array {key}")
            saved = data[key]
            if saved.shape != parameter.value.shape:
                raise CheckpointError(
                    f"weight {key} has shape {saved.shape}, model expects "
                    f"{parameter.value.shape}"
                )
            parameter.value[...] = saved

    return model, _restore_tokenizer(metadata)
