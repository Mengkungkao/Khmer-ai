import numpy as np
import pytest

from khmer_language.models.from_scratch.checkpoint import (
    CheckpointError,
    load_checkpoint,
    read_metadata,
    save_checkpoint,
)
from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT
from khmer_language.tokenizer import (
    BPETokenizer,
    CharacterTokenizer,
    GraphemeTokenizer,
    SAMPLE_CORPUS,
    UnigramTokenizer,
)

CORPUS = list(SAMPLE_CORPUS)


def _model(vocab_size):
    return KhmerGPT(
        GPTConfig(vocab_size=vocab_size, dim=16, num_layers=1, num_heads=2, max_seq_len=32), seed=0
    )


def _grapheme_setup():
    tokenizer = GraphemeTokenizer()
    tokenizer.train(CORPUS)
    return _model(len(tokenizer.vocab)), tokenizer


def test_round_trip_preserves_weights(tmp_path):
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)

    loaded, _ = load_checkpoint(path)
    for original, restored in zip(model.parameters(), loaded.parameters()):
        assert np.array_equal(original.value, restored.value)


def test_round_trip_preserves_architecture(tmp_path):
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)
    loaded, _ = load_checkpoint(path)
    assert loaded.config == model.config
    assert loaded.num_parameters() == model.num_parameters()


def test_round_trip_preserves_the_tokenizer(tmp_path):
    """Without the tokenizer a checkpoint loads but cannot be used: token
    ids have no meaning without the vocabulary that produced them."""
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)
    _, restored = load_checkpoint(path)

    text = CORPUS[0]
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_loaded_model_produces_identical_output(tmp_path):
    """The real test of a checkpoint: same input, same logits."""
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)
    loaded, _ = load_checkpoint(path)

    ids = np.array([tokenizer.encode(CORPUS[0])[:8]])
    assert np.allclose(model.forward(ids), loaded.forward(ids))


def test_loaded_model_generates_identically_for_a_fixed_seed(tmp_path):
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)
    loaded, restored = load_checkpoint(path)

    prompt = tokenizer.encode(CORPUS[0])[:4]
    a = model.generate(prompt, max_new_tokens=10, rng=np.random.default_rng(3))
    b = loaded.generate(prompt, max_new_tokens=10, rng=np.random.default_rng(3))
    assert a == b


def test_bpe_merges_survive_the_round_trip(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=120)
    model = _model(len(tokenizer.vocab))

    path = save_checkpoint(tmp_path / "bpe.npz", model, tokenizer)
    _, restored = load_checkpoint(path)

    assert restored.merges == tokenizer.merges
    assert restored.tokenize(CORPUS[0]) == tokenizer.tokenize(CORPUS[0])


def test_unigram_probabilities_survive_the_round_trip(tmp_path):
    tokenizer = UnigramTokenizer()
    tokenizer.train(CORPUS, vocab_size=90)
    model = _model(len(tokenizer.vocab))

    path = save_checkpoint(tmp_path / "uni.npz", model, tokenizer)
    _, restored = load_checkpoint(path)

    assert restored.tokenize(CORPUS[0]) == tokenizer.tokenize(CORPUS[0])


def test_character_tokenizer_round_trips(tmp_path):
    tokenizer = CharacterTokenizer()
    tokenizer.train(CORPUS)
    model = _model(len(tokenizer.vocab))
    path = save_checkpoint(tmp_path / "char.npz", model, tokenizer)
    _, restored = load_checkpoint(path)
    assert restored.encode(CORPUS[0]) == tokenizer.encode(CORPUS[0])


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load_checkpoint(tmp_path / "absent.npz")


def test_weights_only_file_is_rejected_with_an_explanation(tmp_path):
    """A file with no metadata cannot be used, and must say why rather
    than failing obscurely later."""
    path = tmp_path / "weights_only.npz"
    np.savez(path, p0=np.zeros((2, 2)))
    with pytest.raises(CheckpointError, match="no metadata"):
        load_checkpoint(path)


def test_legacy_checkpoint_without_tie_flag_infers_the_architecture(tmp_path):
    """Regression: `tie_embeddings` was added later with a default of
    True, which changes the PARAMETER LIST (a tied model has no separate
    head matrix). Older files then built the wrong architecture, loaded
    weights into mismatched slots, raised nothing, and generated garbage.
    """
    import json

    tokenizer = GraphemeTokenizer()
    tokenizer.train(CORPUS)
    untied = KhmerGPT(
        GPTConfig(
            vocab_size=len(tokenizer.vocab), dim=16, num_layers=1, num_heads=2,
            max_seq_len=32, tie_embeddings=False,
        ),
        seed=0,
    )
    path = save_checkpoint(tmp_path / "legacy.npz", untied, tokenizer)

    # Strip the flag, as a checkpoint written before it existed would be.
    with np.load(path, allow_pickle=True) as data:
        contents = {k: data[k] for k in data.files}
    metadata = json.loads(str(contents["metadata"]))
    del metadata["config"]["tie_embeddings"]
    contents["metadata"] = np.array(json.dumps(metadata), dtype=object)
    np.savez(path, **contents)

    loaded, _ = load_checkpoint(path)
    assert loaded.config.tie_embeddings is False
    ids = np.array([[1, 2, 3]])
    assert np.allclose(untied.forward(ids), loaded.forward(ids))


def test_array_count_mismatch_is_rejected(tmp_path):
    """A file whose weight count disagrees with the architecture cannot be
    loaded correctly, so it must fail loudly rather than partially."""
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)

    with np.load(path, allow_pickle=True) as data:
        contents = {k: data[k] for k in data.files}
    contents["p999"] = np.zeros(3)  # an extra array the model does not expect
    np.savez(path, **contents)

    with pytest.raises(CheckpointError, match="weight arrays"):
        load_checkpoint(path)


def test_tied_and_untied_checkpoints_do_not_cross_load(tmp_path):
    tokenizer = GraphemeTokenizer()
    tokenizer.train(CORPUS)
    base = dict(vocab_size=len(tokenizer.vocab), dim=16, num_layers=1, num_heads=2, max_seq_len=32)

    tied = KhmerGPT(GPTConfig(**base, tie_embeddings=True), seed=0)
    untied = KhmerGPT(GPTConfig(**base, tie_embeddings=False), seed=0)
    assert len(tied.parameters()) != len(untied.parameters())

    for model, name in ((tied, "tied.npz"), (untied, "untied.npz")):
        path = save_checkpoint(tmp_path / name, model, tokenizer)
        loaded, _ = load_checkpoint(path)
        assert loaded.config.tie_embeddings == model.config.tie_embeddings
        ids = np.array([[1, 2, 3]])
        assert np.allclose(model.forward(ids), loaded.forward(ids))


def test_shape_mismatch_is_detected(tmp_path):
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)

    with np.load(path, allow_pickle=True) as data:
        contents = {k: data[k] for k in data.files}
    contents["p0"] = np.zeros((3, 3))
    np.savez(path, **contents)

    with pytest.raises(CheckpointError, match="shape"):
        load_checkpoint(path)


def test_prompt_format_round_trips(tmp_path):
    """An instruction-tuned model has to be prompted through the format it
    was tuned on, and nothing in the weights says so - the checkpoint has
    to carry it."""
    from khmer_language.training.instruction import PROMPT_FORMAT

    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer, prompt_format=PROMPT_FORMAT)

    assert read_metadata(path)["prompt_format"] == PROMPT_FORMAT


def test_plain_checkpoint_has_no_prompt_format(tmp_path):
    """Absent, not "": a caller checking the field must not read a
    pretrained completion model as instruction-tuned."""
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)

    assert "prompt_format" not in read_metadata(path)


def test_read_metadata_does_not_need_the_weights(tmp_path):
    model, tokenizer = _grapheme_setup()
    path = save_checkpoint(tmp_path / "m.npz", model, tokenizer)

    metadata = read_metadata(path)
    assert metadata["tokenizer"] == "GraphemeTokenizer"
    assert metadata["vocab"] == tokenizer.vocab.id_to_token


def test_read_metadata_rejects_a_missing_file(tmp_path):
    with pytest.raises(CheckpointError, match="no checkpoint"):
        read_metadata(tmp_path / "nope.npz")
