import numpy as np

from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT
from khmer_language.training import evaluate, train


def _model(vocab=5, seed=0):
    return KhmerGPT(
        GPTConfig(vocab_size=vocab, dim=16, num_layers=1, num_heads=2, max_seq_len=16), seed=seed
    )


def test_evaluate_is_deterministic():
    """A fixed eval seed means the validation curve moves only when the
    model changes, not because different windows were sampled."""
    model = _model()
    data = np.tile(np.arange(5), 40)
    assert evaluate(model, data, seq_len=8) == evaluate(model, data, seq_len=8)


def test_validation_losses_are_recorded_when_data_is_supplied():
    model = _model()
    data = np.tile(np.arange(5), 40)
    report = train(model, data, steps=20, batch_size=2, seq_len=8, validation_data=data, eval_every=5)
    assert len(report.validation_losses) >= 4
    steps = [s for s, _ in report.validation_losses]
    assert steps == sorted(steps)


def test_no_validation_losses_without_validation_data():
    model = _model()
    data = np.tile(np.arange(5), 40)
    report = train(model, data, steps=10, batch_size=2, seq_len=8)
    assert report.validation_losses == []
    assert np.isnan(report.final_validation_loss)


def test_validation_loss_falls_when_train_and_validation_share_a_pattern():
    """Same generating process, disjoint samples: a model that genuinely
    learns the pattern must improve on held-out data too."""
    model = _model()
    train_data = np.tile(np.arange(5), 60)
    validation_data = np.tile(np.arange(5), 30)

    report = train(
        model,
        train_data,
        steps=100,
        batch_size=4,
        seq_len=8,
        lr=5e-3,
        validation_data=validation_data,
        eval_every=20,
    )
    first = report.validation_losses[0][1]
    last = report.validation_losses[-1][1]
    assert last < first


def test_overfitting_flag_is_false_while_validation_improves():
    model = _model()
    data = np.tile(np.arange(5), 60)
    report = train(
        model, data, steps=60, batch_size=4, seq_len=8, lr=5e-3,
        validation_data=data, eval_every=10,
    )
    assert not report.overfitting


def test_overfitting_flag_detects_a_rising_validation_curve():
    from khmer_language.training.pretrain import TrainingReport

    report = TrainingReport()
    report.validation_losses = [(10, 1.0), (20, 0.5), (30, 0.9)]
    assert report.overfitting

    improving = TrainingReport()
    improving.validation_losses = [(10, 1.0), (20, 0.7), (30, 0.5)]
    assert not improving.overfitting
