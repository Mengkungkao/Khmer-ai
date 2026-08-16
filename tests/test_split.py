import pytest

from khmer_language.corpus import Document
from khmer_language.corpus.split import split_documents


def _docs(n):
    return [Document(f"d{i}", f"អត្ថបទ {i}", "test", "CC0") for i in range(n)]


def test_split_sizes_sum_to_the_input():
    split = split_documents(_docs(100), validation_fraction=0.1, test_fraction=0.1)
    assert sum(split.sizes.values()) == 100
    assert split.sizes == {"train": 80, "validation": 10, "test": 10}


def test_splits_are_disjoint():
    """Document-level disjointness is the whole point: any overlap lets
    the model see validation text during training."""
    split = split_documents(_docs(100), validation_fraction=0.1, test_fraction=0.1)
    train_ids = {d.id for d in split.train}
    validation_ids = {d.id for d in split.validation}
    test_ids = {d.id for d in split.test}

    assert not (train_ids & validation_ids)
    assert not (train_ids & test_ids)
    assert not (validation_ids & test_ids)
    assert len(train_ids | validation_ids | test_ids) == 100


def test_split_is_deterministic_for_a_given_seed():
    a = split_documents(_docs(50), seed=7)
    b = split_documents(_docs(50), seed=7)
    assert [d.id for d in a.validation] == [d.id for d in b.validation]


def test_different_seeds_give_different_partitions():
    a = split_documents(_docs(200), seed=1)
    b = split_documents(_docs(200), seed=2)
    assert [d.id for d in a.validation] != [d.id for d in b.validation]


def test_documents_are_shuffled_not_taken_in_order():
    """Taking the first N as validation would bias the split by whatever
    order the corpus happened to be built in."""
    split = split_documents(_docs(200), validation_fraction=0.1, test_fraction=0.0)
    ids = [int(d.id[1:]) for d in split.validation]
    assert ids != sorted(range(len(ids)))


def test_small_corpus_still_gets_a_nonempty_validation_set():
    split = split_documents(_docs(5), validation_fraction=0.05, test_fraction=0.05)
    assert len(split.validation) >= 1
    assert len(split.test) >= 1
    assert len(split.train) >= 1


def test_zero_fraction_gives_an_empty_split():
    split = split_documents(_docs(50), validation_fraction=0.0, test_fraction=0.2)
    assert split.validation == []
    assert len(split.test) == 10


def test_fractions_leaving_no_training_data_are_rejected():
    with pytest.raises(ValueError, match="room for training"):
        split_documents(_docs(50), validation_fraction=0.6, test_fraction=0.5)


def test_negative_fraction_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        split_documents(_docs(10), validation_fraction=-0.1)


def test_empty_corpus_does_not_crash():
    split = split_documents([])
    assert split.sizes == {"train": 0, "validation": 0, "test": 0}


def test_str_is_readable():
    assert "documents ->" in str(split_documents(_docs(10)))
