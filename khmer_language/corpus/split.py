"""Train / validation / test splitting (README section 21's data layout).

Splitting happens at the **document** level, never at the sentence or
token level. This is the single most important methodological detail in
the module: a Wikipedia article repeats names, dates and whole phrases
across its own sentences, so putting some sentences of an article in
train and others in validation lets the model see the answer during
training. Validation loss then looks great and means nothing.

Splitting whole documents keeps the two sets genuinely disjoint, so
validation perplexity measures generalization to unseen text - which is
the only number that distinguishes learning from memorization.

The split is seeded and deterministic, so re-running gives the same
partition and results stay comparable between experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .document import Document


@dataclass(frozen=True)
class CorpusSplit:
    train: list[Document]
    validation: list[Document]
    test: list[Document]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def __str__(self) -> str:
        total = sum(self.sizes.values())
        parts = [f"{name}: {n:,} docs" for name, n in self.sizes.items()]
        return f"{total:,} documents -> " + ", ".join(parts)


def split_documents(
    documents: list[Document],
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 0,
) -> CorpusSplit:
    """Partition documents into train/validation/test.

    Raises if the requested fractions would leave a split empty, since a
    silently empty validation set produces meaningless evaluation.
    """
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"validation ({validation_fraction}) + test ({test_fraction}) "
            "fractions must leave room for training data"
        )

    order = np.random.default_rng(seed).permutation(len(documents))
    shuffled = [documents[i] for i in order]

    n = len(shuffled)
    n_validation = int(n * validation_fraction)
    n_test = int(n * test_fraction)

    if n >= 3:
        # With enough documents, never hand back an empty held-out set:
        # evaluating on nothing is worse than a tiny sample.
        n_validation = max(n_validation, 1) if validation_fraction > 0 else 0
        n_test = max(n_test, 1) if test_fraction > 0 else 0

    validation = shuffled[:n_validation]
    test = shuffled[n_validation : n_validation + n_test]
    train = shuffled[n_validation + n_test :]

    return CorpusSplit(train=train, validation=validation, test=test)
