"""Corpus document with provenance metadata (README section 7).

Every document carries where it came from and under what licence,
because section 6's pipeline puts a "Copyright / License Check" step
immediately after collection, and section 16 requires source tracking for
knowledge documents. Metadata that is not recorded at ingestion time
cannot be reconstructed later - by the time a model has trained on a
document, "was this licensed for this use?" is unanswerable without it.

`license` and `source` are deliberately required rather than defaulted:
silently defaulting them to "unknown" makes it easy to build a corpus
nobody can vouch for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Document:
    id: str
    text: str
    source: str
    license: str
    domain: str = "general"
    language: str = "km"
    quality: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in data.items() if k not in known}
        base = {k: v for k, v in data.items() if k in known}
        doc = cls(**base)
        doc.metadata.update(extra)
        return doc


def write_jsonl(documents: list[Document], path: str | Path) -> None:
    """Write documents as JSON Lines - the format README section 26 lists
    first, and the one that streams without loading a whole corpus."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[Document]:
    """Stream documents from a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Document.from_dict(json.loads(line))
