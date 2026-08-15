"""Decompose a Khmer grapheme cluster into its structural parts.

Where `grapheme.py` answers "where do the cluster boundaries fall", this
module answers "what is this cluster made of": a base letter, zero or
more stacked subscript consonants (from COENG sequences, e.g. the two
subscripts ្ត and ្រ in ស្ត្រី), an optional register shifter, an
optional dependent vowel, and any trailing diacritic signs.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import codepoints as cp_db
from .character_types import CharacterType
from .grapheme import Grapheme


@dataclass(frozen=True)
class ConsonantCluster:
    text: str
    base: str
    base_type: CharacterType
    subscripts: tuple[str, ...]
    register_shifter: str | None
    vowel: str | None
    diacritics: tuple[str, ...]

    @property
    def base_series(self) -> str | None:
        consonant = cp_db.CONSONANTS_BY_CHAR.get(self.base)
        if consonant is not None:
            return consonant.series
        if self.base in cp_db.INDEPENDENT_VOWELS_BY_CHAR:
            return "a"
        return None


def analyze_cluster(grapheme: Grapheme) -> ConsonantCluster:
    """Break a single grapheme cluster into base/subscripts/vowel/signs."""
    char, base_type = grapheme.char_types[0]
    subscripts: list[str] = []
    register_shifter: str | None = None
    vowel: str | None = None
    diacritics: list[str] = []

    for ch, ctype in grapheme.char_types[1:]:
        if ctype is CharacterType.SUBSCRIPT_CONSONANT:
            subscripts.append(ch)
        elif ctype is CharacterType.COENG:
            continue  # structural marker, not a unit on its own
        elif ctype is CharacterType.REGISTER_SHIFTER:
            register_shifter = ch
        elif ctype in (CharacterType.DEPENDENT_VOWEL, CharacterType.INHERENT_VOWEL):
            vowel = ch
        elif ctype is CharacterType.DIACRITIC:
            diacritics.append(ch)

    return ConsonantCluster(
        text=grapheme.text,
        base=char,
        base_type=base_type,
        subscripts=tuple(subscripts),
        register_shifter=register_shifter,
        vowel=vowel,
        diacritics=tuple(diacritics),
    )
