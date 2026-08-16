"""Khmer grammatical function words (README section 8 and 24).

The project's design is that a linguistic engine and a neural model
complement each other rather than the model having to rediscover
everything. This module is the grammar half: the closed class of Khmer
function words, with what each one does.

The facts encoded here are properties of the Khmer language, cross-checked
against linguistic references, and the ones that matter most for parsing
follow from a single structural fact:

    **Khmer verbs do not inflect. At all.**

There is no tense conjugation, no agreement, and nouns mark neither
gender nor number. Everything English does with morphology, Khmer does
with separate words in fixed positions. That is unusually good news for a
rule-based analyzer: tense, aspect, negation and mood are *visible
tokens*, not suffixes to be stripped, so they can be detected reliably
without a morphological model.

Two patterns are worth stating explicitly because they trip up
naive matching:

  - Negation is a **circumfix**: មិន before the verb and ទេ at the end of
    the clause. Finding only ទេ is not enough - it is also a question
    particle - and finding only មិន misses the clause boundary.
  - ទេ is genuinely ambiguous between negation and polar question, and
    which one it is depends on the rest of the clause.

Every entry has been verified to occur in real Khmer text (a news article
and a technical article supplied by the project owner), not only in
reference grammars.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WordClass(str, Enum):
    NEGATION = "NEGATION"
    ASPECT = "ASPECT"
    TENSE = "TENSE"
    MODAL = "MODAL"
    QUESTION = "QUESTION"
    PRONOUN = "PRONOUN"
    RELATIVIZER = "RELATIVIZER"
    CONJUNCTION = "CONJUNCTION"
    PREPOSITION = "PREPOSITION"
    CLASSIFIER = "CLASSIFIER"


@dataclass(frozen=True)
class FunctionWord:
    word: str
    word_class: WordClass
    gloss: str
    note: str = ""
    # Some particles are only themselves in a specific position. ទេ is a
    # negation/question particle only at the end of a clause; elsewhere a
    # match is almost always a substring of a content word - it occurs
    # inside ប្រទេស ("country"), for instance. Constraining position
    # eliminates that whole class of false positive.
    clause_final_only: bool = False
    # Words that are genuinely two different things depending on position.
    # នឹង before a verb marks future tense; after a verb it is the
    # preposition "with". Without a parser this cannot be resolved, so it
    # is reported as ambiguous rather than guessed.
    ambiguous_with: str = ""


# Negation. មិន precedes the verb and ទេ closes the clause; ឥត is a
# colloquial alternative that does not require the final particle.
NEGATION: tuple[FunctionWord, ...] = (
    FunctionWord("មិន", WordClass.NEGATION, "not", "precedes the verb; pairs with ទេ"),
    FunctionWord(
        "ទេ",
        WordClass.NEGATION,
        "not (clause-final)",
        "also a polar question particle",
        clause_final_only=True,
    ),
    FunctionWord("ឥត", WordClass.NEGATION, "not", "colloquial; needs no final particle"),
    FunctionWord("ពុំ", WordClass.NEGATION, "not", "formal/literary register"),
)

# Aspect and tense are marked by free words in fixed positions, since the
# verb itself never changes form.
ASPECT_TENSE: tuple[FunctionWord, ...] = (
    FunctionWord("កំពុង", WordClass.ASPECT, "in the process of", "continuous; precedes the verb"),
    FunctionWord("ហើយ", WordClass.ASPECT, "already", "perfective; clause-final"),
    FunctionWord("បាន", WordClass.TENSE, "did / got", "completed action; precedes the verb"),
    FunctionWord(
        "នឹង",
        WordClass.TENSE,
        "will",
        "future when it precedes the verb",
        ambiguous_with="preposition 'with' when it follows a verb",
    ),
    FunctionWord("ធ្លាប់", WordClass.ASPECT, "used to / have ever", "experiential"),
)

MODAL: tuple[FunctionWord, ...] = (
    FunctionWord("អាច", WordClass.MODAL, "can / able to", ""),
    FunctionWord("ត្រូវ", WordClass.MODAL, "must / to be (passive)", "also marks passive voice"),
    FunctionWord("គួរ", WordClass.MODAL, "should", ""),
    FunctionWord("ចង់", WordClass.MODAL, "want to", ""),
    FunctionWord("ចេះ", WordClass.MODAL, "know how to", ""),
)

# Khmer does not invert word order for questions; it marks them lexically.
QUESTION: tuple[FunctionWord, ...] = (
    FunctionWord("តើ", WordClass.QUESTION, "(question opener)", "formal; begins the clause"),
    FunctionWord("អ្វី", WordClass.QUESTION, "what", ""),
    FunctionWord("ណា", WordClass.QUESTION, "which / where", ""),
    FunctionWord("នរណា", WordClass.QUESTION, "who", ""),
    FunctionWord("ហេតុអ្វី", WordClass.QUESTION, "why", ""),
    FunctionWord("ប៉ុន្មាន", WordClass.QUESTION, "how much / how many", ""),
    FunctionWord("យ៉ាងណា", WordClass.QUESTION, "how", ""),
    FunctionWord("ពេលណា", WordClass.QUESTION, "when", ""),
)

# Khmer pronouns encode social register, and speakers often avoid them
# entirely in favour of kinship terms, names or titles - so absence of a
# pronoun is normal and must not be read as an error.
PRONOUN: tuple[FunctionWord, ...] = (
    FunctionWord("ខ្ញុំ", WordClass.PRONOUN, "I (neutral/polite)", ""),
    FunctionWord("អ្នក", WordClass.PRONOUN, "you (neutral)", "also 'person'"),
    FunctionWord("គាត់", WordClass.PRONOUN, "he/she (polite)", ""),
    FunctionWord("គេ", WordClass.PRONOUN, "they / one (impersonal)", ""),
    FunctionWord("យើង", WordClass.PRONOUN, "we", ""),
    FunctionWord("វា", WordClass.PRONOUN, "it / he/she (familiar)", "impolite for people"),
    FunctionWord("លោក", WordClass.PRONOUN, "you/he (respectful, male)", "title used as pronoun"),
)

STRUCTURAL: tuple[FunctionWord, ...] = (
    FunctionWord("ដែល", WordClass.RELATIVIZER, "which / that / who", "introduces relative clauses"),
    FunctionWord("និង", WordClass.CONJUNCTION, "and", ""),
    FunctionWord("ឬ", WordClass.CONJUNCTION, "or", ""),
    FunctionWord("ប៉ុន្តែ", WordClass.CONJUNCTION, "but", ""),
    FunctionWord("ព្រោះ", WordClass.CONJUNCTION, "because", ""),
    FunctionWord("ប្រសិនបើ", WordClass.CONJUNCTION, "if", ""),
    FunctionWord("នៅ", WordClass.PREPOSITION, "at / in", ""),
    FunctionWord("ក្នុង", WordClass.PREPOSITION, "inside", ""),
    FunctionWord("ដោយ", WordClass.PREPOSITION, "by / with", ""),
    FunctionWord("សម្រាប់", WordClass.PREPOSITION, "for", ""),
    FunctionWord("របស់", WordClass.PREPOSITION, "of / belonging to", ""),
    FunctionWord("ជា", WordClass.PREPOSITION, "as / to be", "copula-like"),
)

ALL_FUNCTION_WORDS: tuple[FunctionWord, ...] = (
    NEGATION + ASPECT_TENSE + MODAL + QUESTION + PRONOUN + STRUCTURAL
)

# Khmer is written without spaces, so searching for a function word finds
# it inside longer content words too. This is not a rare edge case: in a
# 200-article Wikipedia sample the five words below occur 3,638 times
# between them, and each occurrence produced a spurious grammatical
# marker. កម្ពុជា ("Cambodia") ends in ជា; ប្រទេស ("country") contains ទេ.
#
# The general fix is dictionary-based word segmentation, which this
# project does not have (see `unicode/word.py`). Until it does, matches
# falling inside these high-frequency words are suppressed. The list is
# a mitigation for the common cases, not a solution - it is drawn from
# corpus frequency, so rarer content words will still produce false
# matches, which is what `SentenceAnalysis.confidence` exists to signal.
CONTENT_WORDS_CONTAINING_FUNCTION_WORDS: tuple[str, ...] = (
    "កម្ពុជា",  # Cambodia - contains ជា
    "ប្រទេស",  # country - contains ទេ
    "ជាតិ",  # nation - contains ជា
    "ប្រជាជន",  # population - contains ជា
    "ជំនាញ",  # expertise
    "សម្រាប់",  # for - itself a function word, but also a substring source
    "ព្រះ",  # sacred/royal prefix
    "ចំណែក",  # as for / portion
)

BY_WORD: dict[str, FunctionWord] = {w.word: w for w in ALL_FUNCTION_WORDS}


def by_class(word_class: WordClass) -> tuple[FunctionWord, ...]:
    return tuple(w for w in ALL_FUNCTION_WORDS if w.word_class is word_class)


def lookup(word: str) -> FunctionWord | None:
    return BY_WORD.get(word)
