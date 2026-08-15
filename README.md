# KhmerAI — Build a Khmer-Native Artificial Intelligence From First Principles

## Status

**Project 1 — Khmer Unicode & Grapheme Engine is implemented**, and
**Project 4 — Tokenizer Lab has a first version**, both in
[`khmer_language/`](khmer_language/). Everything else in this document is
the roadmap, not yet built.

What exists today, with no third-party dependencies (stdlib only):

- `khmer_language/unicode/codepoints.py` — a machine-readable database of
  the full Khmer Unicode block (U+1780–U+17FF): 35 consonants (with
  register/series, romanization, IPA), 17 independent vowels, 16 dependent
  vowels (with per-series pronunciation), signs, digits, punctuation,
  currency. Exportable as JSON via `python -m khmer_language --export-db`.
- `character_types.py` — context-aware classification of Khmer text into
  linguistic categories (consonant vs. subscript consonant, dependent
  vowel, register shifter, coeng, etc).
- `grapheme.py` — a from-scratch Khmer grapheme cluster (KCC) segmenter.
  Not built on `regex`'s `\X`/ICU, because default Unicode grapheme
  clustering does not correctly group Khmer COENG subscript sequences.
- `cluster.py` — decomposes a grapheme into base consonant / stacked
  subscripts / register shifter / vowel / diacritics.
- `syllable.py` — syllable segmentation (v1: approximated as one grapheme
  cluster per syllable; see the module docstring for the known limits).
- `sentence.py` — sentence segmentation on KHAN/BARIYOOSAN (។ ៕) and ASCII
  `.!?`.
- `word.py` — word segmentation, v1: splits on ZWSP/whitespace/punctuation
  boundaries only. A boundary-free run of Khmer is returned as one "word"
  — real segmentation needs a dictionary, which doesn't exist yet
  (Project 3/4). Documented limitation, not a bug.
- `normalizer.py` — NFC + whitespace normalization (conservative by
  design — see module docstring for what it deliberately does not do).
- `validator.py` — structural well-formedness checks (orphan combining
  marks, malformed COENG sequences, unassigned code points, legacy/obsolete
  characters).
- `transliterator.py` — best-effort Khmer→Latin transliteration.
- `analyzer.py` + `cli.py` — ties it together: `python -m khmer_language "កម្ពុជា"`.

`khmer_language/tokenizer/` — the Tokenizer Lab (Project 4 / README
section 10): `CharacterTokenizer`, `GraphemeTokenizer`, `SyllableTokenizer`,
and a grapheme-aware `BPETokenizer` (merges start from Khmer grapheme
clusters, not raw code points, so a learned token can never split a
COENG subscript from its base). All share a common vocab/encode/decode
interface (`tokenizer/base.py`). `tokenizer/compare.py` reports
vocab size / sequence length / compression ratio / unknown-token rate
side by side — run it with `python -m khmer_language --compare-tokenizers`.
There is no real Khmer corpus yet (Project 3, not started), so training
uses a small placeholder `SAMPLE_CORPUS`.

`khmer_language/embeddings/` — Word2Vec Skip-Gram with negative sampling
(Project 5 / README section 11), written from scratch with NumPy: two
weight matrices, the 3/4-power unigram noise distribution, manual
forward/backward pass. The tokenizer is pluggable, so it trains over
Khmer graphemes/syllables/BPE subwords rather than assuming
space-separated words. Verified to actually learn: on a corpus where ថៃ
(Thailand) and ឡាវ (Laos) appear in interchangeable contexts, ថៃ ends up
measurably closer to ឡាវ than to an unrelated token — across every seed
tested, exactly the effect section 11 predicts.

`khmer_language/models/from_scratch/` — neural network primitives with
hand-derived backward passes, NumPy only, no autograd (Project 6 /
README section 13): `Linear`, `LayerNorm`, `GELU`, `Embedding`, stable
`softmax`, and a fused `cross_entropy_loss`. Because a wrong gradient
doesn't crash — it just trains badly — every backward pass is checked
against a central finite-difference gradient (`gradcheck.py`). Those
checks are verified to be meaningful, not vacuous: a deliberately
sabotaged LayerNorm backward scores a relative error of 1.0 against the
1e-6 threshold that correct math passes at 2e-9.

`attention.py` adds multi-head **causal** self-attention and
`transformer.py` the pre-norm `TransformerBlock` (`x + attn(norm(x))`,
`x + ffn(norm(x))`), both with hand-derived backward passes. Causality is
tested as a property, not assumed: scrambling later positions must leave
earlier outputs bit-for-bit identical.

Gradient checking found a genuine mathematical fact worth knowing: the
attention **key bias gradient is exactly zero**. Adding `b_k` shifts every
score in a row by the same amount, and softmax is shift-invariant along
that axis, so `b_k` cannot affect the output at all (verified: scrambling
it changes outputs by 7e-16). That's why gradient checks need both a
relative *and* an absolute tolerance — the finite-difference estimate of a
truly-zero gradient is pure cancellation noise (~1e-11), which no relative
test can accept. The combined check still catches a 0.1% gradient error.

**KhmerGPT-0 exists and trains.** `models/from_scratch/gpt.py` assembles
the above into a decoder-only LM (token + learned positional embeddings →
N transformer blocks → final LayerNorm → linear head), with Adam,
gradient clipping and a next-token training loop in
`khmer_language/training/`. Run it:

```bash
python3 -m khmer_language --train-demo
```

A 112k-parameter KhmerGPT-0 on the placeholder corpus drops from loss
**4.96 → 0.067** in 150 steps (a random-init model starts near
ln(79)=4.37), and generates real phrases from its training data such as
`កម្ពុជាគឺទីក្រុងភ្នំពេញ។ខ្ញុំចង់ទៅភ្នំពេញ។`. That is memorization of a
tiny corpus, not language understanding — but it is exactly the milestone
Project 7 asks for: proof the pipeline works end to end.

Two properties are worth calling out. The model is verified to drive loss
below 0.1 on a perfectly predictable sequence, which is the definitive
end-to-end check that forward, backward and optimizer are all correct
together — a subtly wrong gradient anywhere leaves it stuck well above
that. And because the grapheme tokenizer can only emit whole Khmer
grapheme clusters, **structural Unicode validity (section 29, Levels 1–2)
holds by construction rather than having to be learned** — even the
untrained model emits structurally valid Khmer, confirmed by the Project 1
validator.

177 tests in `tests/`, run with `python3 -m pytest tests/`.

[`fonts/`](fonts/) has Noto Serif Khmer and Noto Sans Khmer (SIL OFL,
bundled from [google/fonts](https://github.com/google/fonts)) for
correctly rendering Khmer text in demos, docs, or HTML output.

Try it:

```bash
python3 -m khmer_language "កម្ពុជា"
```

The rest of this document is the full project plan (Projects 2–10, Phases
1–30) that Project 1 is the foundation for.

---

## 1. Project Vision

Build a Khmer-native Artificial Intelligence system that can eventually:

* Read Khmer correctly.
* Write natural Khmer.
* Understand Khmer vocabulary and sentence structure.
* Understand Khmer context and meaning.
* Answer questions in Khmer.
* Translate between Khmer and English.
* Listen to spoken Khmer.
* Convert Khmer speech into accurate Khmer text.
* Speak Khmer naturally.
* Understand Cambodian culture and local context.
* Run locally/offline where practical.
* Be understandable and auditable rather than functioning as a black box.

The long-term objective is to create a complete Khmer AI stack:

**Khmer Text → Khmer Understanding → Khmer LLM → Khmer Speech Recognition → Khmer Speech Synthesis**

The first major milestone is deliberately **text-only**:

> Build a small Khmer language model completely from scratch, understand every component, validate it, and progressively scale it.

Speech recognition and speech synthesis will be developed after the Khmer language model becomes reliable.

---

# 2. Recommended Project Philosophy

Do not begin by trying to create a Khmer version of ChatGPT.

Instead, build the system in layers:

```text
                    KhmerAI
                       │
        ┌──────────────┴──────────────┐
        │                             │
     TEXT AI                       SPEECH AI
        │                             │
        │                    ┌────────┴────────┐
        │                    │                 │
     Khmer LLM              ASR               TTS
        │                    │                 │
        │                    │                 │
  ┌─────┴─────┐        Khmer Speech      Khmer Voice
  │           │        → Khmer Text      ← Khmer Text
  │           │
Tokenizer   Transformer
  │           │
  │       Attention
  │           │
  │       Embeddings
  │           │
  └─────┬─────┘
        │
   Khmer Corpus
        │
 ┌──────┴─────────┐
 │                │
Khmer linguistic  Khmer general
resources         text corpus
```

This gives you a research project where you can understand every layer instead of simply downloading an existing model.

---

# 3. Main Development Phases

## Phase 0 — Khmer Language Foundation

Before training the LLM, build a Khmer language laboratory.

Study and implement:

* Khmer Unicode.
* Khmer consonants.
* Independent vowels.
* Dependent vowels.
* Subscript consonants.
* Diacritics.
* Sign characters.
* Numerals.
* Punctuation.
* Khmer word boundaries.
* Khmer sentence boundaries.
* Khmer normalization.
* Khmer spelling variations.
* Khmer punctuation conventions.
* Khmer-to-Latin transliteration.
* Khmer phonetic representation.

Create a machine-readable Khmer linguistic database.

Example:

```text
ក
├── Unicode: U+1780
├── Type: consonant
├── Khmer name: ក
├── Transliteration: k
├── Pronunciation information
└── Related subscripts:
      ្ក
```

For a dependent vowel:

```text
ា
├── Unicode: U+17B6
├── Type: dependent vowel
├── Position: after consonant
└── Combination examples:
      ក + ា = កា
```

And for Khmer clusters:

```text
ក + ្ + រ
       ↓
      ក្រ
```

The important point is that the project should understand that these are not necessarily independent "letters" in the same sense as English alphabet characters.

---

# 4. Build a Khmer Unicode Analyzer

Create your own program:

```text
khmer_unicode/
```

with modules such as:

```text
normalizer.py
character_types.py
grapheme.py
syllable.py
cluster.py
validator.py
transliterator.py
```

Example:

```python
text = "កម្ពុជា"

result = analyze(text)
```

Output conceptually:

```text
ក     CONSONANT
ម     CONSONANT
្     SUBSCRIPT_MARK
ព     SUBSCRIPT_CONSONANT
ុ     VOWEL
ជ     CONSONANT
ា     VOWEL
```

This becomes an important foundation for the entire project.

---

# 5. Build a Khmer Grapheme Engine

This is one of the most important components.

Do not immediately tokenize Khmer as individual Unicode code points.

Instead investigate:

```text
Unicode character
       ↓
Unicode sequence
       ↓
Khmer grapheme
       ↓
Khmer syllable
       ↓
Khmer word
       ↓
Khmer sentence
```

For example:

```text
កម្ពុជា
```

should be treated as a structured Khmer linguistic sequence rather than blindly treating every Unicode code point as an independent semantic unit.

Build:

```text
KhmerGraphemeTokenizer
KhmerSyllableTokenizer
KhmerWordSegmenter
KhmerSentenceSegmenter
```

---

# 6. Build Your First Khmer Corpus

The LLM cannot learn Khmer without Khmer text.

Create a dataset pipeline:

```text
Raw Khmer Data
      ↓
Collection
      ↓
Copyright / License Check
      ↓
Deduplication
      ↓
Unicode Normalization
      ↓
Quality Filtering
      ↓
Language Identification
      ↓
Sentence Segmentation
      ↓
Document Classification
      ↓
Training Dataset
```

Separate the corpus into domains:

```text
01_general/
02_news/
03_books/
04_education/
05_history/
06_culture/
07_science/
08_technology/
09_business/
10_law/
11_wikipedia/
12_conversation/
13_social/
14_translation/
15_cambodian_local/
```

Do not mix everything blindly.

You want to know exactly what the model has learned from.

---

# 7. Khmer Dataset Quality System

Create a quality score for every document.

For example:

```text
quality_score =
    language_score
    + unicode_score
    + grammar_score
    + duplication_score
    + source_score
    + readability_score
```

Remove:

* corrupted Unicode;
* duplicated documents;
* machine-generated spam;
* excessive advertisements;
* broken HTML;
* meaningless character sequences;
* mixed-language garbage;
* extremely repetitive documents.

Keep metadata:

```json
{
  "id": "kh_000001",
  "source": "book",
  "domain": "education",
  "language": "km",
  "license": "...",
  "quality": 0.94
}
```

This becomes extremely valuable later when debugging the model.

---

# 8. Build a Khmer Vocabulary Database

Create a separate lexical resource.

For each word:

```text
word
frequency
part_of_speech
definition
English_translation
synonyms
antonyms
example_sentences
pronunciation
syllables
root_information
domain
```

Example:

```json
{
  "word": "កម្ពុជា",
  "english": "Cambodia",
  "type": "proper_noun",
  "domain": "geography",
  "examples": [
    "ប្រទេសកម្ពុជាស្ថិតនៅអាស៊ីអាគ្នេយ៍។"
  ]
}
```

Do not force the LLM to perform all of this work itself.

The linguistic database and the neural model should complement each other.

---

# 9. Study the w3cj Project

Use:

[w3cj/how-llms-work](https://github.com/w3cj/how-llms-work?utm_source=chatgpt.com)

as the **educational foundation**, not necessarily as the final production architecture.

The repository is particularly useful because its progression is:

```text
Pattern matching
      ↓
Neural network
      ↓
BPE tokenizer
      ↓
Word2Vec embeddings
      ↓
Transformer
      ↓
GPT-style generation
```

It even implements the transformer operations manually, including:

* multi-head causal self-attention;
* layer normalization;
* feed-forward layers;
* backpropagation;
* Adam optimization.

That makes it excellent for learning how the components work internally.

---

# 10. Build Your Own Khmer Tokenizer

This should be one of the first real AI experiments.

Test several tokenization strategies.

### Version A — Character tokenizer

```text
ក
ម
្
ព
ុ
ជ
ា
```

Advantages:

* Very simple.
* Handles unknown words.
* Excellent for understanding Khmer morphology.

Disadvantages:

* Very long sequences.
* Training becomes inefficient.

---

### Version B — Grapheme tokenizer

```text
ក
ម
្ព
ុ
ជា
```

Better representation of Khmer writing.

---

### Version C — Word tokenizer

```text
ខ្ញុំ
ចូលចិត្ត
កម្ពុជា
```

Advantages:

* Shorter sequences.

Disadvantages:

* Huge vocabulary.
* Unknown words.
* Difficult with new words.

---

### Version D — Khmer BPE

Train BPE on the Khmer corpus.

The w3cj repository already provides a from-scratch BPE implementation that can serve as your starting point.

However, modify it for Khmer.

Experiment with:

```text
character-level
grapheme-level
syllable-level
BPE
Unigram
hybrid Khmer tokenizer
```

Then compare:

```text
Vocabulary size
Sequence length
Compression ratio
Unknown token rate
Training speed
Validation loss
Generation quality
```

This can become a proper research experiment by itself.

---

# 11. Build Embeddings From Scratch

Next implement:

```text
Khmer tokens
      ↓
Embedding matrix
      ↓
Vector representation
```

Start with Word2Vec.

The w3cj project uses Word2Vec Skip-Gram with negative sampling.

Train:

```text
Khmer Word2Vec
```

Then inspect whether semantically related words become close together.

For example:

```text
កម្ពុជា
ថៃ
វៀតណាម
ឡាវ
```

should develop relationships because of their contexts.

Later compare:

```text
Word2Vec
vs
learned Transformer embeddings
```

---

# 12. Build Khmer GPT-0

Your first real language model should be extremely small.

Example:

```text
KhmerGPT-0

Parameters:
~1M–10M

Layers:
2–4

Attention heads:
2–4

Embedding:
128–256

Context:
128–256 tokens
```

Do not worry about intelligence yet.

The goal is:

> Can the model learn the statistical structure of Khmer text?

Train it to predict:

```text
P(next token | previous tokens)
```

Example:

```text
ខ្ញុំ ចូលចិត្ត ___
```

The model learns probabilities for:

```text
ញ៉ាំ
អាន
ស្តាប់
មើល
...
```

---

# 13. Build the Transformer Yourself

Implement:

```text
Token Embedding
      ↓
Positional Encoding
      ↓
Transformer Block
      ↓
LayerNorm
      ↓
Self Attention
      ↓
Feed Forward
      ↓
Residual Connection
      ↓
Transformer Block
      ↓
Linear Head
      ↓
Softmax
```

Do not hide this behind a framework during the learning stage.

Implement:

```python
class MultiHeadAttention:
    ...

class FeedForward:
    ...

class LayerNorm:
    ...

class TransformerBlock:
    ...

class KhmerGPT:
    ...
```

After the from-scratch implementation works, move to PyTorch for efficient experiments.

This gives you two implementations:

```text
Educational implementation
        +
Production implementation
```

---

# 14. Training Stages

Do not jump directly to instruction tuning.

Use:

### Stage 1 — Character prediction

```text
Khmer characters
```

### Stage 2 — Grapheme prediction

```text
Khmer graphemes
```

### Stage 3 — Word/subword prediction

```text
Khmer tokens
```

### Stage 4 — Sentence modeling

```text
Khmer sentences
```

### Stage 5 — Document modeling

```text
Longer Khmer documents
```

### Stage 6 — Instruction tuning

Teach:

```text
User:
តើប្រទេសកម្ពុជាមានរាជធានីអ្វី?

AI:
រាជធានីរបស់ប្រទេសកម្ពុជាគឺទីក្រុងភ្នំពេញ។
```

---

# 15. Create a Khmer Instruction Dataset

After pretraining, create:

```text
Khmer instruction → answer
```

Examples:

```text
Question answering
Summarisation
Translation
Explanation
Conversation
Classification
Grammar correction
Spelling correction
Rewriting
Reasoning
```

Dataset structure:

```json
{
  "instruction": "ពន្យល់អំពីថាមពលពន្លឺព្រះអាទិត្យ",
  "input": "",
  "output": "ថាមពលពន្លឺព្រះអាទិត្យ..."
}
```

---

# 16. Khmer Cultural Knowledge

A Khmer AI should not only understand Khmer grammar.

It should understand Cambodia.

Build datasets around:

```text
Cambodian history
Cambodian geography
Cambodian provinces
Cambodian culture
Cambodian traditions
Cambodian food
Cambodian education
Cambodian government terminology
Cambodian businesses
Cambodian agriculture
Cambodian technology
Cambodian engineering
Cambodian renewable energy
Cambodian environment
Cambodian everyday conversation
```

This should be handled carefully with source tracking.

Each knowledge document should retain:

```text
source
date
author
license
domain
confidence
```

---

# 17. Khmer Evaluation System

This is extremely important.

Do not evaluate the model only using loss.

Create:

```text
KhmerAI Benchmark
```

with categories:

### Khmer writing

```text
Spelling
Unicode correctness
Punctuation
Word segmentation
Grammar
Naturalness
```

### Khmer understanding

```text
Question answering
Reading comprehension
Summarisation
Inference
```

### Khmer knowledge

```text
Cambodian history
Geography
Culture
Science
Technology
```

### English ↔ Khmer

```text
Translation
Terminology
Technical translation
```

### Reasoning

```text
Mathematics
Logic
Multi-step reasoning
```

---

# 18. Build a Khmer Error Analyzer

Every model answer should be analyzable.

Example:

```text
Input:
ខ្ញុំចង់ទៅភ្នំពេញនៅថ្ងៃស្អែក។

Output:
ខ្ញុំចង់ទៅភ្នំពេញថ្ងៃស្អែក។

Analysis:

Unicode:          PASS
Spelling:         PASS
Grammar:          PASS
Meaning:          PASS
Naturalness:      0.91
```

This becomes much more useful than simply saying:

```text
Loss = 2.31
```

---

# 19. Model Development Roadmap

Use these model generations:

```text
KhmerGPT-0
    ↓
KhmerGPT-1
    ↓
KhmerGPT-2
    ↓
KhmerGPT-3
    ↓
KhmerGPT-Instruct
    ↓
KhmerAI
```

### KhmerGPT-0

Tiny model.

Purpose:

```text
prove the pipeline
```

### KhmerGPT-1

10–50M parameters.

Purpose:

```text
learn Khmer language structure
```

### KhmerGPT-2

50–300M parameters.

Purpose:

```text
useful Khmer language generation
```

### KhmerGPT-3

300M–1B+ parameters.

Purpose:

```text
stronger Khmer understanding
```

### KhmerGPT-Instruct

Instruction tuned.

Purpose:

```text
conversation
question answering
reasoning
```

---

# 20. Important Reality Check About "From Scratch"

There are two meanings of "from scratch."

### Educational from scratch

You implement:

```text
Tokenizer
Embedding
Attention
Transformer
Loss
Backpropagation
Optimizer
Training loop
Inference
```

This is absolutely achievable as a personal research project.

### Industrial-scale from scratch

Training a genuinely competitive multi-billion-parameter LLM entirely from random initialization requires enormous:

```text
dataset
GPU compute
storage
training time
engineering
evaluation
```

Therefore, your project should have both goals:

```text
FROM-SCRATCH RESEARCH MODEL
             +
PRACTICAL KHMER AI MODEL
```

Do not sacrifice the educational goal just to make the biggest model possible.

---

# 21. Recommended Software Architecture

```text
khmer-ai/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── khmer-language/
│   ├── unicode/
│   ├── grapheme/
│   ├── syllable/
│   ├── tokenizer/
│   ├── dictionary/
│   └── phonology/
│
├── tokenizer/
│   ├── character/
│   ├── grapheme/
│   ├── bpe/
│   └── unigram/
│
├── embeddings/
│   └── word2vec/
│
├── models/
│   ├── from_scratch/
│   ├── pytorch/
│   └── checkpoints/
│
├── training/
│   ├── pretraining/
│   ├── instruction/
│   └── evaluation/
│
├── evaluation/
│   ├── spelling/
│   ├── grammar/
│   ├── comprehension/
│   ├── translation/
│   └── reasoning/
│
├── inference/
│
├── api/
│
├── frontend/
│
├── speech/
│   ├── asr/
│   └── tts/
│
└── documentation/
```

> Implementation note: the code in this repo uses `khmer_language/` (underscore)
> as the actual Python package name, since `khmer-language` is not a valid
> Python import name. The `unicode/` submodule currently holds the
> `character_types.py`, `grapheme.py`, `syllable.py`, `cluster.py`,
> `validator.py`, `transliterator.py` and `codepoints.py` modules described
> in section 4, rather than splitting each into its own top-level package —
> that split can happen later if/when `tokenizer/`, `dictionary/` and
> `phonology/` grow enough to justify it.

---

# 22. Later: Khmer ASR

Only begin ASR after the text model is functioning.

Architecture:

```text
Microphone
    ↓
Audio preprocessing
    ↓
Khmer ASR
    ↓
Khmer text
    ↓
Khmer LLM
```

There is already useful Khmer speech data to build upon.

The 2026 Khmer ASR Cultural Dataset V3 contains 134.6 hours of manually curated Khmer speech-text pairs and is licensed CC-BY-SA-4.0.

There is also a smaller OpenSLR-derived Khmer speech dataset containing about 4 hours and 2,906 audio files.

More importantly, recent Khmer ASR work demonstrates that much larger Khmer speech resources are now available; one published model reports training on approximately 700 hours of Khmer speech and reports 1.96% in-domain CER and 7.91% out-of-domain CER in its own evaluation.

For your eventual system:

```text
ASR v1
↓
fine-tune existing multilingual ASR

ASR v2
↓
Khmer-specialized ASR

ASR v3
↓
your own Khmer speech model
```

Do not build the acoustic model from zero until you have a strong reason to.

---

# 23. Later: Khmer TTS

After ASR:

```text
Khmer text
     ↓
Text normalization
     ↓
Khmer phonemizer
     ↓
Acoustic model
     ↓
Vocoder
     ↓
Khmer speech
```

Investigate:

```text
VITS
FastSpeech
Tacotron
MMS-TTS
XTTS-style architectures
```

There is already an open-source Khmer word-level TTS project called KLEA, based on VITS and trained using Khmer speech resources. It demonstrates that Khmer-specific TTS work is feasible, although its stated scope is individual words rather than full sentences.

Your eventual TTS dataset should contain:

```text
speaker
gender
age
recording environment
audio
transcript
phonemes
duration
```

Ideally collect multiple Cambodian speakers rather than building the system around a single voice.

---

# 24. Complete KhmerAI Architecture

Eventually:

```text
                         ┌─────────────────┐
                         │     USER        │
                         └────────┬────────┘
                                  │
                         Voice / Text
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                  TEXT                        AUDIO
                    │                           │
                    │                          ASR
                    │                           │
                    └──────────────┬────────────┘
                                   │
                              Khmer Text
                                   │
                          ┌────────▼────────┐
                          │ Khmer NLP Layer │
                          └────────┬────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
          Linguistic Engine                     Khmer LLM
                 │                                   │
                 │                            ┌──────┴──────┐
                 │                            │             │
                 │                        Knowledge     Generation
                 │                            │             │
                 └────────────────────────────┴─────────────┘
                                   │
                              Khmer Response
                                   │
                       ┌───────────┴───────────┐
                       │                       │
                     TEXT                    TTS
                       │                       │
                       │                 Khmer Speech
                       │                       │
                       └───────────┬───────────┘
                                   │
                                USER
```

---

# 25. Hardware Roadmap

Do not use the Raspberry Pi Zero 2 W for training the LLM.

It can eventually be useful as an **edge client**:

```text
Pi Zero 2 W
    ↓
Microphone
    ↓
ASR
    ↓
Network
    ↓
KhmerAI server
    ↓
TTS
    ↓
Speaker
```

For development:

### Stage 1

A normal desktop/laptop GPU is sufficient.

### Stage 2

Use a stronger NVIDIA GPU.

### Stage 3

Use a multi-GPU workstation or cloud GPU for larger experiments.

### Stage 4

Deploy a quantized Khmer model to:

```text
Raspberry Pi
Jetson
mini PC
local server
```

---

# 26. Recommended Technology Stack

### Language

```text
Python
```

### Learning implementation

```text
Python + NumPy
```

### Production training

```text
PyTorch
```

### Data

```text
JSONL
Parquet
Arrow
```

### Dataset processing

```text
Hugging Face Datasets
pandas
Polars
```

### Tokenization

```text
Your own tokenizer
+
SentencePiece/tokenizers for comparison
```

### Experiment tracking

```text
Weights & Biases
or
MLflow
```

### API

```text
FastAPI
```

### UI

```text
React / Next.js
```

### Speech later

```text
Whisper/Qwen-ASR-style baseline
+
Khmer-specific fine-tuning
```

---

# 27. Recommended Research Sequence

Follow this exact order:

```text
01. Khmer Unicode
        ↓
02. Khmer character database
        ↓
03. Grapheme parser
        ↓
04. Syllable parser
        ↓
05. Word segmentation
        ↓
06. Khmer corpus collection
        ↓
07. Corpus cleaning
        ↓
08. Character tokenizer
        ↓
09. Grapheme tokenizer
        ↓
10. BPE tokenizer
        ↓
11. Tokenizer comparison
        ↓
12. Word2Vec
        ↓
13. Tiny neural language model
        ↓
14. Attention from scratch
        ↓
15. Transformer from scratch
        ↓
16. KhmerGPT-0
        ↓
17. KhmerGPT-1
        ↓
18. Large-scale pretraining
        ↓
19. Instruction tuning
        ↓
20. Khmer evaluation benchmark
        ↓
21. KhmerAI text interface
        ↓
22. Khmer ASR
        ↓
23. Khmer TTS
        ↓
24. Voice conversation
        ↓
25. Local/offline deployment
```

---

# 28. The First 10 Projects

Instead of treating this as one enormous project, make it ten connected projects.

### Project 1 — Khmer Unicode Explorer

Input:

```text
កម្ពុជា
```

Output:

```text
Unicode
character type
grapheme
cluster
```

**Status: implemented** — see `khmer_language/` and the Status section at the top of this file.

---

### Project 2 — Khmer Grapheme & Syllable Parser

Build:

```text
Khmer text
↓
graphemes
↓
syllables
```

**Status: implemented** as part of Project 1 (`khmer_language/unicode/grapheme.py`,
`syllable.py`, `sentence.py`). `word.py` has a v1 `KhmerWordSegmenter` that
only splits on explicit boundary hints (ZWSP/whitespace/punctuation);
dictionary-free segmentation of continuous Khmer needs a corpus first
(see Project 3).

---

### Project 3 — Khmer Corpus Builder

Automatically:

```text
collect
clean
normalize
deduplicate
validate
```

---

### Project 4 — Khmer Tokenizer Lab

Compare:

```text
character
grapheme
syllable
word
BPE
Unigram
```

**Status: first version implemented** — `khmer_language/tokenizer/`
(character, grapheme, syllable, grapheme-aware BPE + comparison harness).
Unigram tokenization and training on a real corpus are not done yet
(waiting on Project 3).

---

### Project 5 — Khmer Word2Vec

**Status: implemented** — `khmer_language/embeddings/word2vec.py`
(skip-gram + negative sampling from scratch in NumPy). Visualization is
not built yet.

---

### Project 5 — Khmer Word2Vec

Train embeddings and visualize them.

---

### Project 6 — Transformer From Scratch

Use the w3cj project as the educational reference.

**Status: implemented** — `models/from_scratch/` (layers, attention,
transformer block, optimizer), every backward pass gradient-checked.

---

### Project 7 — KhmerGPT-0

Train the first Khmer language model.

**Status: implemented** — `models/from_scratch/gpt.py` +
`khmer_language/training/`. Trains end to end on the placeholder corpus
(`--train-demo`). Training on a real corpus awaits Project 3.

---

### Project 8 — KhmerGPT-Instruct

Teach the model to follow Khmer instructions.

---

### Project 9 — KhmerAI

Create a usable chatbot:

```text
Khmer text
↓
Khmer LLM
↓
Khmer response
```

---

### Project 10 — Khmer Voice AI

Add:

```text
Microphone
↓
Khmer ASR
↓
Khmer LLM
↓
Khmer TTS
↓
Speaker
```

---

# 29. What "Success" Looks Like

Do not define success as:

> "My model has X billion parameters."

Instead define measurable milestones.

### Level 1

The model can generate valid Khmer Unicode.

### Level 2

The model can generate valid Khmer graphemes.

### Level 3

The model generates valid Khmer words.

### Level 4

The model generates grammatical sentences.

### Level 5

The model understands simple Khmer questions.

### Level 6

The model can maintain context.

### Level 7

The model can answer Khmer questions reliably.

### Level 8

The model can reason in Khmer.

### Level 9

The model understands Cambodian cultural context.

### Level 10

The model can listen and speak Khmer.

---

# 30. Final Project Goal

The final system should look like:

```text
                    KHMERAI
                       │
        ┌──────────────┼──────────────┐
        │              │              │
       READ           THINK          SPEAK
        │              │              │
      Khmer          Khmer           Khmer
       NLP            LLM             TTS
        │              │              │
        └──────────────┼──────────────┘
                       │
                      HEAR
                       │
                      ASR
```

The important research contribution is not simply creating another chatbot.

The goal is to create a **Khmer-native AI technology stack** where you understand the entire pipeline:

```text
Khmer Unicode
     ↓
Khmer linguistic structure
     ↓
Khmer tokenizer
     ↓
Khmer embeddings
     ↓
Khmer Transformer
     ↓
Khmer language model
     ↓
Khmer instruction model
     ↓
Khmer ASR
     ↓
Khmer TTS
     ↓
Khmer conversational AI
```

That gives you a project that can begin as a small personal experiment and eventually grow into a serious Khmer NLP/AI research platform.

## Recommended first milestone

Do **not** start training the LLM yet.

Start with:

**`Project 1 — Khmer Language & Unicode Engine`** ✅ done, see Status above.

Then move to:

**`Project 2 — Khmer Corpus & Tokenizer Lab`** ← next up.

Then:

**`Project 3 — Transformer From Scratch`**

Then combine them into:

**`Project 4 — KhmerGPT-0`**

That sequence will let you genuinely understand *why* the Khmer model works rather than simply fine-tuning someone else's model.
