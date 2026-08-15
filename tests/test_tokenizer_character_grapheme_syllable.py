from khmer_language.tokenizer.character import CharacterTokenizer
from khmer_language.tokenizer.grapheme import GraphemeTokenizer
from khmer_language.tokenizer.syllable import SyllableTokenizer

WORD = "កម្ពុជា"


def test_character_tokenizer_splits_every_codepoint():
    tok = CharacterTokenizer()
    assert tok.tokenize(WORD) == list(WORD)


def test_grapheme_tokenizer_matches_engine():
    tok = GraphemeTokenizer()
    assert tok.tokenize(WORD) == ["ក", "ម្ពុ", "ជា"]


def test_syllable_tokenizer_matches_engine():
    tok = SyllableTokenizer()
    assert tok.tokenize(WORD) == ["ក", "ម្ពុ", "ជា"]


def test_grapheme_tokenizer_produces_shorter_sequences_than_character():
    char_tok = CharacterTokenizer()
    graph_tok = GraphemeTokenizer()
    assert len(graph_tok.tokenize(WORD)) < len(char_tok.tokenize(WORD))


def test_encode_decode_round_trip_after_training():
    for cls in (CharacterTokenizer, GraphemeTokenizer, SyllableTokenizer):
        tok = cls()
        tok.train([WORD, "ជាប្រទេស"])
        assert tok.decode(tok.encode(WORD)) == WORD


def test_unseen_grapheme_becomes_unk_not_a_crash():
    tok = GraphemeTokenizer()
    tok.train(["ក"])
    ids = tok.encode("ខ")  # not in training data
    assert ids == [tok.vocab.encode_token("<UNK>")]
