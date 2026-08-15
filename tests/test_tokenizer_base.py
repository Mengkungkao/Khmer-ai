from khmer_language.tokenizer.base import BOS, EOS, PAD, UNK, Vocabulary
from khmer_language.tokenizer.character import CharacterTokenizer


def test_special_tokens_get_fixed_low_ids():
    vocab = Vocabulary()
    assert [vocab.token_to_id[t] for t in (PAD, UNK, BOS, EOS)] == [0, 1, 2, 3]


def test_add_and_lookup_round_trip():
    vocab = Vocabulary(["ក", "ខ"])
    assert vocab.decode_id(vocab.encode_token("ក")) == "ក"
    assert len(vocab) == 6  # 4 special + 2


def test_unknown_token_maps_to_unk_id():
    vocab = Vocabulary(["ក"])
    assert vocab.encode_token("ខ") == vocab.token_to_id[UNK]


def test_to_dict_from_dict_round_trip():
    vocab = Vocabulary(["ក", "ខ", "គ"])
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.id_to_token == vocab.id_to_token
    assert restored.token_to_id == vocab.token_to_id


def test_unknown_rate():
    tok = CharacterTokenizer()
    tok.train(["abc"])
    assert tok.unknown_rate("abc") == 0.0
    assert tok.unknown_rate("abcd") == 0.25
