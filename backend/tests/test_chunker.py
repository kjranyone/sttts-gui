from sttts_server.chunker import split_chunks


def test_single_sentence():
    assert split_chunks("こんにちは。") == ["こんにちは。"]


def test_two_sentences_with_min_chars():
    text = "最初の文です。二つ目の文はここから始まり、しばらく続きます。"
    chunks = split_chunks(text, min_chars=16, first_min_chars=1)
    assert chunks[0] == "最初の文です。"
    assert "".join(chunks) == text


def test_first_chunk_threshold():
    # first_min_chars=3 なら「短い。」(3文字)で確定、以降は min_chars=5 で判定
    text = "短い。二文目も短い。"
    chunks = split_chunks(text, min_chars=5, first_min_chars=3)
    assert chunks == ["短い。", "二文目も短い。"]


def test_threshold_not_met_continues_accumulating():
    # first_min_chars=10 なら「こんにちは。」(6文字)では確定せず末尾まで1チャンク
    text = "こんにちは。元気ですか。"
    chunks = split_chunks(text, min_chars=16, first_min_chars=10)
    assert chunks == ["こんにちは。元気ですか。"]


def test_remainder_merges_into_last_chunk():
    text = "これは十分に長い文です。あとがき"
    chunks = split_chunks(text, min_chars=16, first_min_chars=1)
    assert chunks == ["これは十分に長い文です。あとがき"]


def test_remainder_long_enough_stands_alone():
    text = "一文目。これは残りとして十分に長い文です。"
    chunks = split_chunks(text, min_chars=16, first_min_chars=1)
    assert chunks == ["一文目。", "これは残りとして十分に長い文です。"]


def test_exclamation_and_question():
    text = "本当ですか?本当に!そうなんだ…"
    chunks = split_chunks(text, min_chars=3, first_min_chars=1)
    assert chunks == ["本当ですか?", "本当に!", "そうなんだ…"]


def test_newline_is_delimiter():
    chunks = split_chunks("一行目です\n二行目です", min_chars=16, first_min_chars=1)
    # しきい値未満の残りは直前チャンクへ連結される
    assert chunks == ["一行目です二行目です"]


def test_closed_quote_sticks_to_delimiter():
    text = "「こんにちは」と彼は言った。次の文です。"
    chunks = split_chunks(text, min_chars=5, first_min_chars=1)
    assert chunks[0] == "「こんにちは」と彼は言った。"
    assert "".join(chunks) == text


def test_empty_and_whitespace():
    assert split_chunks("") == []
    assert split_chunks("   \n  ") == []


def test_trailers_attach_to_delimiter():
    # 「。」直後の閉じ括弧は同じチャンクに含まれる
    text = "はいそうです。)(続きます。"
    chunks = split_chunks(text, min_chars=3, first_min_chars=1)
    assert chunks[0] == "はいそうです。)"
