from sttts_server.chunker import count_mora, split_chunks


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


def test_short_remainder_is_not_merged_into_last_chunk():
    # 末尾の短い余りを連結すると最終チャンクが長くなり間が空くため、単独で出す
    text = "これは十分に長い文です。あとがき"
    chunks = split_chunks(text, min_chars=16, first_min_chars=1)
    assert chunks == ["これは十分に長い文です。", "あとがき"]


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
    assert chunks == ["一行目です", "二行目です"]


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


def _joined(chunks):
    return "".join(chunks).replace(" ", "")


def test_fullwidth_exclamation_and_question_are_delimiters():
    text = "本当ですか？本当に！そうなんだ…"
    assert split_chunks(text, min_chars=3, first_min_chars=1) == ["本当ですか？", "本当に！", "そうなんだ…"]


def test_symbol_only_remainder_sticks_to_previous():
    assert split_chunks("こんにちは。♪", min_chars=16) == ["こんにちは。♪"]


def test_first_chunk_cut_at_comma():
    text = "こんにちは、今日はいい天気ですね。"
    assert split_chunks(text) == ["こんにちは、", "今日はいい天気ですね。"]


def test_very_short_comma_prefix_is_not_cut():
    # 「はい、」(2モーラ)だけを先に出すことはしない
    chunks = split_chunks("はい、そうです。それでは次の話題に移りましょう。")
    assert chunks[0] == "はい、そうです。"


def test_first_chunk_cut_on_unpunctuated_asr_text():
    # kotoba / ReazonSpeech は句読点を出さない → 文節境界の近似で 8〜12 モーラ付近を切る
    text = "今日は自然言語処理の最新の研究について話したいと思います"
    chunks = split_chunks(text)
    assert chunks[0] == "今日は自然言語処理の"  # [8,12] に境界が無いので 2*max まで延長
    assert 8 <= count_mora(chunks[0]) <= 24
    assert _joined(chunks) == text
    chunks = split_chunks("えーと今日はですね新しいモデルの話をしようと思っていて")
    assert chunks[0] == "えーと今日はですね"
    assert 8 <= count_mora(chunks[0]) <= 12


def test_first_chunk_mora_window_is_configurable():
    text = "今日は自然言語処理の最新の研究について話したいと思います"
    loose = split_chunks(text, first_mora_min=16, first_mora_max=24)
    assert 16 <= count_mora(loose[0]) <= 24
    assert split_chunks(text, first_mora_max=0) == [text]  # 無効化


def test_slightly_long_first_sentence_is_not_split_into_tiny_tail():
    chunks = split_chunks("これは三つ目の発話です。チャンク分割を確認します。")
    assert chunks == ["これは三つ目の発話です。", "チャンク分割を確認します。"]


def test_long_unpunctuated_text_is_split_by_length():
    text = "えーと今日はですね新しいモデルの話をしようと思っていて" * 4
    chunks = split_chunks(text, max_chars=30)
    assert len(chunks) >= 4
    assert all(len(c) <= 30 for c in chunks)
    assert _joined(chunks) == text


def test_length_split_prefers_comma():
    first = "はい。"
    body = "これは長い説明の前半部分なのですが、ここから後半の説明が長く続いていきます"
    chunks = split_chunks(first + body, max_chars=24, first_mora_max=0)
    assert chunks[0] == "はい。"
    assert chunks[1].endswith("、")


def test_later_chunks_respect_min_chars():
    text = "最初です。短い。短い。短い。これで終わりです。"
    chunks = split_chunks(text, min_chars=8)
    assert chunks[0] == "最初です。"
    assert all(len(c) >= 8 for c in chunks[1:-1])
    assert "".join(chunks) == text


def test_count_mora_approximation():
    assert count_mora("きょう") == 2  # 拗音は0
    assert count_mora("東京") == 4
    assert count_mora("、。 ") == 0
