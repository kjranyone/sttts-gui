"""文チャンク分割(TTS の初音を早めるための分割)。

方針:
1. 文末記号(。！？!?…改行)で文に区切り、最小文字数を満たした時点でチャンク確定
   (先頭は first_min_chars、以降は min_chars)。
2. 先頭チャンクだけはさらに短く切る: 読点(、,,)か、約 first_mora_min〜first_mora_max
   モーラの自然な切れ目(空白 / ひらがな→非ひらがなの境界 = 文節境界の近似)で切る。
   Irodori は1チャンク全体を一括生成するため、初音までの時間は先頭チャンク長に比例する。
3. 句読点の無いテキスト(kotoba-whisper / ReazonSpeech の出力)や長すぎるチャンクは
   max_chars を超えないよう長さベースで分割する(読点 > 自然な切れ目 > 強制)。
4. 末尾の短い余りは直前に連結しない(連結すると最終チャンクが長くなり間が空く)。
   ただし文字(かな・漢字・英数字)を含まない余り(記号のみ等)は直前に連結する。
"""

from __future__ import annotations

# チャンク確定のトリガになる文字(全角 ！？ も含む)
_DELIMITERS = set("。．!?！？‼⁇⁈⁉\n…")

# 後続の閉じ括弧等は直前の区切り文字と一体とみなして同じチャンクに含める
_TRAILERS = set("」』)）】〉》”’'\"♪")

# 読点(先頭チャンクの切り位置・長さ分割の第一候補)
_COMMAS = set("、,,､")

_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")


def _is_hiragana(ch: str) -> bool:
    return "\u3041" <= ch <= "\u309f"


def _is_katakana(ch: str) -> bool:
    return "\u30a0" <= ch <= "\u30ff" or ch == "ー"


def _is_kanji(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" or ch in "々〆"


def _is_letter(ch: str) -> bool:
    return _is_hiragana(ch) or _is_katakana(ch) or _is_kanji(ch) or ch.isalnum()


def char_mora(ch: str) -> float:
    """1文字のおおよそのモーラ数(読み推定なしの近似)。"""
    if ch in _SMALL_KANA:
        return 0.0
    if _is_hiragana(ch) or _is_katakana(ch):
        return 1.0
    if _is_kanji(ch):
        return 2.0  # 音読み2モーラ前後が多い
    if ch.isdigit():
        return 2.0
    if ch.isascii() and ch.isalpha():
        return 0.7  # 英字はおおよそ
    return 0.0


def count_mora(text: str) -> float:
    return sum(char_mora(c) for c in text)


def _letters(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def _is_natural_boundary(text: str, i: int) -> bool:
    """text[:i] と text[i:] の間が自然な切れ目か(空白、ひらがな→非ひらがな)。"""
    if i <= 0 or i >= len(text):
        return False
    prev, nxt = text[i - 1], text[i]
    if prev.isspace() or nxt.isspace():
        return True
    return _is_hiragana(prev) and _is_letter(nxt) and not _is_hiragana(nxt)


def _cut_first(text: str, mora_min: float, mora_max: float) -> int | None:
    """先頭チャンクの切り位置(text[:pos] が先頭)。切らない場合 None。"""
    total = count_mora(text)
    if mora_max <= 0 or total <= mora_max:
        return None
    prefix = [0.0]
    for c in text:
        prefix.append(prefix[-1] + char_mora(c))

    def tail_ok(pos: int) -> bool:
        # 残りが短いなら切らない(短い2チャンクより1チャンクの方が自然で、速度差も小さい)
        return total - prefix[pos] >= max(3.0, mora_min)

    # 1) 読点: 先頭が max(4, mora_min/2) 以上、2*mora_max 以下の最初の読点の直後
    #    (読点は元々ポーズが入る位置なので、少し短くても切る。「はい、」程度は切らない)
    comma_min = max(4.0, mora_min / 2)
    for i, c in enumerate(text):
        if c in _COMMAS and comma_min <= prefix[i + 1] <= 2 * mora_max:
            pos = i + 1
            while pos < len(text) and text[pos] in _TRAILERS:
                pos += 1
            if tail_ok(pos):
                return pos
    # 2) 自然な切れ目: [mora_min, mora_max] の中で最も後ろ、無ければ 2*mora_max まで延長して最初
    best = None
    for i in range(1, len(text)):
        if mora_min <= prefix[i] <= mora_max and _is_natural_boundary(text, i) and tail_ok(i):
            best = i
    if best is not None:
        return best
    for i in range(1, len(text)):
        if mora_max < prefix[i] <= 2 * mora_max and _is_natural_boundary(text, i) and tail_ok(i):
            return i
    return None


def _split_long(text: str, min_chars: int, max_chars: int) -> list[str]:
    """max_chars を超える塊を 読点 > 自然な切れ目 > 強制 の優先で分割する。"""
    out: list[str] = []
    rest = text
    while max_chars > 0 and _letters(rest) > max_chars:
        lo = max(1, min(min_chars, max_chars - 1))
        hi = min(len(rest) - 1, max_chars)
        cut = None
        for i in range(hi, lo - 1, -1):  # 読点(直後で切る)
            if rest[i - 1] in _COMMAS:
                cut = i
                break
        if cut is None:
            for i in range(hi, lo - 1, -1):
                if _is_natural_boundary(rest, i):
                    cut = i
                    break
        if cut is None:
            cut = hi
        head, rest = rest[:cut].strip(), rest[cut:].strip()
        if head:
            out.append(head)
    if rest.strip():
        out.append(rest.strip())
    return out


def _sentences(text: str) -> list[str]:
    """文末記号(+直後の閉じ括弧)で区切った文のリスト(区切り文字は前の文に含める)。"""
    out: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        buf.append(ch)
        if ch in _DELIMITERS:
            j = i + 1
            while j < len(text) and (text[j] in _TRAILERS or text[j] in _DELIMITERS):
                buf.append(text[j])
                j += 1
            i = j - 1
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
        i += 1
    s = "".join(buf).strip()
    if s:
        out.append(s)
    return out


def split_chunks(
    text: str,
    min_chars: int = 16,
    first_min_chars: int = 1,
    *,
    max_chars: int = 80,
    first_mora_min: float = 8,
    first_mora_max: float = 12,
) -> list[str]:
    """テキストを発話チャンクに分割する(詳細はモジュール docstring)。

    first_mora_max <= 0 で先頭チャンクの短縮を無効化、max_chars <= 0 で長さ分割を無効化。
    """
    text = text.strip()
    if not text:
        return []

    # 1) 文をしきい値まで束ねる
    merged: list[str] = []
    buf = ""
    for sent in _sentences(text):
        buf = buf + sent if buf else sent
        threshold = first_min_chars if not merged else min_chars
        if _letters(buf) >= threshold:
            merged.append(buf)
            buf = ""
    if buf:
        if merged and not any(_is_letter(c) for c in buf):
            merged[-1] += buf  # 記号だけの余りは直前へ
        else:
            merged.append(buf)

    # 2) 先頭チャンクを短く切る
    chunks: list[str] = []
    for idx, piece in enumerate(merged):
        if idx == 0:
            pos = _cut_first(piece, first_mora_min, first_mora_max)
            if pos is not None:
                head, tail = piece[:pos].strip(), piece[pos:].strip()
                chunks.append(head)
                if tail:
                    chunks.extend(_split_long(tail, min_chars, max_chars))
                continue
            if max_chars > 0 and _letters(piece) > max_chars:
                chunks.extend(_split_long(piece, min_chars, max_chars))
                continue
            chunks.append(piece)
        else:
            # 3) 長すぎるチャンクは長さで分割
            chunks.extend(_split_long(piece, min_chars, max_chars))
    return [c for c in chunks if c]
