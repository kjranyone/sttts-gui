"""文チャンク分割。

Irodori-TTS-Server の SSE モードと同じ発想: 句読点(。!?…改行)で区切り、
最小文字数を満たした時点でチャンク確定。最初のチャンクは最速で発話できるよう
しきい値を緩める(first_chunk_min_chars)。
"""

from __future__ import annotations

import re

# チャンク確定のトリガになる文字
_DELIMITERS = set("。!?!?\n…")

# 後続の閉じ括弧等は直前の区切り文字と一体とみなして同じチャンクに含める
_TRAILERS = set("」』)）】〉》”'\"♪")


def split_chunks(
    text: str,
    min_chars: int = 16,
    first_min_chars: int = 1,
) -> list[str]:
    """テキストを発話チャンクに分割する。

    - 区切り文字(とその直後の閉じ括弧)でチャンクを閉じる
    - 先頭チャンクは first_min_chars、以降は min_chars を満たした時点で確定
    - 末尾の余りが短い場合は直前のチャンクに連結する(単独チャンクのときはそのまま出す)
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    count = 0  # buf 内の非空白文字数

    i = 0
    while i < len(text):
        ch = text[i]
        buf.append(ch)
        if not ch.isspace():
            count += 1

        if ch in _DELIMITERS:
            # 直後の閉じ括弧を連結する
            j = i + 1
            while j < len(text) and text[j] in _TRAILERS:
                buf.append(text[j])
                j += 1
            i = j - 1
            threshold = first_min_chars if not chunks else min_chars
            if count >= threshold:
                chunk = "".join(buf).strip()
                if chunk:
                    chunks.append(chunk)
                buf = []
                count = 0
        i += 1

    rest = "".join(buf).strip()
    if rest:
        rest_count = sum(1 for c in rest if not c.isspace())
        threshold = first_min_chars if not chunks else min_chars
        if chunks and rest_count < threshold:
            chunks[-1] += rest
        else:
            chunks.append(rest)

    return [c for c in chunks if c]
