//! 文チャンク分割(TTS の初音を早めるための分割)。
//!
//! 方針:
//! 1. 文末記号(。！？!?…改行)で文に区切り、最小文字数を満たした時点でチャンク確定
//!    (先頭は `first_min_chars`、以降は `min_chars`)。
//! 2. 先頭チャンクだけはさらに短く切る: 読点(、,,)か、約 `first_mora_min`〜`first_mora_max`
//!    モーラの自然な切れ目(空白 / ひらがな→非ひらがなの境界 = 文節境界の近似)で切る。
//!    Irodori は1チャンク全体を一括生成するため、初音までの時間は先頭チャンク長に比例する。
//! 3. 句読点の無いテキスト(kotoba-whisper の出力)や長すぎるチャンクは `max_chars` を超えない
//!    よう長さベースで分割する(読点 > 自然な切れ目 > 強制)。
//! 4. 末尾の短い余りは直前に連結しない(連結すると最終チャンクが長くなり間が空く)。
//!    ただし文字(かな・漢字・英数字)を含まない余り(記号のみ等)は直前に連結する。
//!
//! 位置はすべて文字(Unicode スカラ値)単位。Python 実装 `chunker.py` と同じ結果になる。

/// チャンク確定のトリガになる文字(全角 ！？ も含む)
const DELIMITERS: &str = "。．!?！？‼⁇⁈⁉\n…";

/// 後続の閉じ括弧等は直前の区切り文字と一体とみなして同じチャンクに含める
const TRAILERS: &str = "」』)）】〉》”’'\"♪";

/// 読点(先頭チャンクの切り位置・長さ分割の第一候補)
const COMMAS: &str = "、,,､";

const SMALL_KANA: &str = "ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ";

/// 先頭チャンク切り出しの既定(モーラ)
pub const FIRST_MORA_MIN: f64 = 8.0;
pub const FIRST_MORA_MAX: f64 = 12.0;

#[derive(Debug, Clone, Copy)]
pub struct ChunkOptions {
    pub min_chars: usize,
    pub first_min_chars: usize,
    pub max_chars: usize,
    pub first_mora_min: f64,
    pub first_mora_max: f64,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self { min_chars: 16, first_min_chars: 1, max_chars: 80, first_mora_min: FIRST_MORA_MIN, first_mora_max: FIRST_MORA_MAX }
    }
}

fn is_hiragana(c: char) -> bool {
    ('\u{3041}'..='\u{309f}').contains(&c)
}

fn is_katakana(c: char) -> bool {
    ('\u{30a0}'..='\u{30ff}').contains(&c) || c == 'ー'
}

fn is_kanji(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c) || ('\u{3400}'..='\u{4dbf}').contains(&c) || c == '々' || c == '〆'
}

fn is_letter(c: char) -> bool {
    is_hiragana(c) || is_katakana(c) || is_kanji(c) || c.is_alphanumeric()
}

/// 1文字のおおよそのモーラ数(読み推定なしの近似)。
pub fn char_mora(c: char) -> f64 {
    if SMALL_KANA.contains(c) {
        0.0
    } else if is_hiragana(c) || is_katakana(c) {
        1.0
    } else if is_kanji(c) {
        2.0 // 音読み2モーラ前後が多い
    } else if c.is_numeric() {
        2.0
    } else if c.is_ascii_alphabetic() {
        0.7 // 英字はおおよそ
    } else {
        0.0
    }
}

pub fn count_mora(text: &str) -> f64 {
    text.chars().map(char_mora).sum()
}

fn letters(text: &[char]) -> usize {
    text.iter().filter(|c| !c.is_whitespace()).count()
}

fn s(chars: &[char]) -> String {
    chars.iter().collect()
}

/// `text[..i]` と `text[i..]` の間が自然な切れ目か(空白、ひらがな→非ひらがな)。
fn is_natural_boundary(text: &[char], i: usize) -> bool {
    if i == 0 || i >= text.len() {
        return false;
    }
    let (prev, next) = (text[i - 1], text[i]);
    if prev.is_whitespace() || next.is_whitespace() {
        return true;
    }
    is_hiragana(prev) && is_letter(next) && !is_hiragana(next)
}

/// 先頭チャンクの切り位置(`text[..pos]` が先頭)。切らない場合 None。
fn cut_first(text: &[char], mora_min: f64, mora_max: f64) -> Option<usize> {
    let total: f64 = text.iter().map(|&c| char_mora(c)).sum();
    if mora_max <= 0.0 || total <= mora_max {
        return None;
    }
    let mut prefix = vec![0.0f64];
    for &c in text {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + char_mora(c));
    }
    // 残りが短いなら切らない(短い2チャンクより1チャンクの方が自然で、速度差も小さい)
    let tail_ok = |pos: usize| total - prefix[pos] >= 3.0f64.max(mora_min);

    // 1) 読点: 先頭が max(4, mora_min/2) 以上、2*mora_max 以下の最初の読点の直後
    //    (読点は元々ポーズが入る位置なので、少し短くても切る。「はい、」程度は切らない)
    let comma_min = 4.0f64.max(mora_min / 2.0);
    for (i, &c) in text.iter().enumerate() {
        if COMMAS.contains(c) && comma_min <= prefix[i + 1] && prefix[i + 1] <= 2.0 * mora_max {
            let mut pos = i + 1;
            while pos < text.len() && TRAILERS.contains(text[pos]) {
                pos += 1;
            }
            if tail_ok(pos) {
                return Some(pos);
            }
        }
    }
    // 2) 自然な切れ目: [mora_min, mora_max] の中で最も後ろ、無ければ 2*mora_max まで延長して最初
    let mut best = None;
    for i in 1..text.len() {
        if mora_min <= prefix[i] && prefix[i] <= mora_max && is_natural_boundary(text, i) && tail_ok(i) {
            best = Some(i);
        }
    }
    if best.is_some() {
        return best;
    }
    (1..text.len()).find(|&i| mora_max < prefix[i] && prefix[i] <= 2.0 * mora_max && is_natural_boundary(text, i) && tail_ok(i))
}

/// `max_chars` を超える塊を 読点 > 自然な切れ目 > 強制 の優先で分割する。
fn split_long(text: &[char], min_chars: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest: Vec<char> = text.to_vec();
    while max_chars > 0 && letters(&rest) > max_chars {
        let lo = 1usize.max(min_chars.min(max_chars.saturating_sub(1)));
        let hi = (rest.len() - 1).min(max_chars);
        let mut cut = None;
        for i in (lo..=hi).rev() {
            // 読点(直後で切る)
            if COMMAS.contains(rest[i - 1]) {
                cut = Some(i);
                break;
            }
        }
        if cut.is_none() {
            cut = (lo..=hi).rev().find(|&i| is_natural_boundary(&rest, i));
        }
        let cut = cut.unwrap_or(hi);
        let head = s(&rest[..cut]).trim().to_string();
        let tail = s(&rest[cut..]).trim().to_string();
        rest = tail.chars().collect();
        if !head.is_empty() {
            out.push(head);
        }
    }
    let r = s(&rest).trim().to_string();
    if !r.is_empty() {
        out.push(r);
    }
    out
}

/// 文末記号(+直後の閉じ括弧)で区切った文のリスト(区切り文字は前の文に含める)。
fn sentences(text: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < text.len() {
        let ch = text[i];
        buf.push(ch);
        if DELIMITERS.contains(ch) {
            let mut j = i + 1;
            while j < text.len() && (TRAILERS.contains(text[j]) || DELIMITERS.contains(text[j])) {
                buf.push(text[j]);
                j += 1;
            }
            i = j - 1;
            let t = buf.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            buf.clear();
        }
        i += 1;
    }
    let t = buf.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

/// テキストを発話チャンクに分割する(詳細はモジュール docs)。
///
/// `first_mora_max <= 0` で先頭チャンクの短縮を無効化、`max_chars == 0` で長さ分割を無効化。
pub fn split_chunks(text: &str, opt: &ChunkOptions) -> Vec<String> {
    let text: Vec<char> = text.trim().chars().collect();
    if text.is_empty() {
        return Vec::new();
    }

    // 1) 文をしきい値まで束ねる
    let mut merged: Vec<String> = Vec::new();
    let mut buf = String::new();
    for sent in sentences(&text) {
        buf.push_str(&sent);
        let threshold = if merged.is_empty() { opt.first_min_chars } else { opt.min_chars };
        let bc: Vec<char> = buf.chars().collect();
        if letters(&bc) >= threshold {
            merged.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        if let Some(last) = merged.last_mut().filter(|_| !buf.chars().any(is_letter)) {
            last.push_str(&buf); // 記号だけの余りは直前へ
        } else {
            merged.push(buf);
        }
    }

    // 2) 先頭チャンクを短く切る
    let mut chunks: Vec<String> = Vec::new();
    for (idx, piece) in merged.iter().enumerate() {
        let pc: Vec<char> = piece.chars().collect();
        if idx == 0 {
            if let Some(pos) = cut_first(&pc, opt.first_mora_min, opt.first_mora_max) {
                let head = s(&pc[..pos]).trim().to_string();
                let tail = s(&pc[pos..]).trim().to_string();
                chunks.push(head);
                if !tail.is_empty() {
                    let tc: Vec<char> = tail.chars().collect();
                    chunks.extend(split_long(&tc, opt.min_chars, opt.max_chars));
                }
                continue;
            }
            if opt.max_chars > 0 && letters(&pc) > opt.max_chars {
                chunks.extend(split_long(&pc, opt.min_chars, opt.max_chars));
                continue;
            }
            chunks.push(piece.clone());
        } else {
            // 3) 長すぎるチャンクは長さで分割
            chunks.extend(split_long(&pc, opt.min_chars, opt.max_chars));
        }
    }
    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(min: usize, first_min: usize) -> ChunkOptions {
        ChunkOptions { min_chars: min, first_min_chars: first_min, ..Default::default() }
    }

    fn joined(chunks: &[String]) -> String {
        chunks.concat().replace(' ', "")
    }

    #[test]
    fn single_sentence() {
        assert_eq!(split_chunks("こんにちは。", &ChunkOptions::default()), ["こんにちは。"]);
    }

    #[test]
    fn two_sentences_with_min_chars() {
        let text = "最初の文です。二つ目の文はここから始まり、しばらく続きます。";
        let chunks = split_chunks(text, &opt(16, 1));
        assert_eq!(chunks[0], "最初の文です。");
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn first_chunk_threshold() {
        assert_eq!(split_chunks("短い。二文目も短い。", &opt(5, 3)), ["短い。", "二文目も短い。"]);
    }

    #[test]
    fn threshold_not_met_continues_accumulating() {
        assert_eq!(split_chunks("こんにちは。元気ですか。", &opt(16, 10)), ["こんにちは。元気ですか。"]);
    }

    #[test]
    fn short_remainder_is_not_merged_into_last_chunk() {
        assert_eq!(split_chunks("これは十分に長い文です。あとがき", &opt(16, 1)), ["これは十分に長い文です。", "あとがき"]);
    }

    #[test]
    fn remainder_long_enough_stands_alone() {
        assert_eq!(split_chunks("一文目。これは残りとして十分に長い文です。", &opt(16, 1)), ["一文目。", "これは残りとして十分に長い文です。"]);
    }

    #[test]
    fn exclamation_and_question() {
        assert_eq!(split_chunks("本当ですか?本当に!そうなんだ…", &opt(3, 1)), ["本当ですか?", "本当に!", "そうなんだ…"]);
    }

    #[test]
    fn newline_is_delimiter() {
        assert_eq!(split_chunks("一行目です\n二行目です", &opt(16, 1)), ["一行目です", "二行目です"]);
    }

    #[test]
    fn closed_quote_sticks_to_delimiter() {
        let text = "「こんにちは」と彼は言った。次の文です。";
        let chunks = split_chunks(text, &opt(5, 1));
        assert_eq!(chunks[0], "「こんにちは」と彼は言った。");
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn empty_and_whitespace() {
        assert!(split_chunks("", &ChunkOptions::default()).is_empty());
        assert!(split_chunks("   \n  ", &ChunkOptions::default()).is_empty());
    }

    #[test]
    fn trailers_attach_to_delimiter() {
        let chunks = split_chunks("はいそうです。)(続きます。", &opt(3, 1));
        assert_eq!(chunks[0], "はいそうです。)");
    }

    #[test]
    fn fullwidth_exclamation_and_question_are_delimiters() {
        assert_eq!(split_chunks("本当ですか？本当に！そうなんだ…", &opt(3, 1)), ["本当ですか？", "本当に！", "そうなんだ…"]);
    }

    #[test]
    fn symbol_only_remainder_sticks_to_previous() {
        assert_eq!(split_chunks("こんにちは。♪", &opt(16, 1)), ["こんにちは。♪"]);
    }

    #[test]
    fn first_chunk_cut_at_comma() {
        assert_eq!(split_chunks("こんにちは、今日はいい天気ですね。", &ChunkOptions::default()), ["こんにちは、", "今日はいい天気ですね。"]);
    }

    #[test]
    fn very_short_comma_prefix_is_not_cut() {
        let chunks = split_chunks("はい、そうです。それでは次の話題に移りましょう。", &ChunkOptions::default());
        assert_eq!(chunks[0], "はい、そうです。");
    }

    #[test]
    fn first_chunk_cut_on_unpunctuated_asr_text() {
        let text = "今日は自然言語処理の最新の研究について話したいと思います";
        let chunks = split_chunks(text, &ChunkOptions::default());
        assert_eq!(chunks[0], "今日は自然言語処理の");
        assert!((8.0..=24.0).contains(&count_mora(&chunks[0])));
        assert_eq!(joined(&chunks), text);
        let chunks = split_chunks("えーと今日はですね新しいモデルの話をしようと思っていて", &ChunkOptions::default());
        assert_eq!(chunks[0], "えーと今日はですね");
        assert!((8.0..=12.0).contains(&count_mora(&chunks[0])));
    }

    #[test]
    fn first_chunk_mora_window_is_configurable() {
        let text = "今日は自然言語処理の最新の研究について話したいと思います";
        let loose = split_chunks(text, &ChunkOptions { first_mora_min: 16.0, first_mora_max: 24.0, ..Default::default() });
        assert!((16.0..=24.0).contains(&count_mora(&loose[0])));
        assert_eq!(split_chunks(text, &ChunkOptions { first_mora_max: 0.0, ..Default::default() }), [text]);
    }

    #[test]
    fn slightly_long_first_sentence_is_not_split_into_tiny_tail() {
        let chunks = split_chunks("これは三つ目の発話です。チャンク分割を確認します。", &ChunkOptions::default());
        assert_eq!(chunks, ["これは三つ目の発話です。", "チャンク分割を確認します。"]);
    }

    #[test]
    fn long_unpunctuated_text_is_split_by_length() {
        let text = "えーと今日はですね新しいモデルの話をしようと思っていて".repeat(4);
        let chunks = split_chunks(&text, &ChunkOptions { max_chars: 30, ..Default::default() });
        assert!(chunks.len() >= 4);
        assert!(chunks.iter().all(|c| c.chars().count() <= 30));
        assert_eq!(joined(&chunks), text);
    }
}
