//! テキスト前処理。`normalize_text` は `irodori_tts/text_normalization.py` の移植。

use std::sync::LazyLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

/// `SIMPLE_REPLACE_MAP`(順序は原典の dict 順)
const SIMPLE_REPLACE: &[(&str, &str)] = &[
    ("\t", ""),
    ("[n]", ""),
    (r"\[n\]", ""),
    ("\u{3000}", ""),
    ("？", "?"),
    ("！", "!"),
    ("♥", "♡"),
    ("●", "○"),
    ("◯", "○"),
    ("〇", "○"),
];

/// `REGEX_REPLACE_MAP`(順序は原典の dict 順)
static REGEX_REPLACE: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    let table: [(&str, &str); 4] = [
        ("[;▼♀♂《》≪≫①②③④⑤⑥]", ""),
        (
            r"[\x{02d7}\x{2010}-\x{2015}\x{2043}\x{2212}\x{23af}\x{23e4}\x{2500}\x{2501}\x{2e3a}\x{2e3b}]",
            "",
        ),
        (r"[\x{ff5e}\x{301c}]", "ー"),
        ("…{3,}", "……"),
    ];
    table.iter().map(|(p, r)| (Regex::new(p).expect("static regex"), *r)).collect()
});

fn closing_bracket(c: char) -> Option<char> {
    match c {
        '「' => Some('」'),
        '『' => Some('』'),
        '（' => Some('）'),
        '【' => Some('】'),
        '(' => Some(')'),
        _ => None,
    }
}

/// 全体を囲む対応した括弧を外側から繰り返し外す。
fn strip_outer_brackets(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    loop {
        if chars.len() < 2 {
            break;
        }
        let start = chars[0];
        let end = chars[chars.len() - 1];
        if closing_bracket(start) == Some(end) {
            let mut depth: i64 = 0;
            let mut encloses_all = true;
            for (i, &c) in chars.iter().enumerate() {
                if c == start {
                    depth += 1;
                } else if c == end {
                    depth -= 1;
                }
                if depth == 0 && i < chars.len() - 1 {
                    encloses_all = false;
                    break;
                }
            }
            if encloses_all && depth == 0 {
                chars = chars[1..chars.len() - 1].to_vec();
                continue;
            }
        }
        break;
    }
    chars.into_iter().collect()
}

/// 発話テキストの正規化(原典 `normalize_text`)。呼び出し側が続けて `.trim()` する(runtime と同じ)。
pub fn normalize_text(s: &str) -> String {
    let mut text = s.to_string();
    for (old, new) in SIMPLE_REPLACE {
        text = text.replace(old, new);
    }
    for (re, rep) in REGEX_REPLACE.iter() {
        text = re.replace_all(&text, *rep).into_owned();
    }
    let text = strip_outer_brackets(&text);
    let text: String = text.nfkc().collect();
    text.replace("...", "…").replace("..", "…")
}
