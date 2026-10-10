//! 表示言語(英語 / 日本語 / 簡体字中国語)の選択と、文言のインライン翻訳。
//!
//! 訳は呼び出し箇所に 3 言語を並べて書く(`tr!("en", "ja", "zh")`)。キーと辞書ファイルを
//! 分けないので、文言の意味が手元で読め、訳漏れはコンパイルエラーになる。
//!
//! 言語はプロセス全体で 1 つ(GUI とエンジンが同じプロセスで動くため、エンジンが送る
//! メッセージも同じ言語になる)。切替はすぐに次の描画・次のメッセージから効く。

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    En,
    Ja,
    Zh,
}

impl Lang {
    pub const ALL: [Lang; 3] = [Lang::En, Lang::Ja, Lang::Zh];

    /// 保存用のコード(data/config.json の `language`)
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ja => "ja",
            Lang::Zh => "zh",
        }
    }

    /// 言語の自称(選択欄では、今の表示言語に関係なく読めるようにする)
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Ja => "日本語",
            Lang::Zh => "简体中文",
        }
    }

    /// `ja` / `ja-JP` / `zh-CN` / `zh_Hans` / `en-US` 等。知らない言語は None。
    pub fn from_tag(tag: &str) -> Option<Lang> {
        let primary = tag.trim().split(['-', '_', '.']).next()?.to_ascii_lowercase();
        match primary.as_str() {
            "en" => Some(Lang::En),
            "ja" => Some(Lang::Ja),
            "zh" | "cn" => Some(Lang::Zh),
            _ => None,
        }
    }

    pub fn from_native_name(name: &str) -> Option<Lang> {
        Lang::ALL.into_iter().find(|l| l.native_name() == name)
    }
}

// 既定は日本語(元の文言。テストの期待値もこれ)。アプリは起動時に OS の言語か保存値で上書きする。
static LANG: AtomicU8 = AtomicU8::new(1);

pub fn lang() -> Lang {
    match LANG.load(Ordering::Relaxed) {
        0 => Lang::En,
        2 => Lang::Zh,
        _ => Lang::Ja,
    }
}

pub fn set_lang(lang: Lang) {
    let v = match lang {
        Lang::En => 0,
        Lang::Ja => 1,
        Lang::Zh => 2,
    };
    LANG.store(v, Ordering::Relaxed);
}

/// 起動時の言語: 保存値(data/config.json の `language`)→ OS の表示言語 → 英語。
/// GUI と CLI(`sttts-say`)で同じ規則にする。
pub fn initial(saved: Option<&str>) -> Lang {
    saved.and_then(Lang::from_tag).or_else(os_language).unwrap_or(Lang::En)
}

#[cfg(windows)]
fn os_language() -> Option<Lang> {
    // 主言語 ID(LANGID の下位 10 bit): 0x09 英語 / 0x11 日本語 / 0x04 中国語
    let id = unsafe { windows::Win32::Globalization::GetUserDefaultUILanguage() } & 0x3ff;
    match id {
        0x09 => Some(Lang::En),
        0x11 => Some(Lang::Ja),
        0x04 => Some(Lang::Zh),
        _ => None,
    }
}

#[cfg(not(windows))]
fn os_language() -> Option<Lang> {
    ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
        .and_then(|v| Lang::from_tag(&v))
}

/// 今の言語の文言を選ぶ(書式なし)。値は 3 つとも同じ型の式なら何でもよい。
///
/// ```
/// let s: &str = sttts_i18n::tr!("Ready", "準備完了", "就绪");
/// ```
#[macro_export]
macro_rules! tr {
    ($en:expr, $ja:expr, $zh:expr $(,)?) => {
        match $crate::lang() {
            $crate::Lang::En => $en,
            $crate::Lang::Ja => $ja,
            $crate::Lang::Zh => $zh,
        }
    };
}

/// 今の言語で `format!` する。引数は書式文字列内の変数名で埋め込む(`{name}`)。
/// 言語ごとに語順が違っても、使う変数が揃っていればよい。
///
/// ```
/// let n = 3;
/// let s = sttts_i18n::trf!("{n} turns", "{n} ターン", "{n} 轮");
/// ```
#[macro_export]
macro_rules! trf {
    ($en:literal, $ja:literal, $zh:literal $(,)?) => {
        match $crate::lang() {
            $crate::Lang::En => format!($en),
            $crate::Lang::Ja => format!($ja),
            $crate::Lang::Zh => format!($zh),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_map_to_languages() {
        assert_eq!(Lang::from_tag("ja-JP"), Some(Lang::Ja));
        assert_eq!(Lang::from_tag("zh_Hans"), Some(Lang::Zh));
        assert_eq!(Lang::from_tag("en"), Some(Lang::En));
        assert_eq!(Lang::from_tag("fr-FR"), None);
        for l in Lang::ALL {
            assert_eq!(Lang::from_tag(l.code()), Some(l));
            assert_eq!(Lang::from_native_name(l.native_name()), Some(l));
        }
    }

    #[test]
    fn macros_follow_current_language() {
        let n = 2;
        for l in Lang::ALL {
            set_lang(l);
            assert_eq!(lang(), l);
            let s = tr!("a", "b", "c");
            let f = trf!("{n} en", "{n} ja", "zh {n}");
            match l {
                Lang::En => assert_eq!((s, f.as_str()), ("a", "2 en")),
                Lang::Ja => assert_eq!((s, f.as_str()), ("b", "2 ja")),
                Lang::Zh => assert_eq!((s, f.as_str()), ("c", "zh 2")),
            }
        }
        set_lang(Lang::Ja);
    }
}
