//! 表示言語の決定と適用。保存値(data/config.json の `language`)があればそれ、
//! 無ければ OS の表示言語(en / ja / zh 以外は英語)。

pub use sttts_i18n::Lang;

/// 起動時の言語: 保存値 → OS の表示言語 → 英語
pub fn initial(saved: Option<&str>) -> Lang {
    saved.and_then(Lang::from_tag).or_else(os_language).unwrap_or(Lang::En)
}

/// アプリの文言と、gpui-component の組み込み文言(右クリックメニュー等)を切り替える。
/// gpui-component に日本語の訳は無いので、日本語のときは英語になる。
pub fn apply(lang: Lang) {
    sttts_i18n::set_lang(lang);
    gpui_kit::component::set_locale(match lang {
        Lang::Zh => "zh-CN",
        Lang::En | Lang::Ja => "en",
    });
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
