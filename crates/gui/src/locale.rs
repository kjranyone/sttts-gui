//! 表示言語の決定と適用。保存値(data/config.json の `language`)があればそれ、
//! 無ければ OS の表示言語(en / ja / zh 以外は英語)。判定は `sttts_i18n::initial`(CLI と共通)。

pub use sttts_i18n::Lang;

/// 起動時の言語: 保存値 → OS の表示言語 → 英語
pub fn initial(saved: Option<&str>) -> Lang {
    sttts_i18n::initial(saved)
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
