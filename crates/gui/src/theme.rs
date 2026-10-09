//! デザイントークン。配色はアプリアイコン(assets/app-icon.png)に揃える:
//!
//! - インク(紺): 地。グラデーションは使わず、面の明度差だけで階層を作る
//! - 桜(ピンク): **話した内容**(入力側。マイク・テキスト入力)
//! - 藤(紫): **届けた声**(出力側。合成・再生)。主操作のアクセントも兼ねる
//! - ミント: **ライブ / 準備完了**(アイコンの緑点)
//!
//! 「入力=桜 / 出力=藤」はターンカード・レベルメーター等で一貫させ、色だけで
//! どちら側の情報かが分かるようにする。

use std::rc::Rc;

use gpui_kit::component::{Theme, ThemeMode, ThemeSet};
use gpui_kit::{App, Rgba, rgb, rgba};

// ---- 面(暗 → 明)
pub const BG: u32 = 0x12111f;
pub const SURFACE: u32 = 0x17152a;
pub const CARD: u32 = 0x1d1b33;
pub const ELEVATED: u32 = 0x25223f;
pub const BORDER: u32 = 0x2b2845;
pub const BORDER_STRONG: u32 = 0x3a3660;

// ---- 文字
pub const TEXT: u32 = 0xecebf5;
pub const TEXT_MUTED: u32 = 0xa19dbd;
pub const TEXT_FAINT: u32 = 0x6f6a8f;

// ---- 意味色
pub const INPUT: u32 = 0xf6a5c6; // 桜: 話した内容
pub const VOICE: u32 = 0x9b7cf5; // 藤: 届けた声
pub const LIVE: u32 = 0x7eeec4; // ミント: ライブ / 準備完了
pub const WARN: u32 = 0xf0c27b;
pub const ERROR: u32 = 0xff7a8a;

// ---- ログ欄(gui.log と同じ行を表示。読みやすさ優先で地より明るい文字)
pub const LOG_TEXT: u32 = 0xc4bfe0;
pub const LOG_ERROR: u32 = 0xff8a8a;

pub fn c(hex: u32) -> Rgba {
    rgb(hex)
}

/// 0xRRGGBB に不透明度(0..=255)を付ける
pub fn ca(hex: u32, alpha: u8) -> Rgba {
    rgba((hex << 8) | alpha as u32)
}

/// gpui-kit コンポーネント(Button / Select / Input / Switch / タイトルバーのウィンドウ操作等)
/// の配色を上のトークンへ寄せる。指定しない項目は既定の濃色テーマを使う。
const COMPONENT_THEME: &str = r##"{
  "name": "sttts",
  "themes": [{
    "name": "sttts Ink",
    "mode": "dark",
    "colors": {
      "background": "#12111f",
      "foreground": "#ecebf5",
      "border": "#2b2845",
      "input.border": "#3a3660",
      "caret": "#ecebf5",
      "ring": "#9b7cf5",
      "selection.background": "#9b7cf566",
      "muted.background": "#1d1b33",
      "muted.foreground": "#a19dbd",
      "popover.background": "#1d1b33",
      "popover.foreground": "#ecebf5",
      "accent.background": "#2b2850",
      "accent.foreground": "#ecebf5",
      "primary.background": "#9b7cf5",
      "primary.hover.background": "#a98ef7",
      "primary.active.background": "#8a6be8",
      "primary.foreground": "#ffffff",
      "secondary.background": "#25223f",
      "secondary.hover.background": "#2e2a4d",
      "secondary.active.background": "#211e38",
      "secondary.foreground": "#ecebf5",
      "list.background": "#1d1b33",
      "list.active.background": "#9b7cf533",
      "list.active.border": "#9b7cf5",
      "list.even.background": "#17152a66",
      "list.head.background": "#17152a",
      "switch.background": "#3a3660",
      "progress_bar.background": "#9b7cf5",
      "slider.bar.background": "#9b7cf5",
      "slider.thumb.background": "#ecebf5",
      "success.background": "#7eeec4",
      "warning.background": "#f0c27b",
      "tab_bar.background": "#17152a",
      "tab_bar.segmented.background": "#17152a",
      "tab.active.background": "#25223f",
      "title_bar.background": "#12111f",
      "title_bar.border": "#2b2845",
      "status_bar.background": "#12111f",
      "status_bar.border": "#2b2845",
      "scrollbar.thumb.background": "#3a3660cc",
      "scrollbar.thumb.hover.background": "#4a4578",
      "overlay": "#0a091499",
      "window.border": "#2b2845"
    }
  }]
}"##;

/// 濃色モードに上の配色を登録して適用する。gpui_kit::init の後に呼ぶ。
pub fn install(cx: &mut App) {
    let set: ThemeSet = serde_json::from_str(COMPONENT_THEME).expect("組み込みテーマの JSON が不正");
    if let Some(config) = set.themes.into_iter().next() {
        Theme::update(cx, |theme| theme.dark_theme = Rc::new(config));
    }
    Theme::change(ThemeMode::Dark, None, cx);
}
