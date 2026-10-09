//! 画面共通の小さな部品(見出し・項目行・状態チップ・補足文)。

use gpui_kit::component::*;
use gpui_kit::*;

use crate::theme::{self, c, ca};

/// レールの区画: 小見出し + 内容。区画どうしは罫線で区切る。
pub(super) fn section(title: &'static str, content: impl IntoElement) -> Div {
    section_frame(section_title(title), content)
}

/// 見出しの横に「?」(解説)を置く区画
pub(super) fn section_with_help(title: &'static str, help: impl IntoElement, content: impl IntoElement) -> Div {
    section_frame(h_flex().gap_1p5().items_center().child(section_title(title)).child(help), content)
}

fn section_title(title: &'static str) -> Div {
    div()
        .text_xs()
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(c(theme::TEXT_FAINT))
        .child(title)
}

fn section_frame(header: impl IntoElement, content: impl IntoElement) -> Div {
    v_flex()
        .gap_2()
        .px_4()
        .py_3()
        .border_b_1()
        .border_color(c(theme::BORDER))
        .child(header)
        .child(content)
}

/// ラベル付きの設定項目(ラベルは上、操作は下)
pub(super) fn field(label: &'static str, control: impl IntoElement) -> Div {
    v_flex()
        .gap_1()
        .child(div().text_xs().text_color(c(theme::TEXT_MUTED)).child(label))
        .child(control)
}

/// ラベルの横に「?」(解説)を置く設定項目
pub(super) fn field_with_help(label: &'static str, help: impl IntoElement, control: impl IntoElement) -> Div {
    v_flex()
        .gap_1()
        .child(
            h_flex()
                .gap_1p5()
                .items_center()
                .child(div().text_xs().text_color(c(theme::TEXT_MUTED)).child(label))
                .child(help),
        )
        .child(control)
}

/// 操作(スイッチ等)の横に「?」を置く行
pub(super) fn with_help(control: impl IntoElement, help: impl IntoElement) -> Div {
    h_flex().gap_2().items_center().child(control).child(help)
}

/// 色付きドット + 短いラベル
pub(super) fn chip(dot: u32, label: impl Into<SharedString>) -> Div {
    h_flex()
        .gap_1p5()
        .items_center()
        .px_2()
        .h(px(20.))
        .rounded_full()
        .bg(ca(theme::TEXT, 0x0d))
        .child(div().size(px(6.)).rounded_full().bg(c(dot)))
        .child(div().text_xs().text_color(c(theme::TEXT_MUTED)).child(label.into()))
}

/// 補足説明(小さく、薄く)
pub(super) fn hint(text: impl Into<SharedString>) -> Div {
    div()
        .text_xs()
        .line_height(rems(1.25))
        .text_color(c(theme::TEXT_FAINT))
        .child(text.into())
}

/// 状態の色(エンジン phase)
pub(super) fn phase_color(phase: &str) -> u32 {
    match phase {
        "ready" => theme::LIVE,
        "loading" => theme::WARN,
        "error" => theme::ERROR,
        _ => theme::TEXT_FAINT,
    }
}
