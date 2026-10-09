//! タイトルバー(OS のタイトルバーの代わり)。左: ブランドと全体の状態、右: 応答速度と画面の開閉。
//! 空き領域はドラッグ/ダブルクリック最大化/スナップが効く。操作群だけ occlude して
//! ドラッグ領域の hit-test から外す(外さないとボタン押下が窓移動になる)。

use std::sync::{Arc, OnceLock};

use gpui_kit::component::button::*;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{MicTransition, StttsApp, kit};
use crate::theme::{self, c, ca};

fn app_icon() -> Arc<Image> {
    static ICON: OnceLock<Arc<Image>> = OnceLock::new();
    ICON.get_or_init(|| {
        Arc::new(Image::from_bytes(
            ImageFormat::Png,
            include_bytes!("../../../../assets/app-icon.png").to_vec(),
        ))
    })
    .clone()
}

impl StttsApp {
    /// 全体の状態を1つにまとめる(個別の TTS / 認識の状態はレールと詳細設定で見せる)
    fn health(&self) -> (u32, &'static str) {
        if self.backend.is_none() {
            return (theme::ERROR, "バックエンド起動エラー");
        }
        if !self.connected {
            return (theme::TEXT_FAINT, "接続中…");
        }
        let (tts, asr) = (self.tts_state.phase.as_str(), self.asr_state.phase.as_str());
        if tts == "error" {
            (theme::ERROR, "音声合成エラー")
        } else if asr == "error" {
            (theme::ERROR, "認識エラー")
        } else if tts == "loading" || asr == "loading" {
            (theme::WARN, "読み込み中…")
        } else if tts == "ready" {
            (theme::LIVE, if self.mock { "準備完了(モック)" } else { "準備完了" })
        } else {
            (theme::TEXT_FAINT, "待機中")
        }
    }

    pub(super) fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let (dot, health) = self.health();
        let live = self.mic_running && self.mic_transition == MicTransition::None;

        TitleBar::new()
            .bg(c(theme::BG))
            .border_color(c(theme::BORDER))
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(img(app_icon()).size(px(18.)).rounded_sm())
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(c(theme::TEXT))
                            .child("sttts"),
                    )
                    .child(div().w(px(1.)).h(px(14.)).mx_1().bg(c(theme::BORDER_STRONG)))
                    .child(kit::chip(dot, health))
                    .when(live, |row| {
                        row.child(
                            h_flex()
                                .gap_1p5()
                                .items_center()
                                .px_2()
                                .h(px(20.))
                                .rounded_full()
                                .bg(ca(theme::LIVE, 0x1f))
                                .child(div().size(px(6.)).rounded_full().bg(c(theme::LIVE)))
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(c(theme::LIVE))
                                        .child("ライブ中"),
                                ),
                        )
                    }),
            )
            .child(
                h_flex()
                    .gap_3()
                    .items_center()
                    .pr_2()
                    .child(match self.latency_summary() {
                        Some((last, median)) => h_flex()
                            .gap_1p5()
                            .items_baseline()
                            .child(div().text_xs().text_color(c(theme::TEXT_FAINT)).child("応答"))
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(c(theme::TEXT))
                                    .child(format!("{last}ms")),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(c(theme::TEXT_FAINT))
                                    .child(format!("中央値 {median}ms")),
                            ),
                        None => h_flex().child(
                            div()
                                .text_xs()
                                .text_color(c(theme::TEXT_FAINT))
                                .child("応答 —(話し終え → 発音)"),
                        ),
                    })
                    .child(
                        h_flex()
                            .id("title-actions")
                            .occlude()
                            .gap_0p5()
                            .items_center()
                            .child(
                                Button::new("open-settings")
                                    .small()
                                    .ghost()
                                    .icon(IconName::Settings)
                                    .selected(self.settings_open)
                                    .tooltip("詳細設定(デバイス・音声合成モデル)")
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.settings_open = !this.settings_open;
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("toggle-log")
                                    .small()
                                    .ghost()
                                    .icon(IconName::PanelBottom)
                                    .selected(self.log_open)
                                    .tooltip("ログ")
                                    .on_click(cx.listener(|this, _, _, cx| this.toggle_log(cx))),
                            ),
                    ),
            )
    }
}
