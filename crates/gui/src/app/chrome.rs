//! 画面の骨格: 全体レイアウト、ステータスバー、ログ欄、ライブ開始/停止中のオーバーレイ。

use gpui_kit::component::button::*;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{MicTransition, StttsApp, is_error_line, kit, phase_label};
use crate::theme::{self, c, ca};

impl Render for StttsApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // デバイス一覧は window を要求する API なので render で Select へ反映する
        if let Some(items) = self.pending_input_items.take() {
            let selected = self
                .selected_input_name
                .clone()
                .unwrap_or_else(|| super::DEFAULT_INPUT_LABEL.to_string());
            self.input_select.update(cx, |s, cx| {
                s.set_items(items, window, cx);
                s.set_selected_value(&selected, window, cx);
            });
        }

        if let Some(paths) = self.pending_voice_import.take() {
            self.import_voice_files(&paths, window, cx);
        }

        div()
            .id("root")
            .relative()
            .size_full()
            .bg(c(theme::BG))
            .text_color(c(theme::TEXT))
            .drag_over::<ExternalPaths>(|d, _, _, _| d.bg(ca(theme::VOICE, 0x14)))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, window, cx| {
                this.import_voice_files(paths.paths(), window, cx);
            }))
            .child(
                v_flex()
                    .size_full()
                    .child(self.render_title_bar(cx))
                    .child(
                        div()
                            .relative()
                            .flex_1()
                            .min_h_0()
                            .child(
                                h_flex()
                                    .size_full()
                                    .child(self.render_stream(cx))
                                    .child(self.render_rail(cx)),
                            )
                            .when(self.settings_open, |d| d.child(self.render_settings_sheet(cx))),
                    )
                    .when(self.log_open, |d| d.child(self.render_log(cx)))
                    .child(self.render_status_bar(cx)),
            )
            .children(self.render_help_modal(cx))
            .children(self.render_mic_overlay())
    }
}

impl StttsApp {
    fn render_status_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let fmt = |v: Option<u64>| v.map(|x| format!("{x}ms")).unwrap_or_else(|| "—".into());
        let rtf = self.last_rtf.map(|r| format!("{r:.2}")).unwrap_or_else(|| "—".into());
        let conn_dot = if self.connected { theme::LIVE } else { theme::TEXT_FAINT };
        h_flex()
            .h(px(26.))
            .px_3()
            .gap_4()
            .items_center()
            .border_t_1()
            .border_color(c(theme::BORDER))
            .bg(c(theme::BG))
            .text_xs()
            .text_color(c(theme::TEXT_FAINT))
            .child(
                h_flex()
                    .gap_1p5()
                    .items_center()
                    .child(div().size(px(6.)).rounded_full().bg(c(conn_dot)))
                    .child(self.status_hint.clone()),
            )
            .child(format!(
                "認識 {} · 初音まで {} · 合成速度 RTF {}",
                fmt(self.last_asr_ms),
                fmt(self.last_first_chunk_ms),
                rtf
            ))
            .child(div().flex_1())
            .child(format!("{} ターン", self.turns.len()))
            .child(
                Button::new("status-log")
                    .xsmall()
                    .ghost()
                    .label(if self.unread_errors > 0 {
                        format!("ログ ⚠ {}", self.unread_errors)
                    } else {
                        "ログ".into()
                    })
                    .when(self.unread_errors > 0, |b| b.text_color(c(theme::ERROR)))
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_log(cx))),
            )
    }

    /// ログ欄。新しい行で最下部へ追従し、保持している300行すべてを遡れる。
    /// 選択・コピーはできないので、同じ内容を data/gui.log にも書いている。
    fn render_log(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .border_t_1()
            .border_color(c(theme::BORDER))
            .bg(ca(0x000000, 0x40))
            .child(
                h_flex()
                    .px_3()
                    .py_1()
                    .justify_between()
                    .items_center()
                    .text_xs()
                    .text_color(c(theme::TEXT_FAINT))
                    .child(div().text_color(c(theme::TEXT_MUTED)).child("ログ"))
                    .child(
                        Button::new("close-log")
                            .xsmall()
                            .ghost()
                            .label("閉じる")
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_log(cx))),
                    ),
            )
            .child(
                div()
                    .id("log-body")
                    .h(px(150.))
                    .px_3()
                    .pb_2()
                    .overflow_y_scroll()
                    .track_scroll(&self.log_scroll)
                    .text_xs()
                    .text_color(c(theme::LOG_TEXT))
                    .children(self.logs.iter().map(|line| {
                        div()
                            .w_full()
                            .when(is_error_line(line), |d| d.text_color(c(theme::LOG_ERROR)))
                            .child(line.clone())
                    })),
            )
    }

    /// ライブ開始/停止の応答待ち中は全面を覆い、再操作を受け付けない
    /// (デバイスの短時間反復 open/close は BugCheck 0xD1 の実績あり)。
    fn render_mic_overlay(&self) -> Option<impl IntoElement> {
        if self.mic_transition == MicTransition::None {
            return None;
        }
        let (title, body) = match self.mic_transition {
            MicTransition::Stopping => (
                "ライブを停止しています…",
                div(),
            ),
            _ => (
                "ライブを準備しています…",
                div().child(
                    h_flex()
                        .gap_2()
                        .child(kit::chip(
                            kit::phase_color(&self.tts_state.phase),
                            format!("音声合成 · {}", phase_label(&self.tts_state)),
                        ))
                        .child(kit::chip(
                            kit::phase_color(&self.asr_state.phase),
                            format!("認識 · {}", phase_label(&self.asr_state)),
                        )),
                ),
            ),
        };
        Some(
            div()
                .id("mic-transition-overlay")
                .occlude()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .bg(ca(0x07060f, 0xb3))
                .flex()
                .items_center()
                .justify_center()
                .with_animation(
                    "mic-transition-fade",
                    Animation::new(std::time::Duration::from_millis(160)).with_easing(ease_in_out),
                    |el, delta| el.opacity(0.4 + 0.6 * delta),
                )
                .child(
                    v_flex()
                        .gap_3()
                        .px_8()
                        .py_6()
                        .rounded_lg()
                        .bg(c(theme::CARD))
                        .border_1()
                        .border_color(c(theme::BORDER_STRONG))
                        .shadow_lg()
                        .items_center()
                        .child(
                            div()
                                .text_lg()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(c(theme::TEXT))
                                .child(title),
                        )
                        .child(body),
                ),
        )
    }
}
