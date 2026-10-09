//! 右レール: ライブ / 音声キュー / 声 / 認識。主画面に常に出す設定はここだけに置く。

use gpui_kit::component::button::*;
use gpui_kit::component::input::Input;
use gpui_kit::component::select::Select;
use gpui_kit::component::switch::Switch;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::help::HelpTopic;
use super::{GEMINI_KEY_URL, MicTransition, StttsApp, kit, phase_label};
use crate::theme::{self, c};

const RAIL_W: f32 = 340.;

impl StttsApp {
    pub(super) fn render_rail(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("rail")
            .w(px(RAIL_W))
            .h_full()
            .flex_shrink_0()
            .border_l_1()
            .border_color(c(theme::BORDER))
            .bg(c(theme::SURFACE))
            .overflow_y_scroll()
            .child(self.render_live(cx))
            .child(self.render_delivery(cx))
            .child(self.render_voice(cx))
            .child(self.render_recognition(cx))
    }

    fn render_live(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let busy = self.mic_transition != MicTransition::None;
        let level = ((self.mic_level_db + 60.0) / 60.0).clamp(0.0, 1.0);
        let input_name = self
            .selected_input_name
            .clone()
            .unwrap_or_else(|| super::DEFAULT_INPUT_LABEL.to_string());

        let button = Button::new("live")
            .w_full()
            .large()
            .icon(if self.mic_running { IconName::Square } else { IconName::Mic })
            .label(match self.mic_transition {
                MicTransition::Starting => "ライブを準備中…",
                MicTransition::Stopping => "ライブを停止中…",
                MicTransition::None if self.mic_running => "ライブを停止",
                MicTransition::None => "ライブ開始",
            })
            .when(!self.mic_running, |b| b.primary())
            .when(self.mic_running, |b| b.danger())
            .loading(busy)
            .disabled(busy || !self.connected)
            .on_click(cx.listener(|this, _, _, cx| this.toggle_mic(cx)));

        let meter = h_flex()
            .gap_2()
            .items_center()
            .child(
                div()
                    .flex_1()
                    .h(px(6.))
                    .rounded_full()
                    .bg(c(theme::ELEVATED))
                    .overflow_hidden()
                    .child(
                        div()
                            .h_full()
                            .w(relative(if self.mic_running { level } else { 0.0 }))
                            .rounded_full()
                            .bg(linear_gradient(
                                90.,
                                linear_color_stop(c(theme::INPUT), 0.),
                                linear_color_stop(c(theme::VOICE), 1.),
                            )),
                    ),
            )
            .child(
                div()
                    .w(px(52.))
                    .text_xs()
                    .text_right()
                    .text_color(c(theme::TEXT_FAINT))
                    .child(if self.mic_running {
                        format!("{:+.0} dB", self.mic_level_db.max(-99.0))
                    } else {
                        "—".to_string()
                    }),
            );

        kit::section(
            "ライブ",
            v_flex()
                .gap_3()
                .child(button)
                .child(meter)
                .child(
                    h_flex()
                        .gap_1()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .min_w_0()
                                .overflow_hidden()
                                .text_ellipsis()
                                .whitespace_nowrap()
                                .text_xs()
                                .text_color(c(theme::TEXT_FAINT))
                                .child(format!("入力: {input_name}")),
                        )
                        .child(
                            Button::new("change-input")
                                .xsmall()
                                .ghost()
                                .label("変更")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.settings_open = true;
                                    cx.notify();
                                })),
                        ),
                ),
        )
    }

    /// 確定文を音声キューへ自動で流すか(OFF = カードで止めて手動で発話)と、テンポと間の再現。
    fn render_delivery(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let weak_auto = cx.weak_entity();
        let weak_expr = cx.weak_entity();
        kit::section(
            "音声キュー",
            v_flex()
                .gap_3()
                .child(kit::with_help(
                    Switch::new("auto-play")
                        .checked(self.auto_speak)
                        .label("自動再生")
                        .on_click(move |checked, _, cx| {
                            let _ = weak_auto.update(cx, |this, cx| this.set_auto_speak(*checked, cx));
                        }),
                    self.help_icon(HelpTopic::AutoPlay, cx),
                ))
                .child(kit::with_help(
                    Switch::new("performance")
                        .checked(self.performance_enabled)
                        .label("テンポと間を再現")
                        .on_click(move |checked, _, cx| {
                            let _ = weak_expr.update(cx, |this, cx| this.set_performance_enabled(*checked, cx));
                        }),
                    self.help_icon(HelpTopic::Tempo, cx),
                )),
        )
    }

    fn render_voice(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let icon = self
            .selected_voice_name
            .as_deref()
            .and_then(|n| self.voice_image(n));
        let has_voice = self.selected_voice_name.is_some();
        kit::section_with_help(
            "声",
            self.help_icon(HelpTopic::Voice, cx),
            v_flex()
                .gap_3()
                .child(
                    h_flex()
                        .gap_1()
                        .items_center()
                        .when_some(icon, |d, path| {
                            d.child(
                                img(path)
                                    .size(px(32.))
                                    .flex_shrink_0()
                                    .rounded_md()
                                    .object_fit(ObjectFit::Cover),
                            )
                        })
                        .child(div().flex_1().min_w_0().child(Select::new(&self.voice_select).small()))
                        .child(
                            Button::new("add-voice")
                                .small()
                                .ghost()
                                .icon(IconName::Plus)
                                .tooltip("声を追加(wav / flac / 画像。ドロップも可)")
                                .on_click(cx.listener(|this, _, _, cx| this.pick_voice_files(cx))),
                        )
                        .when(has_voice, |d| {
                            d.child(
                                Button::new("delete-voice")
                                    .small()
                                    .ghost()
                                    .icon(IconName::Delete)
                                    .tooltip("この声を削除")
                                    .on_click(cx.listener(|this, _, window, cx| this.delete_selected_voice(window, cx))),
                            )
                        }),
                )
                .child(kit::field_with_help(
                    "話し方",
                    self.help_icon(HelpTopic::Style, cx),
                    Input::new(&self.caption_input).small(),
                )),
        )
    }

    fn render_recognition(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let gemini = self.selected_asr_engine == "gemini";
        let has_key = !self.gemini_api_key.is_empty();
        let phase = self.asr_state.phase.as_str();

        let detail = if gemini {
            v_flex()
                .gap_2()
                .child(kit::field(
                    "API キー",
                    h_flex()
                        .gap_1()
                        .child(div().flex_1().min_w_0().child(Input::new(&self.gemini_key_input).mask_toggle().small()))
                        .child(
                            Button::new("get-gemini-key")
                                .small()
                                .ghost()
                                .icon(IconName::ExternalLink)
                                .label("取得")
                                .tooltip("Google AI Studio")
                                .on_click(cx.listener(|_, _, _, cx| cx.open_url(GEMINI_KEY_URL))),
                        ),
                ))
                .child(
                    div()
                        .text_xs()
                        .text_color(c(if has_key { theme::LIVE } else { theme::WARN }))
                        .child(if has_key { "保存済み" } else { "未設定" }),
                )
        } else {
            v_flex()
        };

        kit::section_with_help(
            "認識",
            self.help_icon(HelpTopic::Recognition, cx),
            v_flex()
                .gap_3()
                .child(Select::new(&self.asr_select).small())
                .child(detail)
                .child(kit::chip(
                    kit::phase_color(phase),
                    format!("{} · {}", self.asr_provider_label(), phase_label(&self.asr_state)),
                )),
        )
    }
}
