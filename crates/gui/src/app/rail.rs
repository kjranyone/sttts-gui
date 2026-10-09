//! 右レール: ライブ / 届け方 / 声 / 認識。主画面に常に出す設定はここだけに置く。

use gpui_kit::component::button::*;
use gpui_kit::component::input::Input;
use gpui_kit::component::select::Select;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{GEMINI_KEY_URL, MicTransition, StttsApp, kit, phase_label};
use crate::theme::{self, c};

const RAIL_W: f32 = 340.;

fn segment(id: &'static str, label: &'static str, selected: bool) -> Button {
    Button::new(id)
        .small()
        .flex_1()
        .label(label)
        .when(selected, |b| b.primary())
        .when(!selected, |b| b.ghost())
}

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

    fn render_delivery(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let auto = self.auto_speak;
        kit::section(
            "届け方",
            v_flex()
                .gap_2()
                .child(
                    // 2択の切替。選ばれている側を塗りつぶして、今の届け方を一目で分かるようにする
                    h_flex()
                        .gap_0p5()
                        .p_0p5()
                        .rounded_md()
                        .bg(c(theme::ELEVATED))
                        .child(segment("mode-auto", "すぐ話す", auto).on_click(
                            cx.listener(|this, _, _, cx| this.set_auto_speak(true, cx)),
                        ))
                        .child(segment("mode-confirm", "確認してから", !auto).on_click(
                            cx.listener(|this, _, _, cx| this.set_auto_speak(false, cx)),
                        )),
                )
                .child(kit::hint(if auto {
                    "認識が確定したら、すぐその声で話します。誤認識もそのまま届くので、気づいたらカードの「止める」「訂正」で直します。スピーカーで鳴らすとマイクが拾って繰り返すことがあるため、ヘッドホン推奨です。"
                } else {
                    "確定した文はカードで止まります。「この内容で話す」か「訂正する」を選んでから届けます。"
                })),
        )
    }

    fn render_voice(&self, cx: &mut Context<Self>) -> impl IntoElement {
        kit::section(
            "声",
            v_flex()
                .gap_3()
                .child(kit::field(
                    "どの声で届けるか",
                    h_flex()
                        .gap_1()
                        .child(div().flex_1().min_w_0().child(Select::new(&self.voice_select).small()))
                        .child(
                            Button::new("open-voices")
                                .small()
                                .ghost()
                                .icon(IconName::FolderOpen)
                                .tooltip("声フォルダを開く(wav を置くと声として選べます)")
                                .on_click(cx.listener(|this, _, window, cx| this.open_voice_folder(window, cx))),
                        ),
                ))
                .child(kit::field("話し方の指示(任意)", Input::new(&self.caption_input).small()))
                .child(kit::hint(
                    "声の質や性別ではなく、場面や伝え方を書くと安定します(声そのものは上の選択で決まります)。",
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
                    "Gemini API キー",
                    h_flex()
                        .gap_1()
                        .child(div().flex_1().min_w_0().child(Input::new(&self.gemini_key_input).mask_toggle().small()))
                        .child(
                            Button::new("get-gemini-key")
                                .small()
                                .ghost()
                                .icon(IconName::ExternalLink)
                                .label("取得")
                                .tooltip("Google AI Studio でキーを発行する")
                                .on_click(cx.listener(|_, _, _, cx| cx.open_url(GEMINI_KEY_URL))),
                        ),
                ))
                .child(
                    div()
                        .text_xs()
                        .text_color(c(if has_key { theme::LIVE } else { theme::WARN }))
                        .child(if has_key {
                            "✓ 保存済み(この PC のユーザーだけが読める形で暗号化)"
                        } else {
                            "未設定: 「取得」でキーを発行し、貼り付けて Enter"
                        }),
                )
                .child(kit::hint("話した音声は発話ごとに Google へ送られます。"))
        } else {
            v_flex().child(kit::hint("この PC の中で認識します(音声は外部に送りません)。"))
        };

        kit::section(
            "認識",
            v_flex()
                .gap_3()
                .child(kit::field("話した内容を文字にする方法", Select::new(&self.asr_select).small()))
                .child(detail)
                .child(kit::chip(
                    kit::phase_color(phase),
                    format!("{} · {}", self.asr_provider_label(), phase_label(&self.asr_state)),
                )),
        )
    }
}
