//! ストリーム: ターンカード(話した内容 → 届けた声)と、文字で話す入力欄。
//!
//! カードの上段は入力側(桜)、下段は出力側(藤)。二者の会話としては並べず、
//! 1つの入力とその届け方を1枚に対応付ける。

use gpui_kit::component::button::*;
use gpui_kit::component::input::Textarea;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{StttsApp, voice_phrase};
use crate::theme::{self, c, ca};
use crate::turns::{Turn, TurnSource, TurnStatus};

const CARD_MAX_W: f32 = 780.;

fn status_label(turn: &Turn) -> (String, u32) {
    match turn.status {
        TurnStatus::Listening => ("聞き取り中…".into(), theme::INPUT),
        TurnStatus::AwaitingConfirm => ("確認待ち".into(), theme::WARN),
        TurnStatus::Queued => ("発話待ち".into(), theme::TEXT_MUTED),
        TurnStatus::Speaking => {
            let (ready, total) = (turn.ready_chunks(), turn.chunks.len());
            if total > 0 && ready == total {
                ("再生中".into(), theme::VOICE)
            } else {
                (format!("合成中 {ready}/{}", total.max(1)), theme::VOICE)
            }
        }
        TurnStatus::Done => ("届けました".into(), theme::LIVE),
        TurnStatus::Cancelled => ("中止".into(), theme::TEXT_FAINT),
        TurnStatus::Failed => ("失敗".into(), theme::ERROR),
        TurnStatus::Unheard => ("聞き取れませんでした".into(), theme::TEXT_FAINT),
        TurnStatus::Skipped => ("話していません".into(), theme::TEXT_FAINT),
    }
}

impl StttsApp {
    pub(super) fn render_stream(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let body: Vec<AnyElement> = if self.turns.is_empty() {
            vec![self.render_empty_state().into_any_element()]
        } else {
            self.turns.iter().map(|t| self.render_turn(t, cx).into_any_element()).collect()
        };
        v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            .child(
                v_flex()
                    .id("stream")
                    .track_scroll(&self.stream_scroll)
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .px_6()
                    .py_5()
                    .gap_3()
                    .items_center()
                    .children(body),
            )
            .child(self.render_composer(cx))
    }

    fn render_empty_state(&self) -> impl IntoElement {
        let step = |n: &'static str, text: &'static str| {
            h_flex()
                .gap_3()
                .items_start()
                .child(
                    div()
                        .size(px(20.))
                        .flex_shrink_0()
                        .rounded_full()
                        .bg(c(theme::ELEVATED))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_xs()
                        .text_color(c(theme::TEXT_MUTED))
                        .child(n),
                )
                .child(div().text_sm().text_color(c(theme::TEXT_MUTED)).child(text))
        };
        v_flex()
            .flex_1()
            .w_full()
            .max_w(px(460.))
            .justify_center()
            .gap_5()
            .py_10()
            .child(
                v_flex()
                    .gap_2()
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(c(theme::TEXT))
                            .child("話した言葉を、選んだ声で届けます"),
                    )
                    .child(div().text_sm().text_color(c(theme::TEXT_FAINT)).child(
                        "話した内容と、それを届けた声が、ここに1組ずつ並びます。",
                    )),
            )
            .child(
                v_flex()
                    .gap_3()
                    .child(step("1", "右の「ライブ開始」を押して話しかける"))
                    .child(step("2", "「届け方」と「声」を選ぶ(いつでも変えられます)"))
                    .child(step("3", "文字で話すときは下の欄に入力して Ctrl+Enter")),
            )
    }

    fn render_turn(&self, turn: &Turn, cx: &mut Context<Self>) -> impl IntoElement {
        let id = turn.id;
        let (status, status_color) = status_label(turn);
        let is_mic = matches!(turn.source, TurnSource::Mic { .. });
        let speaking = turn.status == TurnStatus::Speaking;
        let quiet = matches!(
            turn.status,
            TurnStatus::Unheard | TurnStatus::Skipped | TurnStatus::Cancelled
        );
        let show_delivery = matches!(
            turn.status,
            TurnStatus::Queued | TurnStatus::Speaking | TurnStatus::Done | TurnStatus::Cancelled | TurnStatus::Failed
        ) && !(turn.status == TurnStatus::Cancelled && turn.request.is_none());

        // ---- 上段: 何を話したか
        let said = v_flex()
            .px_4()
            .pt_3()
            .pb_3()
            .gap_1p5()
            .child(
                h_flex()
                    .justify_between()
                    .items_center()
                    .child(
                        h_flex()
                            .gap_1p5()
                            .items_center()
                            .text_color(c(theme::INPUT))
                            .child(Icon::new(if is_mic { IconName::Mic } else { IconName::ALargeSmall }).xsmall())
                            .child(div().text_xs().child(if is_mic { "マイク" } else { "文字入力" })),
                    )
                    .child(
                        h_flex()
                            .gap_1p5()
                            .items_center()
                            .child(div().size(px(6.)).rounded_full().bg(c(status_color)))
                            .child(div().text_xs().text_color(c(status_color)).child(status)),
                    ),
            )
            .child(
                div()
                    .text_base()
                    .line_height(rems(1.6))
                    .text_color(c(match turn.status {
                        TurnStatus::Listening => theme::TEXT_MUTED,
                        _ if quiet => theme::TEXT_FAINT,
                        _ => theme::TEXT,
                    }))
                    .child(if turn.text.is_empty() { "…".to_string() } else { turn.text.clone() }),
            );

        // ---- 確認待ち: 話す / 訂正 / 話さない
        let confirm = (turn.status == TurnStatus::AwaitingConfirm).then(|| {
            h_flex()
                .gap_2()
                .px_4()
                .pb_3()
                .child(
                    Button::new(SharedString::from(format!("confirm-{id}")))
                        .small()
                        .primary()
                        .icon(IconName::Play)
                        .label("この内容で話す")
                        .on_click(cx.listener(move |this, _, _, cx| this.confirm_turn(id, cx))),
                )
                .child(
                    Button::new(SharedString::from(format!("edit-{id}")))
                        .small()
                        .ghost()
                        .label("訂正する")
                        .on_click(cx.listener(move |this, _, window, cx| this.edit_turn(id, window, cx))),
                )
                .child(
                    Button::new(SharedString::from(format!("dismiss-{id}")))
                        .small()
                        .ghost()
                        .label("話さない")
                        .on_click(cx.listener(move |this, _, _, cx| this.dismiss_turn(id, cx))),
                )
        });

        // ---- 下段: 届けた声
        let delivery = show_delivery.then(|| {
            let voice = match turn.status {
                // 受付前は、これから使う声を見せる
                TurnStatus::Queued => self.voice_phrase(),
                _ => voice_phrase(turn.voice.as_deref()),
            };
            let has_audio = !turn.audio_paths().is_empty();
            h_flex()
                .px_4()
                .py_2()
                .gap_3()
                .items_center()
                .border_t_1()
                .border_color(c(theme::BORDER))
                .bg(ca(theme::VOICE, 0x0c))
                .child(div().w(px(3.)).h(px(14.)).rounded_full().bg(c(theme::VOICE)))
                .child(
                    div()
                        .text_xs()
                        .text_color(c(theme::VOICE))
                        .child(voice),
                )
                .when(!turn.chunks.is_empty(), |row| {
                    row.child(h_flex().gap_1().children(turn.chunks.iter().map(|ch| {
                        div()
                            .w(px(14.))
                            .h(px(4.))
                            .rounded_full()
                            .bg(c(if ch.ready { theme::VOICE } else { theme::BORDER_STRONG }))
                    })))
                })
                .when_some(turn.e2e_ms, |row, ms| {
                    row.child(div().text_xs().text_color(c(theme::TEXT_FAINT)).child(format!("応答 {ms}ms")))
                })
                .child(div().flex_1())
                .when(speaking, |row| {
                    row.child(
                        Button::new(SharedString::from(format!("stop-{id}")))
                            .xsmall()
                            .ghost()
                            .icon(IconName::Square)
                            .label("止める")
                            .on_click(cx.listener(|this, _, _, cx| this.cancel_speak(cx))),
                    )
                })
                .when(!turn.status.is_active(), |row| {
                    row.when(has_audio, |row| {
                        row.child(
                            Button::new(SharedString::from(format!("replay-{id}")))
                                .xsmall()
                                .ghost()
                                .icon(IconName::Play)
                                .tooltip("もう一度聞く")
                                .on_click(cx.listener(move |this, _, _, _| this.replay_turn(id))),
                        )
                    })
                    .child(
                        Button::new(SharedString::from(format!("respeak-{id}")))
                            .xsmall()
                            .ghost()
                            .icon(IconName::RefreshCw)
                            .tooltip("今の声と話し方でもう一度話す")
                            .on_click(cx.listener(move |this, _, _, cx| this.respeak_turn(id, cx))),
                    )
                    .child(
                        Button::new(SharedString::from(format!("fix-{id}")))
                            .xsmall()
                            .ghost()
                            .label("訂正")
                            .tooltip("文を入力欄に移して直す")
                            .on_click(cx.listener(move |this, _, window, cx| this.edit_turn(id, window, cx))),
                    )
                })
        });

        v_flex()
            .w_full()
            .max_w(px(CARD_MAX_W))
            .flex_shrink_0()
            .rounded_lg()
            .border_1()
            .border_color(if speaking { ca(theme::VOICE, 0x80) } else { c(theme::BORDER) })
            .bg(c(theme::CARD))
            .overflow_hidden()
            .when(quiet, |card| card.opacity(0.7))
            .child(said)
            .children(confirm)
            .children(delivery)
    }

    fn render_composer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let delivering = self.is_delivering();
        v_flex()
            .px_6()
            .pt_3()
            .pb_3()
            .gap_1p5()
            .border_t_1()
            .border_color(c(theme::BORDER))
            .bg(c(theme::SURFACE))
            .items_center()
            .child(
                h_flex()
                    .w_full()
                    .max_w(px(CARD_MAX_W))
                    .gap_2()
                    .items_end()
                    .child(div().flex_1().child(Textarea::new(&self.composer).text_sm()))
                    .when(delivering, |row| {
                        row.child(
                            Button::new("stop-speaking")
                                .outline()
                                .icon(IconName::Square)
                                .label("止める")
                                .tooltip("合成中・再生中の発話をすべて止める")
                                .on_click(cx.listener(|this, _, _, cx| this.cancel_speak(cx))),
                        )
                    })
                    .child(
                        Button::new("speak")
                            .primary()
                            .icon(IconName::Play)
                            .label("話す")
                            .on_click(cx.listener(|this, _, window, cx| this.speak_from_composer(window, cx))),
                    ),
            )
            .child(
                h_flex()
                    .w_full()
                    .max_w(px(CARD_MAX_W))
                    .justify_between()
                    .text_xs()
                    .text_color(c(theme::TEXT_FAINT))
                    .child("Ctrl+Enter で話す")
                    .child(
                        h_flex()
                            .gap_1p5()
                            .items_center()
                            .child(div().w(px(3.)).h(px(10.)).rounded_full().bg(c(theme::VOICE)))
                            .child(format!("{}で届けます", self.voice_phrase())),
                    ),
            )
    }
}
