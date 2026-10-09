//! ストリーム: ターンカード(話した内容 → 届けた声)と、文字で話す入力欄。
//!
//! カードの上段は入力側(桜)、下段は出力側(藤)。二者の会話としては並べず、
//! 1つの入力とその届け方を1枚に対応付ける。

use gpui_kit::component::button::*;
use gpui_kit::component::input::Textarea;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::help::HelpTopic;
use super::{StttsApp, voice_phrase};
use crate::theme::{self, c, ca};
use crate::turns::{Turn, TurnSource, TurnStatus};

const CARD_MAX_W: f32 = 780.;
/// Irodori-TTS v4 の入力パレット。依存ピン `89f9d8f` の
/// `irodori_tts/gradio_emoji_palette.py` `EMOJI_PALETTE_ITEMS` と、
/// v4 / v4.1 の `EMOJI_ANNOTATIONS.md` と同じ 45 種・同じ順。
/// 意味は公式表の日本語欄。`⏸️` だけは末尾へ入れることをツールチップに書く。
const ANNOTATION_CHOICES: &[(&str, &str)] = &[
    ("👂", "囁き、耳元"),
    ("😮‍💨", "吐息、溜息、寝息"),
    ("⏸️", "間、沈黙。末尾に追加"),
    ("🤭", "くすくす、含み笑い"),
    ("🥵", "喘ぎ、うめき"),
    ("📢", "エコー、リバーブ"),
    ("😏", "からかう、甘える"),
    ("🥺", "声を震わせて、自信なさげ"),
    ("🌬️", "息切れ、荒い息"),
    ("😮", "息をのむ"),
    ("👅", "舐める音、咀嚼音"),
    ("💋", "リップノイズ"),
    ("🫶", "優しく"),
    ("😭", "泣き声、悲しみ"),
    ("😱", "悲鳴、叫び"),
    ("😪", "眠そう、気だるげ"),
    ("😴", "寝言、いびき"),
    ("⏩", "早口"),
    ("📞", "電話越し、スピーカー越し"),
    ("🐢", "ゆっくり"),
    ("🥤", "唾を飲み込む"),
    ("🤧", "咳、くしゃみ、鼻をすする"),
    ("😒", "舌打ち"),
    ("😰", "慌て、緊張、どもり"),
    ("😆", "喜びながら"),
    ("💥", "勢いよく"),
    ("😠", "怒り、不満"),
    ("😲", "驚き"),
    ("🥱", "あくび"),
    ("😖", "苦しげ"),
    ("😟", "心配そう"),
    ("🫣", "恥ずかしそう、照れ"),
    ("🙄", "呆れて"),
    ("😊", "楽しげ"),
    ("😎", "得意げ、自信ありげ"),
    ("👌", "相槌"),
    ("🙏", "懇願"),
    ("🥴", "酔っ払って"),
    ("🎵", "鼻歌"),
    ("🤐", "口を塞がれて"),
    ("😌", "安堵、満足"),
    ("🤔", "疑問"),
    ("💪", "力強く"),
    ("👃", "匂いを嗅ぐ"),
    ("📖", "ナレーション、独白"),
];
const _: () = assert!(ANNOTATION_CHOICES.len() == 45);

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
        TurnStatus::Interrupted => ("中断".into(), theme::TEXT_FAINT),
    }
}

impl StttsApp {
    pub(super) fn render_stream(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let body: Vec<AnyElement> = self
            .turns
            .iter()
            .map(|t| self.render_turn(t, cx).into_any_element())
            .collect();
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

    fn render_turn(&self, turn: &Turn, cx: &mut Context<Self>) -> impl IntoElement {
        let id = turn.id;
        let (status, status_color) = status_label(turn);
        let is_mic = matches!(turn.source, TurnSource::Mic { .. });
        let speaking = turn.status == TurnStatus::Speaking;
        let quiet = matches!(
            turn.status,
            TurnStatus::Unheard
                | TurnStatus::Skipped
                | TurnStatus::Cancelled
                | TurnStatus::Interrupted
        );
        let show_delivery = matches!(
            turn.status,
            TurnStatus::Queued
                | TurnStatus::Speaking
                | TurnStatus::Done
                | TurnStatus::Cancelled
                | TurnStatus::Failed
        ) && !(turn.status == TurnStatus::Cancelled && turn.request.is_none());
        let expression = turn.delivery.as_ref().and_then(|d| {
            let emoji = d.emoji.as_deref().unwrap_or("");
            let style = d.style.as_deref().unwrap_or("");
            if !emoji.is_empty() || !style.is_empty() {
                Some(format!("表現 {emoji} {style}"))
            } else if let Some(scale) = d.duration_scale {
                Some(format!("話速 {:.0}%", 100.0 / scale))
            } else {
                None
            }
        });

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
                            .child(
                                Icon::new(if is_mic {
                                    IconName::Mic
                                } else {
                                    IconName::ALargeSmall
                                })
                                .xsmall(),
                            )
                            .child(div().text_xs().child(if is_mic {
                                "マイク"
                            } else {
                                "文字入力"
                            })),
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
                    .child(if turn.text.is_empty() {
                        "…".to_string()
                    } else {
                        turn.text.clone()
                    }),
            )
            .when_some(expression, |col, label| {
                col.child(div().text_xs().text_color(c(theme::VOICE)).child(label))
            });

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
                        .on_click(
                            cx.listener(move |this, _, window, cx| this.edit_turn(id, window, cx)),
                        ),
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
                .child(
                    div()
                        .w(px(3.))
                        .h(px(14.))
                        .rounded_full()
                        .bg(c(theme::VOICE)),
                )
                .child(div().text_xs().text_color(c(theme::VOICE)).child(voice))
                .when(!turn.chunks.is_empty(), |row| {
                    row.child(h_flex().gap_1().children(turn.chunks.iter().map(|ch| {
                        div().w(px(14.)).h(px(4.)).rounded_full().bg(c(if ch.ready {
                            theme::VOICE
                        } else {
                            theme::BORDER_STRONG
                        }))
                    })))
                })
                .when_some(turn.e2e_ms, |row, ms| {
                    row.child(
                        div()
                            .text_xs()
                            .text_color(c(theme::TEXT_FAINT))
                            .child(format!("応答 {ms}ms")),
                    )
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
                                .tooltip("再生")
                                .on_click(cx.listener(move |this, _, _, _| this.replay_turn(id))),
                        )
                    })
                    .child(
                        Button::new(SharedString::from(format!("respeak-{id}")))
                            .xsmall()
                            .ghost()
                            .icon(IconName::RefreshCw)
                            .tooltip("再発話")
                            .on_click(cx.listener(move |this, _, _, cx| this.respeak_turn(id, cx))),
                    )
                    .child(
                        Button::new(SharedString::from(format!("fix-{id}")))
                            .xsmall()
                            .ghost()
                            .label("訂正")
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.edit_turn(id, window, cx)
                            })),
                    )
                })
        });

        v_flex()
            .w_full()
            .max_w(px(CARD_MAX_W))
            .flex_shrink_0()
            .rounded_lg()
            .border_1()
            .border_color(if speaking {
                ca(theme::VOICE, 0x80)
            } else {
                c(theme::BORDER)
            })
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
                    .flex_wrap()
                    .w_full()
                    .max_w(px(CARD_MAX_W))
                    .gap_1()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(c(theme::TEXT_FAINT))
                            .child("演技"),
                    )
                    .child(self.help_icon(HelpTopic::Acting, cx))
                    .children(ANNOTATION_CHOICES.iter().enumerate().map(
                        |(i, &(emoji, description))| {
                            Button::new(SharedString::from(format!("annotation-{i}")))
                                .xsmall()
                                .ghost()
                                .label(emoji)
                                .tooltip(description)
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.insert_annotation(emoji, window, cx)
                                }))
                        },
                    )),
            )
            .child(
                h_flex()
                    .w_full()
                    .max_w(px(CARD_MAX_W))
                    .gap_2()
                    .items_end()
                    .child(
                        div()
                            .flex_1()
                            .child(Textarea::new(&self.composer).text_sm()),
                    )
                    .when(delivering, |row| {
                        row.child(
                            Button::new("stop-speaking")
                                .outline()
                                .icon(IconName::Square)
                                .label("止める")
                                .on_click(cx.listener(|this, _, _, cx| this.cancel_speak(cx))),
                        )
                    })
                    .child(
                        Button::new("speak")
                            .primary()
                            .icon(IconName::Play)
                            .label("話す")
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.speak_from_composer(window, cx)
                            })),
                    ),
            )
            .child(
                h_flex()
                    .w_full()
                    .max_w(px(CARD_MAX_W))
                    .justify_end()
                    .text_xs()
                    .text_color(c(theme::TEXT_FAINT))
                    .child(
                        h_flex()
                            .gap_1p5()
                            .items_center()
                            .child(
                                div()
                                    .w(px(3.))
                                    .h(px(10.))
                                    .rounded_full()
                                    .bg(c(theme::VOICE)),
                            )
                            .child(self.voice_phrase()),
                    ),
            )
    }
}
