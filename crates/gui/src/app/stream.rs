//! ストリーム: ターンカード(話した内容 → 届けた声)と、文字で話す入力欄。
//!
//! カードの上段は入力側(桜)、下段は出力側(藤)。二者の会話としては並べず、
//! 1つの入力とその届け方を1枚に対応付ける。

use gpui_kit::component::button::*;
use gpui_kit::component::input::Textarea;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use sttts_i18n::{tr, trf};

use super::help::HelpTopic;
use super::{StttsApp, voice_phrase};
use crate::theme::{self, c, ca};
use crate::turns::{Turn, TurnSource, TurnStatus};

const CARD_MAX_W: f32 = 780.;
/// Irodori-TTS v4 の入力パレット。依存ピン `89f9d8f` の
/// `irodori_tts/gradio_emoji_palette.py` `EMOJI_PALETTE_ITEMS` と、
/// v4 / v4.1 の `EMOJI_ANNOTATIONS.md` と同じ 45 種・同じ順。
/// 意味は公式表の日本語欄(英語・中国語はその訳)。クリックは入力欄のカーソル位置へ入れる。
/// 並びは (絵文字, 英語, 日本語, 中国語)。
const ANNOTATION_CHOICES: &[(&str, &str, &str, &str)] = &[
    ("👂", "Whisper, close to the ear", "囁き、耳元", "耳语、贴耳"),
    ("😮‍💨", "Breath, sigh, sleeping breath", "吐息、溜息、寝息", "吐气、叹气、睡眠呼吸"),
    ("⏸️", "Pause, silence", "間、沈黙", "停顿、沉默"),
    ("🤭", "Giggle, suppressed laugh", "くすくす、含み笑い", "窃笑、偷笑"),
    ("🥵", "Panting, groaning", "喘ぎ、うめき", "喘息、呻吟"),
    ("📢", "Echo, reverb", "エコー、リバーブ", "回声、混响"),
    ("😏", "Teasing, coaxing", "からかう、甘える", "调侃、撒娇"),
    ("🥺", "Trembling voice, unsure", "声を震わせて、自信なさげ", "声音颤抖、没有自信"),
    ("🌬️", "Out of breath, heavy breathing", "息切れ、荒い息", "气喘、粗重呼吸"),
    ("😮", "Gasp", "息をのむ", "倒吸一口气"),
    ("👅", "Licking, chewing sounds", "舐める音、咀嚼音", "舔舐声、咀嚼声"),
    ("💋", "Lip noise", "リップノイズ", "唇齿音"),
    ("🫶", "Gently", "優しく", "温柔地"),
    ("😭", "Crying, sadness", "泣き声、悲しみ", "哭声、悲伤"),
    ("😱", "Scream, shout", "悲鳴、叫び", "尖叫、呼喊"),
    ("😪", "Sleepy, languid", "眠そう、気だるげ", "困倦、慵懒"),
    ("😴", "Sleep talking, snoring", "寝言、いびき", "梦话、打鼾"),
    ("⏩", "Fast talking", "早口", "语速快"),
    ("📞", "Over the phone, through a speaker", "電話越し、スピーカー越し", "电话里、扬声器里"),
    ("🐢", "Slowly", "ゆっくり", "慢慢地"),
    ("🥤", "Swallowing", "唾を飲み込む", "咽口水"),
    ("🤧", "Cough, sneeze, sniffle", "咳、くしゃみ、鼻をすする", "咳嗽、打喷嚏、吸鼻子"),
    ("😒", "Tongue click", "舌打ち", "咂舌"),
    ("😰", "Flustered, nervous, stammering", "慌て、緊張、どもり", "慌张、紧张、结巴"),
    ("😆", "Joyfully", "喜びながら", "高兴地"),
    ("💥", "Forcefully", "勢いよく", "有气势地"),
    ("😠", "Anger, discontent", "怒り、不満", "愤怒、不满"),
    ("😲", "Surprise", "驚き", "惊讶"),
    ("🥱", "Yawn", "あくび", "打哈欠"),
    ("😖", "In distress", "苦しげ", "痛苦地"),
    ("😟", "Worried", "心配そう", "担心地"),
    ("🫣", "Embarrassed, shy", "恥ずかしそう、照れ", "害羞、难为情"),
    ("🙄", "Exasperated", "呆れて", "无语地"),
    ("😊", "Cheerful", "楽しげ", "愉快地"),
    ("😎", "Proud, confident", "得意げ、自信ありげ", "得意、自信"),
    ("👌", "Backchannel (uh-huh)", "相槌", "附和"),
    ("🙏", "Pleading", "懇願", "恳求"),
    ("🥴", "Drunk", "酔っ払って", "醉醺醺地"),
    ("🎵", "Humming", "鼻歌", "哼歌"),
    ("🤐", "Mouth covered", "口を塞がれて", "被捂住嘴"),
    ("😌", "Relief, satisfaction", "安堵、満足", "安心、满足"),
    ("🤔", "Questioning", "疑問", "疑问"),
    ("💪", "Powerfully", "力強く", "有力地"),
    ("👃", "Sniffing", "匂いを嗅ぐ", "闻气味"),
    ("📖", "Narration, monologue", "ナレーション、独白", "旁白、独白"),
];
const _: () = assert!(ANNOTATION_CHOICES.len() == 45);

/// 再生中のチャンクに合わせた本文の色分け(カラオケ風): 再生済み → 藤、再生中 → 地に藤、
/// これから → 控えめ。チャンク文が本文中に見つからなければ None(色分けしない)。
fn playing_highlights(turn: &Turn, playing: u32) -> Option<Vec<(std::ops::Range<usize>, HighlightStyle)>> {
    let mut cursor = 0;
    let mut current = None;
    for ch in &turn.chunks {
        let needle = ch.text.trim();
        if needle.is_empty() {
            continue;
        }
        let Some(pos) = turn.text[cursor..].find(needle) else { continue };
        let range = cursor + pos..cursor + pos + needle.len();
        cursor = range.end;
        if ch.index == playing {
            current = Some(range);
            break;
        }
    }
    let current = current?;
    let style = |color: u32| HighlightStyle { color: Some(c(color).into()), ..Default::default() };
    let mut out = Vec::new();
    if current.start > 0 {
        out.push((0..current.start, style(theme::VOICE)));
    }
    out.push((
        current.clone(),
        HighlightStyle {
            color: Some(c(theme::TEXT).into()),
            background_color: Some(ca(theme::VOICE, 0x55).into()),
            ..Default::default()
        },
    ));
    if current.end < turn.text.len() {
        out.push((current.end..turn.text.len(), style(theme::TEXT_MUTED)));
    }
    Some(out)
}

fn status_label(turn: &Turn, tts_ready: bool, playing: Option<u32>) -> (String, u32) {
    if let Some(index) = playing {
        let position = turn.chunks.iter().position(|ch| ch.index == index).map_or(1, |p| p + 1);
        let total = turn.chunks.len().max(position);
        return (trf!("Playing {position}/{total}", "再生中 {position}/{total}", "播放中 {position}/{total}"), theme::VOICE);
    }
    let secs = turn.waited_secs();
    match turn.status {
        TurnStatus::Listening => (tr!("Listening…", "聞き取り中…", "正在聆听…").into(), theme::INPUT),
        TurnStatus::Transcribing => (tr!("Transcribing…", "文字起こし中…", "正在转写…").into(), theme::TEXT_MUTED),
        TurnStatus::AwaitingConfirm => (tr!("Awaiting confirmation", "確認待ち", "等待确认").into(), theme::WARN),
        TurnStatus::Queued if !tts_ready => (
            trf!("Waiting for TTS · {secs}s", "音声合成の準備待ち · {secs}秒", "等待语音合成就绪 · {secs}秒"),
            theme::WARN,
        ),
        TurnStatus::Queued => (trf!("Queued · {secs}s", "発話待ち · {secs}秒", "等待发话 · {secs}秒"), theme::TEXT_MUTED),
        TurnStatus::Speaking => {
            let (ready, total) = (turn.ready_chunks(), turn.chunks.len());
            if total > 0 && ready == total {
                (tr!("Waiting to play", "再生待ち", "等待播放").into(), theme::VOICE)
            } else {
                let total = total.max(1);
                (
                    trf!(
                        "Synthesizing {ready}/{total} · {secs}s",
                        "合成中 {ready}/{total} · {secs}秒",
                        "合成中 {ready}/{total} · {secs}秒"
                    ),
                    theme::VOICE,
                )
            }
        }
        TurnStatus::Done => (tr!("Delivered", "届けました", "已送达").into(), theme::LIVE),
        TurnStatus::Cancelled => (tr!("Cancelled", "中止", "已取消").into(), theme::TEXT_FAINT),
        TurnStatus::Failed => (tr!("Failed", "失敗", "失败").into(), theme::ERROR),
        TurnStatus::Unheard => (tr!("Couldn't catch that", "聞き取れませんでした", "没有听清").into(), theme::TEXT_FAINT),
        TurnStatus::Skipped => (tr!("Not spoken", "話していません", "未发话").into(), theme::TEXT_FAINT),
        TurnStatus::Interrupted => (tr!("Interrupted", "中断", "已中断").into(), theme::TEXT_FAINT),
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

    fn render_turn(&self, turn: &Turn, cx: &mut Context<Self>) -> AnyElement {
        let id = turn.id;
        // いま鳴っているチャンク(このターンのものなら)
        let playing = self.playback.current().filter(|(t, _)| *t == id).map(|(_, chunk)| chunk);
        let (status, status_color) = status_label(turn, self.tts_state.phase == "ready", playing);
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
                Some(trf!("Expression {emoji} {style}", "表現 {emoji} {style}", "表现 {emoji} {style}"))
            } else if let Some(scale) = d.duration_scale {
                let rate = 100.0 / scale;
                Some(trf!("Speed {rate:.0}%", "話速 {rate:.0}%", "语速 {rate:.0}%"))
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
                                tr!("Mic", "マイク", "麦克风")
                            } else {
                                tr!("Typed", "文字入力", "文字输入")
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
                        TurnStatus::Listening | TurnStatus::Transcribing => theme::TEXT_MUTED,
                        _ if quiet => theme::TEXT_FAINT,
                        _ => theme::TEXT,
                    }))
                    .map(|text| match playing.and_then(|p| playing_highlights(turn, p)) {
                        Some(highlights) => text.child(
                            StyledText::new(turn.text.clone()).with_highlights(highlights),
                        ),
                        None => text.child(if turn.text.is_empty() {
                            "…".to_string()
                        } else {
                            turn.text.clone()
                        }),
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
                        .label(tr!("Speak this", "この内容で話す", "按此内容发话"))
                        .on_click(cx.listener(move |this, _, _, cx| this.confirm_turn(id, cx))),
                )
                .child(
                    Button::new(SharedString::from(format!("edit-{id}")))
                        .small()
                        .ghost()
                        .label(tr!("Correct", "訂正する", "修改"))
                        .on_click(
                            cx.listener(move |this, _, window, cx| this.edit_turn(id, window, cx)),
                        ),
                )
                .child(
                    Button::new(SharedString::from(format!("dismiss-{id}")))
                        .small()
                        .ghost()
                        .label(tr!("Don't speak", "話さない", "不发话"))
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
                .child(
                    div()
                        .flex_shrink_0()
                        .whitespace_nowrap()
                        .text_xs()
                        .text_color(c(theme::VOICE))
                        .child(voice),
                )
                .when(!turn.chunks.is_empty(), |row| {
                    // 長文でチャンクが多いときは、文字を縮めず丸の列の方を折り返す
                    row.child(h_flex().min_w_0().flex_wrap().gap_1().items_center().children(turn.chunks.iter().map(|ch| {
                        let pill = div().h(px(4.)).rounded_full();
                        if playing == Some(ch.index) {
                            // 再生中のチャンクは太く明るく脈打たせる
                            pill.w(px(24.))
                                .h(px(6.))
                                .bg(c(theme::TEXT))
                                .with_animation(
                                    SharedString::from(format!("pill-{id}-{}", ch.index)),
                                    Animation::new(std::time::Duration::from_millis(900))
                                        .repeat()
                                        .with_easing(pulsating_between(0.45, 1.0)),
                                    |el, delta| el.opacity(delta),
                                )
                                .into_any_element()
                        } else {
                            pill.w(px(14.))
                                .bg(c(if ch.ready { theme::VOICE } else { theme::BORDER_STRONG }))
                                .into_any_element()
                        }
                    })))
                })
                .when_some(turn.e2e_ms, |row, ms| {
                    row.child(
                        div()
                            .flex_shrink_0()
                            .whitespace_nowrap()
                            .text_xs()
                            .text_color(c(theme::TEXT_FAINT))
                            .child(trf!("Latency {ms}ms", "応答 {ms}ms", "响应 {ms}ms")),
                    )
                })
                .child(div().flex_1())
                .when(speaking, |row| {
                    row.child(
                        Button::new(SharedString::from(format!("stop-{id}")))
                            .xsmall()
                            .ghost()
                            .icon(IconName::Square)
                            .label(tr!("Stop", "止める", "停止"))
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
                                .tooltip(tr!("Play again", "再生", "重新播放"))
                                .on_click(cx.listener(move |this, _, _, _| this.replay_turn(id))),
                        )
                    })
                    .child(
                        Button::new(SharedString::from(format!("respeak-{id}")))
                            .xsmall()
                            .ghost()
                            .icon(IconName::RefreshCw)
                            .tooltip(tr!("Speak again with the current voice", "再発話", "用当前声音重新发话"))
                            .on_click(cx.listener(move |this, _, _, cx| this.respeak_turn(id, cx))),
                    )
                    .child(
                        Button::new(SharedString::from(format!("fix-{id}")))
                            .xsmall()
                            .ghost()
                            .label(tr!("Correct", "訂正", "修改"))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.edit_turn(id, window, cx)
                            })),
                    )
                })
        });

        let card = v_flex()
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
            .children(delivery);
        if playing.is_none() {
            return card.into_any_element();
        }
        // 再生中: 藤の枠が呼吸するように明滅し、カードがわずかに浮く
        card
            .bg(c(theme::ELEVATED))
            .shadow(vec![BoxShadow {
                color: ca(theme::VOICE, 0x40).into(),
                offset: point(px(0.), px(0.)),
                blur_radius: px(18.),
                spread_radius: px(1.),
                inset: false,
            }])
            .with_animation(
                SharedString::from(format!("playing-{id}")),
                Animation::new(std::time::Duration::from_millis(1400))
                    .repeat()
                    .with_easing(pulsating_between(0.0, 1.0)),
                |card, delta| card.border_color(ca(theme::VOICE, (0x70 as f32 + 0x8f as f32 * delta) as u8)),
            )
            .into_any_element()
    }

    fn render_composer(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
                            .child(tr!("Acting", "演技", "表演")),
                    )
                    .child(self.help_icon(HelpTopic::Acting, cx))
                    .children(ANNOTATION_CHOICES.iter().enumerate().map(
                        |(i, &(emoji, en, ja, zh))| {
                            let description = tr!(en, ja, zh);
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
                    .child(
                        Button::new("speak")
                            .primary()
                            .icon(IconName::Play)
                            .label(tr!("Speak", "話す", "发话"))
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.speak_from_composer(window, cx)
                            })),
                    ),
            )
    }
}
