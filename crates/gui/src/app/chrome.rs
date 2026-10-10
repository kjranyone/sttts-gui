//! 画面の骨格: 全体レイアウト、ステータスバー、ログ欄、ライブ開始/停止中のオーバーレイ。

use gpui_kit::component::button::*;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use sttts_i18n::{tr, trf};

use super::{MicTransition, StttsApp, is_error_line, kit, phase_label};
use crate::theme::{self, c, ca};

impl StttsApp {
    /// エンジンの状態を主画面の最上段に出す(ログを開かなくても気づける)。
    /// エラーは赤、読み込み中は黄(経過秒つき)。
    fn render_engine_error(&self) -> Option<Div> {
        let engines = [
            (tr!("Speech synthesis", "音声合成", "语音合成"), &self.tts_state, self.tts_loading_since),
            (tr!("Recognition", "認識", "识别"), &self.asr_state, self.asr_loading_since),
        ];
        let (what, state, since, is_error) = engines
            .iter()
            .find(|(_, s, _)| s.phase == "error")
            .map(|(w, s, t)| (*w, *s, *t, true))
            .or_else(|| {
                engines
                    .iter()
                    .find(|(_, s, _)| s.phase == "loading")
                    .map(|(w, s, t)| (*w, *s, *t, false))
            })?;
        let color = if is_error { theme::ERROR } else { theme::WARN };
        let mut detail = state.detail.clone().unwrap_or_else(|| {
            if is_error {
                tr!("Error", "エラー", "错误").into()
            } else {
                tr!("Loading", "読み込み中", "加载中").into()
            }
        });
        if let Some(t) = since.filter(|_| !is_error) {
            let secs = t.elapsed().as_secs();
            detail = trf!("{detail} · {secs}s", "{detail} · {secs}秒", "{detail} · {secs}秒");
        }
        Some(
            h_flex()
                .gap_2()
                .items_center()
                .px_4()
                .py_2()
                .bg(ca(color, 0x24))
                .border_b_1()
                .border_color(c(color))
                .text_sm()
                .text_color(c(theme::TEXT))
                .child(div().font_weight(FontWeight::SEMIBOLD).text_color(c(color)).child(what))
                .child(div().min_w_0().flex_1().child(detail)),
        )
    }
}

impl Render for StttsApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // デバイス一覧は window を要求する API なので render で Select へ反映する
        self.input_dev.sync(window, cx);
        self.output_dev.sync(window, cx);

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
                    .children(self.render_engine_error())
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
                            .when(self.settings_open, |d| d.child(self.render_settings_sheet(cx)))
                            .when(self.voice_library_open, |d| d.child(self.render_voice_library(cx))),
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
            .child({
                let (asr, first) = (fmt(self.last_asr_ms), fmt(self.last_first_chunk_ms));
                trf!(
                    "ASR {asr} · first audio {first} · synthesis RTF {rtf}",
                    "認識 {asr} · 初音まで {first} · 合成速度 RTF {rtf}",
                    "识别 {asr} · 首音 {first} · 合成速度 RTF {rtf}"
                )
            })
            .when_some(self.sys, |d, s| {
                let hot = s.vram_ratio().is_some_and(|r| r >= 0.9)
                    || (s.ram_total > 0 && s.ram_used as f64 / s.ram_total as f64 >= 0.9);
                d.child(
                    div()
                        .when(hot, |d| d.text_color(c(theme::WARN)))
                        .child(super::format_sample(&s)),
                )
            })
            .child(div().flex_1())
            .child({
                let (queued, speaking) = self.turns.queue_counts();
                let turns = self.turns.len();
                trf!(
                    "Queued {queued} · synthesizing/playing {speaking} · {turns} turns",
                    "待機 {queued} · 合成/再生 {speaking} · {turns} ターン",
                    "排队 {queued} · 合成/播放 {speaking} · {turns} 轮"
                )
            })
            .child(
                Button::new("status-log")
                    .xsmall()
                    .ghost()
                    .label(if self.unread_errors > 0 {
                        let n = self.unread_errors;
                        trf!("Log ⚠ {n}", "ログ ⚠ {n}", "日志 ⚠ {n}")
                    } else {
                        tr!("Log", "ログ", "日志").into()
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
                    .child(div().text_color(c(theme::TEXT_MUTED)).child(tr!("Log", "ログ", "日志")))
                    .child(
                        Button::new("close-log")
                            .xsmall()
                            .ghost()
                            .label(tr!("Close", "閉じる", "关闭"))
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
                tr!("Stopping live…", "ライブを停止しています…", "正在停止直播…"),
                div(),
            ),
            _ => (
                tr!("Preparing live…", "ライブを準備しています…", "正在准备直播…"),
                div().child(
                    h_flex()
                        .gap_2()
                        .child(kit::chip(
                            kit::phase_color(&self.tts_state.phase),
                            format!("{} · {}", tr!("Speech synthesis", "音声合成", "语音合成"), phase_label(&self.tts_state)),
                        ))
                        .child(kit::chip(
                            kit::phase_color(&self.asr_state.phase),
                            format!("{} · {}", tr!("Recognition", "認識", "识别"), phase_label(&self.asr_state)),
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
