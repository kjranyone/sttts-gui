//! 詳細設定シート(既定は閉)。環境で一度決まり、普段は触らない設定を置く:
//! 入出力デバイス、音声合成モデル。末尾にリポジトリとクレジット。
//! seed と合成パラメータ(tts.sampling)は Irodori の設定として右レールの「声」に置く。

use gpui_kit::component::button::*;
use gpui_kit::component::select::Select;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{StttsApp, kit, phase_label};
use crate::theme::{self, c, ca};

const SHEET_W: f32 = 420.;
const REPO_URL: &str = "https://github.com/kjranyone/sttts-gui";

impl StttsApp {
    pub(super) fn render_settings_sheet(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let close = cx.listener(|this, _, _, cx| {
            this.settings_open = false;
            this.persist_settings(cx);
            cx.notify();
        });
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(
                // 背景の幕(クリックで閉じる)
                div()
                    .id("settings-scrim")
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full()
                    .bg(ca(0x07060f, 0x99))
                    .on_click(close),
            )
            .child(
                v_flex()
                    .id("settings-sheet")
                    .occlude()
                    .absolute()
                    .top_0()
                    .right_0()
                    .h_full()
                    .w(px(SHEET_W))
                    .bg(c(theme::SURFACE))
                    .border_l_1()
                    .border_color(c(theme::BORDER_STRONG))
                    .shadow_lg()
                    .child(
                        h_flex()
                            .px_4()
                            .py_3()
                            .justify_between()
                            .items_center()
                            .border_b_1()
                            .border_color(c(theme::BORDER))
                            .child(
                                div()
                                    .text_base()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(c(theme::TEXT))
                                    .child("詳細設定"),
                            )
                            .child(
                                Button::new("close-settings")
                                    .small()
                                    .ghost()
                                    .icon(IconName::Close)
                                    .tooltip("閉じる")
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.settings_open = false;
                                        this.persist_settings(cx);
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(
                        v_flex()
                            .id("settings-body")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .child(self.render_devices())
                            .child(self.render_models(cx))
                            .child(self.render_about(cx)),
                    ),
            )
    }

    fn render_devices(&self) -> impl IntoElement {
        kit::section(
            "オーディオデバイス",
            v_flex()
                .gap_3()
                .child(kit::field("マイク(入力)", Select::new(&self.input_select).small()))
                .child(kit::field("スピーカー(出力)", Select::new(&self.output_select).small())),
        )
    }

    fn render_models(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let rows: Vec<AnyElement> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let selected = m.id == self.selected_model_id;
                let meta = [m.size.as_deref(), m.note.as_deref()]
                    .into_iter()
                    .flatten()
                    .filter(|s| !s.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join(" · ");
                h_flex()
                    .id(SharedString::from(format!("model-{i}")))
                    .gap_3()
                    .items_start()
                    .px_3()
                    .py_2()
                    .rounded_md()
                    .border_1()
                    .border_color(if selected { ca(theme::VOICE, 0x99) } else { c(theme::BORDER) })
                    .when(selected, |row| row.bg(ca(theme::VOICE, 0x14)))
                    .hover(|s| s.bg(c(theme::ELEVATED)))
                    .cursor_pointer()
                    .child(
                        div()
                            .mt(px(3.))
                            .size(px(12.))
                            .flex_shrink_0()
                            .rounded_full()
                            .border_2()
                            .border_color(c(if selected { theme::VOICE } else { theme::BORDER_STRONG }))
                            .when(selected, |d| d.bg(c(theme::VOICE))),
                    )
                    .child(
                        v_flex()
                            .gap_0p5()
                            .min_w_0()
                            .child(div().text_sm().text_color(c(theme::TEXT)).child(m.label.clone()))
                            .when(!meta.is_empty(), |col| col.child(kit::hint(meta))),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| this.select_model(i, cx)))
                    .into_any_element()
            })
            .collect();

        kit::section(
            "音声合成モデル",
            v_flex()
                .gap_2()
                .children(rows)
                .child(kit::chip(
                    kit::phase_color(&self.tts_state.phase),
                    format!("音声合成 · {}", phase_label(&self.tts_state)),
                )),
        )
    }

    /// 末尾: リポジトリと軽いクレジット
    fn render_about(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .gap_1()
            .px_4()
            .py_4()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_xs()
                            .text_color(c(theme::TEXT_MUTED))
                            .child(concat!("sttts-gui ", env!("CARGO_PKG_VERSION"))),
                    )
                    .child(
                        Button::new("open-github")
                            .xsmall()
                            .ghost()
                            .icon(IconName::Github)
                            .label("GitHub")
                            .tooltip(REPO_URL)
                            .on_click(cx.listener(|_, _, _, cx| cx.open_url(REPO_URL))),
                    ),
            )
            .child(kit::hint(REPO_URL))
            .child(kit::hint("音声合成 Irodori-TTS(Aratako)· 認識 Nemotron / kotoba-whisper / Gemini"))
            .child(kit::hint("UI GPUI · 推論 burn"))
            .child(kit::hint("© 2026 Kojiro Tanaka · MIT License"))
    }
}
