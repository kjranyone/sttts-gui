//! 詳細設定シート(既定は閉)。環境で一度決まり、普段は触らない設定を置く:
//! 入出力デバイス、音声合成モデル、seed。Irodori の細かな sampling 項目は
//! data/backend.json の tts.sampling で指定する(GUI が塞がない。AGENTS.md「設計の前提」)。

use gpui_kit::component::button::*;
use gpui_kit::component::input::Input;
use gpui_kit::component::select::Select;
use gpui_kit::component::switch::Switch;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{StttsApp, kit, phase_label};
use crate::theme::{self, c, ca};

const SHEET_W: f32 = 420.;

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
                            .child(self.render_seed(cx))
                            .child(self.render_advanced(cx)),
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

    fn render_seed(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let weak = cx.weak_entity();
        kit::section(
            "乱数(seed)",
            v_flex()
                .gap_2()
                .child(
                    Switch::new("random-seed")
                        .checked(self.random_seed)
                        .label("ランダム")
                        .on_click(move |checked, _, cx| {
                            let _ = weak.update(cx, |this, cx| this.set_random_seed(*checked, cx));
                        }),
                )
                .when(!self.random_seed, |col| col.child(Input::new(&self.seed_input).small())),
        )
    }

    fn render_advanced(&self, cx: &mut Context<Self>) -> impl IntoElement {
        // tts.sampling / tts.codec_*(data/backend.json)と gui.log の置き場所
        kit::section(
            "ファイル",
            Button::new("open-data")
                .small()
                .outline()
                .icon(IconName::FolderOpen)
                .label("data")
                .on_click(cx.listener(|this, _, _, _| this.open_data_folder())),
        )
    }
}
