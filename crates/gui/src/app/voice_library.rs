//! 声のライブラリ(右から出るシート): data/voices の声の一覧・選択・試聴・アイコン設定・削除。
//!
//! 右レールの「声」には選択と追加だけを置き、取り消せない削除はここへ分ける。削除は行の中で確認してから行う。

use gpui_kit::component::button::*;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use sttts_i18n::{tr, trf};

use super::{StttsApp, kit};
use crate::theme::{self, c, ca};

const ICON: f32 = 36.;

impl StttsApp {
    /// レールのボタンから開く(開いていれば閉じる)。詳細設定とは同時に開かない。
    pub(super) fn open_voice_library(&mut self, cx: &mut Context<Self>) {
        if self.voice_library_open {
            self.close_voice_library(cx);
        } else {
            self.voice_library_open = true;
            if self.settings_open {
                self.settings_open = false;
                self.persist_settings(cx);
            }
        }
        cx.notify();
    }

    pub(super) fn close_voice_library(&mut self, cx: &mut Context<Self>) {
        self.voice_library_open = false;
        self.voice_delete_confirm = None;
        if self.previewing_voice.is_some() {
            self.stop_voice_preview(cx);
        }
    }

    pub(super) fn render_voice_library(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let missing = sttts_engine::presets::missing_presets(&self.root);
        let rows: Vec<AnyElement> = self.voices.iter().map(|(name, path)| self.render_voice_row(name, path, cx)).collect();
        let empty = rows.is_empty();
        self.side_sheet(
            "voice-library",
            tr!("Voice library", "声のライブラリ", "声音库"),
            |this, cx| this.close_voice_library(cx),
            v_flex()
                .child(kit::section(
                    tr!("Add", "追加", "添加"),
                    v_flex()
                        .gap_2()
                        .child(
                            h_flex()
                                .gap_2()
                                .child(
                                    Button::new("library-add")
                                        .small()
                                        .icon(IconName::Plus)
                                        .label(tr!("Add a voice", "声を追加", "添加声音"))
                                        .on_click(cx.listener(|this, _, _, cx| this.pick_voice_files(cx))),
                                )
                                .child(
                                    Button::new("library-folder")
                                        .small()
                                        .ghost()
                                        .icon(IconName::FolderOpen)
                                        .label(tr!("Open folder", "フォルダを開く", "打开文件夹"))
                                        .on_click(cx.listener(|this, _, _, cx| this.open_voice_folder(cx))),
                                ),
                        )
                        .child(kit::hint(tr!(
                            "wav / flac, about 10 seconds of speech. You can also drop files onto the window.",
                            "wav / flac(10 秒ほどの話し声)。ウィンドウへのドロップでも追加できます。",
                            "wav / flac(约 10 秒的说话声)。也可以拖放到窗口中添加。"
                        ))),
                ))
                .child(kit::section(
                    tr!("Voices", "声", "声音"),
                    v_flex()
                        .gap_2()
                        .children(rows)
                        .when(empty, |col| col.child(kit::hint(tr!("No voices yet.", "まだ声がありません。", "还没有声音。")))),
                ))
                .when(!missing.is_empty(), |body| {
                    let count = missing.len();
                    let names = missing.join(", ");
                    body.child(kit::section(
                        tr!("Bundled voices", "同梱の声", "内置声音"),
                        v_flex()
                            .gap_2()
                            .child(kit::hint(trf!(
                                "Deleted bundled voices: {names}",
                                "削除した同梱の声: {names}",
                                "已删除的内置声音:{names}"
                            )))
                            .child(
                                Button::new("library-restore")
                                    .small()
                                    .ghost()
                                    .icon(IconName::Undo2)
                                    .label(trf!("Restore ({count})", "戻す({count})", "恢复({count})"))
                                    .on_click(cx.listener(|this, _, window, cx| this.restore_preset_voices(window, cx))),
                            ),
                    ))
                }),
            cx,
        )
    }

    fn render_voice_row(&self, name: &str, path: &std::path::Path, cx: &mut Context<Self>) -> AnyElement {
        let selected = self.selected_voice_name.as_deref() == Some(name);
        let confirming = self.voice_delete_confirm.as_deref() == Some(name);
        let previewing = self.previewing_voice.as_deref() == Some(name);
        let preset = sttts_engine::presets::is_preset_audio(name, path);
        let ext = path.extension().and_then(|x| x.to_str()).unwrap_or_default().to_ascii_uppercase();
        let mut meta = Vec::new();
        if let Some(secs) = self.voice_secs.get(name) {
            let secs = format!("{secs:.1}");
            meta.push(trf!("{secs}s", "{secs} 秒", "{secs} 秒"));
        }
        meta.push(ext);
        if preset {
            meta.push(tr!("bundled", "同梱", "内置").to_string());
        }
        let id = |what: &str| SharedString::from(format!("voice-{what}-{name}"));
        let owned = name.to_string();

        let icon = match self.voice_image(name) {
            Some(img_path) => img(img_path).size(px(ICON)).flex_shrink_0().rounded_md().object_fit(ObjectFit::Cover).into_any_element(),
            None => div()
                .size(px(ICON))
                .flex_shrink_0()
                .rounded_md()
                .flex()
                .items_center()
                .justify_center()
                .bg(ca(theme::VOICE, 0x22))
                .text_color(c(theme::VOICE))
                .font_weight(FontWeight::SEMIBOLD)
                .child(name.chars().next().map(String::from).unwrap_or_default())
                .into_any_element(),
        };

        // 選択は名前の部分だけで受ける(行の中のボタンと取り合わない)
        let pick = {
            let name = owned.clone();
            h_flex()
                .id(id("pick"))
                .flex_1()
                .min_w_0()
                .gap_3()
                .items_center()
                .cursor_pointer()
                .child(icon)
                .child(
                    v_flex()
                        .min_w_0()
                        .child(div().text_sm().text_color(c(theme::TEXT)).truncate().child(owned.clone()))
                        .child(kit::hint(meta.join(" · "))),
                )
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.refresh_voices(Some(&name), window, cx);
                    this.apply_voice(name.clone(), cx);
                    cx.notify();
                }))
        };

        let actions = {
            let (play, icon_for, del) = (owned.clone(), owned.clone(), owned.clone());
            h_flex()
                .gap_0p5()
                .flex_shrink_0()
                .child(
                    Button::new(id("play"))
                        .small()
                        .ghost()
                        .icon(if previewing { IconName::Square } else { IconName::Play })
                        .tooltip(if previewing { tr!("Stop", "停止", "停止") } else { tr!("Preview", "試聴", "试听") })
                        .on_click(cx.listener(move |this, _, _, cx| this.toggle_voice_preview(&play, cx))),
                )
                .child(
                    Button::new(id("icon"))
                        .small()
                        .ghost()
                        .icon(IconName::Palette)
                        .tooltip(tr!("Set an icon image", "アイコン画像を設定", "设置图标图片"))
                        .on_click(cx.listener(move |this, _, _, cx| this.pick_voice_icon(icon_for.clone(), cx))),
                )
                .child(
                    Button::new(id("delete"))
                        .small()
                        .ghost()
                        .icon(IconName::Delete)
                        .tooltip(tr!("Delete…", "削除…", "删除…"))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.voice_delete_confirm = Some(del.clone());
                            cx.notify();
                        })),
                )
        };

        v_flex()
            .px_3()
            .py_2()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(if confirming {
                c(theme::ERROR)
            } else if selected {
                ca(theme::VOICE, 0x99)
            } else {
                c(theme::BORDER)
            })
            .when(selected && !confirming, |row| row.bg(ca(theme::VOICE, 0x14)))
            .when(confirming, |row| row.bg(ca(theme::ERROR, 0x14)))
            .child(h_flex().gap_2().items_center().child(pick).child(actions))
            .when(confirming, |row| row.child(self.render_delete_confirm(&owned, preset, cx)))
            .into_any_element()
    }

    fn render_delete_confirm(&self, name: &str, preset: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let undo = if preset {
            tr!(
                "You can bring it back with \"Restore\" under Bundled voices.",
                "「同梱の声」の「戻す」で元に戻せます。",
                "可以在「内置声音」中点「恢复」找回。"
            )
        } else {
            tr!("This cannot be undone.", "元に戻せません。", "此操作无法撤销。")
        };
        let (yes, id_yes, id_no) = (name.to_string(), SharedString::from(format!("voice-delete-yes-{name}")), SharedString::from(format!("voice-delete-no-{name}")));
        v_flex()
            .gap_2()
            .child(
                div().text_sm().text_color(c(theme::TEXT)).child(trf!(
                    "Delete \"{name}\"? Its reference audio, icon and sttts-say settings (.json) are removed. {undo}",
                    "「{name}」を削除しますか？参照音声・アイコン・sttts-say の設定(.json)を消します。{undo}",
                    "要删除「{name}」吗？将删除参考音频、图标和 sttts-say 设置(.json)。{undo}"
                )),
            )
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(
                        Button::new(id_no)
                            .small()
                            .ghost()
                            .label(tr!("Cancel", "やめる", "取消"))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.voice_delete_confirm = None;
                                cx.notify();
                            })),
                    )
                    .child(
                        Button::new(id_yes)
                            .small()
                            .danger()
                            .icon(IconName::Delete)
                            .label(tr!("Delete", "削除する", "删除"))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.voice_delete_confirm = None;
                                this.delete_voice(&yes, window, cx);
                            })),
                    ),
            )
    }
}
