//! 声バンク(data/voices): 参照音声の取り込み・選択・削除と、声のアイコン画像。
//!
//! ファイル操作は GUI に依存しない関数に分け、`StttsApp` 側は選択欄とエンジンへの反映だけを持つ。

use std::path::PathBuf;

use gpui_kit::*;
use sttts_i18n::{tr, trf};

use super::StttsApp;

impl StttsApp {
    /// 声バンクの選択適用。参照音声が変わるとウォームアップもやり直される。
    pub(super) fn apply_voice(&mut self, name: String, cx: &mut Context<Self>) {
        self.selected_voice_name = (name != default_voice_label()).then_some(name);
        self.send_voice_config(cx);
        match &self.selected_voice_name {
            Some(n) => self.push_log(trf!(
                "Voice: {n} (synthesizing from the reference audio)",
                "声を切替: {n}(参照音声で合成します)",
                "已切换声音:{n}(使用参考音频合成)"
            )),
            None => self.push_log(
                tr!(
                    "Back to the default voice (synthesizing from the style prompt / automatic voice)",
                    "声を既定に戻しました(話し方の指示/自動音質で合成)",
                    "已恢复默认声音(按说话方式提示/自动音色合成)"
                ),
            ),
        }
        self.persist_settings(cx);
    }

    /// 声ライブラリへ取り込む(ドロップ/ファイル選択の共通入口)。
    /// 音声(wav/flac)は data/voices へコピーして選択し、画像は選択中の声のアイコンにする。
    /// 音声を先に処理するので、音声と画像を同時に渡せば新しい声に画像が付く。
    pub(super) fn import_voice_files(&mut self, paths: &[PathBuf], window: &mut Window, cx: &mut Context<Self>) {
        let dir = self.root.join("data").join("voices");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.push_log(trf!(
                "[error:voice] Cannot create the voice folder: {e}",
                "[error:voice] 声フォルダを作れません: {e}",
                "[error:voice] 无法创建声音文件夹:{e}"
            ));
            return;
        }
        let (audio, rest): (Vec<_>, Vec<_>) = paths.iter().partition(|p| is_voice_audio(p));
        let mut imported = None;
        for src in audio {
            match copy_into_voice_bank(src, &dir) {
                Ok(name) => {
                    self.push_log(trf!("Voice added: {name}", "声を追加: {name}", "已添加声音:{name}"));
                    imported = Some(name);
                }
                Err(e) => self.push_log(format!("[error:voice] {}: {e}", src.display())),
            }
        }
        self.refresh_voices(imported.as_deref(), window, cx);
        if let Some(name) = imported {
            self.apply_voice(name, cx);
        }
        for src in rest {
            if !is_voice_image(src) {
                let file = src.display();
                self.push_log(trf!(
                    "[error:voice] Unsupported file: {file}",
                    "[error:voice] 非対応のファイル: {file}",
                    "[error:voice] 不支持的文件:{file}"
                ));
                continue;
            }
            let Some(name) = self.selected_voice_name.clone() else {
                self.push_log(
                    tr!(
                        "[error:voice] Select a voice before adding an image",
                        "[error:voice] 画像は声を選んでから追加してください",
                        "[error:voice] 请先选择声音再添加图片"
                    ),
                );
                continue;
            };
            match set_voice_image(src, &dir, &name) {
                Ok(()) => self.push_log(trf!("Voice icon set: {name}", "声のアイコンを設定: {name}", "已设置声音图标:{name}")),
                Err(e) => self.push_log(format!("[error:voice] {}: {e}", src.display())),
            }
        }
        cx.notify();
    }

    /// ファイル選択ダイアログから声を追加する。
    pub(super) fn pick_voice_files(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: None,
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                let _ = this.update(cx, |this, cx| {
                    // Window が要るので、取り込みは次の render で実行する
                    this.pending_voice_import = Some(paths);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    /// 声をライブラリから削除する(参照音声・アイコン・sttts-say の設定 json)。選択中なら既定の声に戻す。
    /// 確認は呼び出し側(声のライブラリの行)で済ませておく。
    pub(super) fn delete_voice(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) {
        let dir = self.voices_dir();
        if self.previewing_voice.as_deref() == Some(name) {
            self.stop_voice_preview(cx);
        }
        let mut failed = None;
        if let Some((_, path)) = self.voices.iter().find(|(n, _)| n == name) {
            failed = std::fs::remove_file(path).err();
        }
        if failed.is_none() {
            remove_voice_images(&dir, name);
            let _ = std::fs::remove_file(dir.join(format!("{name}.json")));
            self.push_log(trf!("Voice deleted: {name}", "声を削除: {name}", "已删除声音:{name}"));
        }
        if let Some(e) = failed {
            self.push_log(trf!(
                "[error:voice] Could not delete {name}: {e}",
                "[error:voice] {name} を削除できませんでした: {e}",
                "[error:voice] 无法删除 {name}:{e}"
            ));
        }
        let was_selected = self.selected_voice_name.as_deref() == Some(name);
        self.refresh_voices(None, window, cx);
        if was_selected {
            self.apply_voice(default_voice_label().to_string(), cx);
        }
        cx.notify();
    }

    /// 削除した同梱の声を書き戻す(同じ名前の声があるものは上書きしない)。
    pub(super) fn restore_preset_voices(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match sttts_engine::presets::restore_presets(&self.root) {
            Ok(names) => {
                let names = names.join(", ");
                self.push_log(trf!(
                    "Restored the bundled voices: {names}",
                    "同梱の声を戻しました: {names}",
                    "已恢复内置声音:{names}"
                ));
            }
            Err(e) => self.push_log(trf!(
                "[error:voice] Could not restore the bundled voices: {e:#}",
                "[error:voice] 同梱の声を戻せませんでした: {e:#}",
                "[error:voice] 无法恢复内置声音:{e:#}"
            )),
        }
        self.refresh_voices(None, window, cx);
        cx.notify();
    }

    /// 参照音声を試聴する(合成音の再生とは別に鳴らす)。同じ声をもう一度押すと止める。
    pub(super) fn toggle_voice_preview(&mut self, name: &str, cx: &mut Context<Self>) {
        if self.previewing_voice.as_deref() == Some(name) {
            self.stop_voice_preview(cx);
            return;
        }
        let Some((_, path)) = self.voices.iter().find(|(n, _)| n == name) else { return };
        let Some(audio) = &self.audio else {
            self.push_log(tr!(
                "[error:audio] No output device to play on",
                "[error:audio] 再生できる出力デバイスがありません",
                "[error:audio] 没有可用于播放的输出设备"
            ));
            return;
        };
        let played = std::fs::read(path).map_err(anyhow::Error::from).and_then(|bytes| audio.preview(bytes));
        if let Err(e) = played {
            self.push_log(trf!(
                "[error:audio] Could not play {name}: {e:#}",
                "[error:audio] {name} を再生できませんでした: {e:#}",
                "[error:audio] 无法播放 {name}:{e:#}"
            ));
            return;
        }
        self.previewing_voice = Some(name.to_string());
        cx.notify();
        // 鳴り終わったら再生ボタンを戻す
        cx.spawn(async move |this, cx| loop {
            cx.background_executor().timer(std::time::Duration::from_millis(200)).await;
            let done = this.update(cx, |app, cx| {
                let playing = app.previewing_voice.is_some() && app.audio.as_ref().is_some_and(|a| a.is_previewing());
                if !playing {
                    app.previewing_voice = None;
                    cx.notify();
                }
                !playing
            });
            if done.unwrap_or(true) {
                break;
            }
        })
        .detach();
    }

    pub(super) fn stop_voice_preview(&mut self, cx: &mut Context<Self>) {
        if let Some(audio) = &self.audio {
            audio.stop_preview();
        }
        self.previewing_voice = None;
        cx.notify();
    }

    /// 画像を選んで、その声のアイコンにする。
    pub(super) fn pick_voice_icon(&mut self, name: String, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: None });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            let Some(src) = paths.into_iter().next() else { return };
            let _ = this.update(cx, |this, cx| {
                if !is_voice_image(&src) {
                    let file = src.display();
                    this.push_log(trf!(
                        "[error:voice] Choose an image (png / jpg / webp): {file}",
                        "[error:voice] 画像(png / jpg / webp)を選んでください: {file}",
                        "[error:voice] 请选择图片(png / jpg / webp):{file}"
                    ));
                    return;
                }
                match set_voice_image(&src, &this.voices_dir(), &name) {
                    Ok(()) => this.push_log(trf!("Voice icon set: {name}", "声のアイコンを設定: {name}", "已设置声音图标:{name}")),
                    Err(e) => this.push_log(format!("[error:voice] {}: {e}", src.display())),
                }
                cx.notify();
            });
        })
        .detach();
    }

    pub(super) fn open_voice_folder(&mut self, cx: &mut Context<Self>) {
        let dir = self.voices_dir();
        let _ = std::fs::create_dir_all(&dir);
        cx.open_with_system(&dir);
    }

    pub(super) fn voices_dir(&self) -> PathBuf {
        self.root.join("data").join("voices")
    }

    /// data/voices を再スキャンして選択肢を更新する(`select` が None なら現在の選択を維持)。
    pub(super) fn refresh_voices(&mut self, select: Option<&str>, window: &mut Window, cx: &mut Context<Self>) {
        self.voices = scan_voice_bank(&self.root);
        self.voice_secs = voice_durations(&self.voices);
        let mut items = vec![default_voice_label().to_string()];
        items.extend(self.voices.iter().map(|(n, _)| n.clone()));
        let selected = select
            .map(str::to_string)
            .or_else(|| self.selected_voice_name.clone())
            .filter(|n| self.voices.iter().any(|(vn, _)| vn == n))
            .unwrap_or_else(|| default_voice_label().to_string());
        self.voice_select.update(cx, |s, cx| {
            s.set_items(items, window, cx);
            s.set_selected_value(&selected, window, cx);
        });
    }

    /// 声のアイコン画像(あれば)。
    pub(super) fn voice_image(&self, name: &str) -> Option<PathBuf> {
        find_voice_image(&self.root.join("data").join("voices"), name)
    }
}

/// 声の選択欄で「声の見本を使わない」を表す項目
pub(crate) fn default_voice_label() -> &'static str {
    tr!("Default voice", "既定の声", "默认声音")
}

/// 声の表示名(None = 既定の声)。「〜で届けます」等に続けて使う
pub(crate) fn voice_phrase(name: Option<&str>) -> String {
    match name {
        Some(n) => trf!("Voice: {n}", "{n} の声", "{n} 的声音"),
        None => default_voice_label().to_string(),
    }
}

const VOICE_AUDIO_EXT: &[&str] = &["wav", "flac"];
const VOICE_IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "webp"];

fn has_ext(p: &std::path::Path, exts: &[&str]) -> bool {
    p.extension()
        .and_then(|x| x.to_str())
        .is_some_and(|x| exts.iter().any(|e| x.eq_ignore_ascii_case(e)))
}

fn is_voice_audio(p: &std::path::Path) -> bool {
    has_ext(p, VOICE_AUDIO_EXT)
}

fn is_voice_image(p: &std::path::Path) -> bool {
    has_ext(p, VOICE_IMAGE_EXT)
}

/// data/voices の音声(wav/flac)を声バンクとして読み込む(ファイル名=話者名)。
pub(super) fn scan_voice_bank(root: &std::path::Path) -> Vec<(String, PathBuf)> {
    let dir = root.join("data").join("voices");
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            if is_voice_audio(&p)
                && let Some(name) = p.file_stem().and_then(|s| s.to_str()).filter(|n| !n.is_empty())
            {
                out.push((name.to_string(), p.clone()));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// 参照音声の長さ(秒)。読めないものは載せない。ヘッダだけを読む
pub(super) fn voice_durations(voices: &[(String, PathBuf)]) -> std::collections::HashMap<String, f32> {
    voices
        .iter()
        .filter_map(|(name, path)| {
            let file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
            let secs = rodio::Source::total_duration(&rodio::Decoder::new(file).ok()?)?.as_secs_f32();
            Some((name.clone(), secs))
        })
        .collect()
}

fn find_voice_image(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    VOICE_IMAGE_EXT
        .iter()
        .map(|e| dir.join(format!("{name}.{e}")))
        .find(|p| p.is_file())
}

fn remove_voice_images(dir: &std::path::Path, name: &str) {
    for e in VOICE_IMAGE_EXT {
        let _ = std::fs::remove_file(dir.join(format!("{name}.{e}")));
    }
}

/// 音声を声バンクへコピーし、声の名前(拡張子なし)を返す。同名があれば連番を付ける。
fn copy_into_voice_bank(src: &std::path::Path, dir: &std::path::Path) -> std::io::Result<String> {
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.trim().replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_"))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "voice".to_string());
    let ext = src
        .extension()
        .and_then(|x| x.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "wav".to_string());
    let taken = |n: &str| VOICE_AUDIO_EXT.iter().any(|e| dir.join(format!("{n}.{e}")).exists());
    let mut name = stem.clone();
    let mut i = 2;
    while taken(&name) {
        name = format!("{stem} ({i})");
        i += 1;
    }
    std::fs::copy(src, dir.join(format!("{name}.{ext}")))?;
    Ok(name)
}

/// 画像を声のアイコンとして data/voices/<name>.<ext> へコピーする(既存のアイコンは置換)。
fn set_voice_image(src: &std::path::Path, dir: &std::path::Path, name: &str) -> std::io::Result<()> {
    let ext = src
        .extension()
        .and_then(|x| x.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "png".to_string());
    remove_voice_images(dir, name);
    std::fs::copy(src, dir.join(format!("{name}.{ext}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // super::* は gpui の #[test] を持ち込むので個別に使う
    use super::{copy_into_voice_bank, find_voice_image, scan_voice_bank, set_voice_image};
    use std::path::{Path, PathBuf};

    /// テストごとの空フォルダ(終了時に消す)
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("sttts-gui-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn touch(path: &Path) {
        std::fs::write(path, b"x").unwrap();
    }

    #[test]
    fn import_names_voices_after_the_file_and_numbers_duplicates() {
        let tmp = TempDir::new("voice-import");
        let (src, bank) = (tmp.0.join("src"), tmp.0.join("data").join("voices"));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&bank).unwrap();
        let a = src.join("Sakura.WAV");
        let b = src.join("Sakura.flac");
        touch(&a);
        touch(&b);

        assert_eq!(copy_into_voice_bank(&a, &bank).unwrap(), "Sakura");
        // 拡張子が違っても同じ名前の声は連番で区別する
        assert_eq!(copy_into_voice_bank(&b, &bank).unwrap(), "Sakura (2)");
        touch(&bank.join("notes.txt"));

        let names: Vec<String> = scan_voice_bank(&tmp.0).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["Sakura", "Sakura (2)"]);
    }

    #[test]
    fn voice_image_is_replaced_not_duplicated() {
        let tmp = TempDir::new("voice-image");
        let (png, jpg) = (tmp.0.join("a.png"), tmp.0.join("b.JPG"));
        touch(&png);
        touch(&jpg);
        let bank = tmp.0.join("bank");
        std::fs::create_dir_all(&bank).unwrap();

        assert_eq!(find_voice_image(&bank, "v"), None);
        set_voice_image(&png, &bank, "v").unwrap();
        assert_eq!(find_voice_image(&bank, "v"), Some(bank.join("v.png")));
        set_voice_image(&jpg, &bank, "v").unwrap();
        assert_eq!(find_voice_image(&bank, "v"), Some(bank.join("v.jpg")));
        assert!(!bank.join("v.png").exists());
    }
}
