//! 同梱の声プリセット(参照音声)。起動時に `data/voices` へ書き出す。
//!
//! - 声の中身は `crates/engine/assets/voices/<name>.{flac,json}`(Gemini TTS で作った合成音声。実在の人物ではない)。
//! - 書き出した名前は `data/voices/.presets.json` に記録し、利用者が消したプリセットは次の起動で復活させない。
//! - 同名の声(wav / flac / json のどれか)が既にあれば上書きしない(利用者の声が優先)。
//! - 新しいバージョンで増えたプリセットだけが、次の起動で追加される。

use std::path::Path;

use anyhow::{Context, Result};

use crate::say::voices_dir;

macro_rules! preset {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("../assets/voices/", $name, ".flac")),
            include_str!(concat!("../assets/voices/", $name, ".json")),
        )
    };
}

/// (名前, 参照音声 FLAC, 声の設定 JSON)
const PRESETS: &[(&str, &[u8], &str)] = &[
    preset!("genki"),
    preset!("kuudere"),
    preset!("narrator"),
    preset!("ojisan"),
    preset!("oneesan"),
    preset!("seinen"),
    preset!("shounen"),
    preset!("tsundere"),
];

const MARKER: &str = ".presets.json";

/// 同梱プリセットの名前。
pub fn preset_names() -> impl Iterator<Item = &'static str> {
    PRESETS.iter().map(|(n, _, _)| *n)
}

/// まだ書き出していないプリセットを `data/voices` へ書き出し、書き出した名前を返す。
pub fn install_presets(root: &Path) -> Result<Vec<String>> {
    write_presets(root, false)
}

/// 消したプリセットも含め、`data/voices` に無いプリセットを書き戻す(利用者の操作で呼ぶ)。
/// 同名の声があるものは上書きしない。
pub fn restore_presets(root: &Path) -> Result<Vec<String>> {
    write_presets(root, true)
}

/// `data/voices` に無いプリセットの名前(`restore_presets` で戻せるもの)。
pub fn missing_presets(root: &Path) -> Vec<&'static str> {
    let dir = voices_dir(root);
    preset_names().filter(|n| !name_taken(&dir, n)).collect()
}

/// `path` が同梱プリセットの参照音声そのものか(利用者が同じ名前で置いた声と区別する)
pub fn is_preset_audio(name: &str, path: &Path) -> bool {
    PRESETS
        .iter()
        .find(|(n, _, _)| *n == name)
        .is_some_and(|(_, flac, _)| std::fs::metadata(path).is_ok_and(|m| m.len() == flac.len() as u64))
}

fn name_taken(dir: &Path, name: &str) -> bool {
    ["wav", "flac", "json"].iter().any(|e| dir.join(format!("{name}.{e}")).exists())
}

fn write_presets(root: &Path, include_done: bool) -> Result<Vec<String>> {
    let dir = voices_dir(root);
    let marker = dir.join(MARKER);
    let mut done: Vec<String> = match std::fs::read_to_string(&marker) {
        Ok(text) => serde_json::from_str(&text).with_context(|| marker.display().to_string())?,
        Err(_) => Vec::new(),
    };
    let pending: Vec<_> = PRESETS.iter().filter(|(n, _, _)| include_done || !done.iter().any(|d| d == n)).collect();
    if pending.is_empty() {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
    let mut installed = Vec::new();
    for (name, flac, json) in pending {
        if !name_taken(&dir, name) {
            let audio = dir.join(format!("{name}.flac"));
            std::fs::write(&audio, flac).with_context(|| audio.display().to_string())?;
            let conf = dir.join(format!("{name}.json"));
            std::fs::write(&conf, json).with_context(|| conf.display().to_string())?;
            installed.push(name.to_string());
        }
        if !done.iter().any(|d| d == name) {
            done.push(name.to_string());
        }
    }
    std::fs::write(&marker, serde_json::to_string_pretty(&done)? + "\n").with_context(|| marker.display().to_string())?;
    Ok(installed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::say::{VoiceFile, list_voices};

    #[test]
    fn presets_are_valid_voices() {
        for (name, flac, json) in PRESETS {
            assert!(flac.starts_with(b"fLaC"), "{name}: not FLAC");
            serde_json::from_str::<VoiceFile>(json).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    #[test]
    fn install_writes_all_presets_once() {
        let tmp = tempfile::tempdir().unwrap();
        let installed = install_presets(tmp.path()).unwrap();
        assert_eq!(installed, preset_names().map(String::from).collect::<Vec<_>>());
        let voices = list_voices(tmp.path()).unwrap();
        assert_eq!(voices.len(), PRESETS.len());
        assert!(voices.iter().all(|v| v.ref_wav.as_ref().is_some_and(|p| p.extension().unwrap() == "flac")));
        assert!(install_presets(tmp.path()).unwrap().is_empty());
    }

    #[test]
    fn deleted_preset_is_not_restored() {
        let tmp = tempfile::tempdir().unwrap();
        install_presets(tmp.path()).unwrap();
        let dir = voices_dir(tmp.path());
        std::fs::remove_file(dir.join("genki.flac")).unwrap();
        std::fs::remove_file(dir.join("genki.json")).unwrap();
        assert!(install_presets(tmp.path()).unwrap().is_empty());
        assert!(!dir.join("genki.flac").exists());
    }

    #[test]
    fn user_voice_with_same_name_is_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = voices_dir(tmp.path());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("genki.wav"), b"mine").unwrap();
        let installed = install_presets(tmp.path()).unwrap();
        assert!(!installed.iter().any(|n| n == "genki"));
        assert_eq!(std::fs::read(dir.join("genki.wav")).unwrap(), b"mine");
        assert!(!dir.join("genki.flac").exists());
    }

    #[test]
    fn restore_brings_back_deleted_presets_but_keeps_user_voices() {
        let tmp = tempfile::tempdir().unwrap();
        install_presets(tmp.path()).unwrap();
        let dir = voices_dir(tmp.path());
        for e in ["flac", "json"] {
            std::fs::remove_file(dir.join(format!("genki.{e}"))).unwrap();
            std::fs::remove_file(dir.join(format!("kuudere.{e}"))).unwrap();
        }
        std::fs::write(dir.join("kuudere.wav"), b"mine").unwrap();
        assert_eq!(missing_presets(tmp.path()), ["genki"]);
        assert_eq!(restore_presets(tmp.path()).unwrap(), ["genki"]);
        assert!(is_preset_audio("genki", &dir.join("genki.flac")));
        assert!(!is_preset_audio("kuudere", &dir.join("kuudere.wav")));
        assert!(missing_presets(tmp.path()).is_empty());
        // 記録は重複しない
        let marker: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join(MARKER)).unwrap()).unwrap();
        assert_eq!(marker.len(), PRESETS.len());
    }

    #[test]
    fn new_presets_are_added_on_upgrade() {
        // 古いバージョンが一部だけ書き出した状態
        let tmp = tempfile::tempdir().unwrap();
        let dir = voices_dir(tmp.path());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(MARKER), r#"["genki"]"#).unwrap();
        let installed = install_presets(tmp.path()).unwrap();
        assert!(!installed.iter().any(|n| n == "genki"));
        assert_eq!(installed.len(), PRESETS.len() - 1);
    }
}
