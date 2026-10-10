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
    let dir = voices_dir(root);
    let marker = dir.join(MARKER);
    let mut done: Vec<String> = match std::fs::read_to_string(&marker) {
        Ok(text) => serde_json::from_str(&text).with_context(|| marker.display().to_string())?,
        Err(_) => Vec::new(),
    };
    let pending: Vec<_> = PRESETS.iter().filter(|(n, _, _)| !done.iter().any(|d| d == n)).collect();
    if pending.is_empty() {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
    let mut installed = Vec::new();
    for (name, flac, json) in pending {
        let taken = ["wav", "flac", "json"].iter().any(|e| dir.join(format!("{name}.{e}")).exists());
        if !taken {
            let audio = dir.join(format!("{name}.flac"));
            std::fs::write(&audio, flac).with_context(|| audio.display().to_string())?;
            let conf = dir.join(format!("{name}.json"));
            std::fs::write(&conf, json).with_context(|| conf.display().to_string())?;
            installed.push(name.to_string());
        }
        done.push(name.to_string());
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
