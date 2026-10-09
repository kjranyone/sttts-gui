//! GUI 側の設定永続化(data/config.json)。
//! バックエンドへの設定反映は sttts-protocol::GuiMessage::Configure で行う。

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppSettings {
    #[serde(default)]
    pub mock: Option<bool>,
    #[serde(default)]
    pub tts_model: Option<String>,
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default)]
    pub auto_speak: Option<bool>,
    #[serde(default)]
    pub random_seed: Option<bool>,
    /// 入力デバイス名(None = システム既定)
    #[serde(default)]
    pub input_device: Option<String>,
    /// 出力デバイス名(None = システム既定)
    #[serde(default)]
    pub output_device: Option<String>,
    /// 選択中の声バンク名(data/voices のファイル名。None = 既定の声)
    #[serde(default)]
    pub voice: Option<String>,
    /// ASR プロバイダ(asr.engine 値。None = nemotron)
    #[serde(default)]
    pub asr_provider: Option<String>,
}

pub fn config_path(root: &Path) -> PathBuf {
    root.join("data").join("config.json")
}

impl AppSettings {
    pub fn load(root: &Path) -> Self {
        let path = config_path(root);
        match std::fs::read_to_string(&path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, root: &Path) {
        let path = config_path(root);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }
}
