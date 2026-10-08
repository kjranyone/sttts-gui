//! sttts-gui の GUI(Rust)⇄ バックエンド(Python)間 stdio NDJSON プロトコル定義。
//!
//! 1メッセージ = 1行のJSONオブジェクト。`type` フィールドがタグ。
//! Python 側の対応実装は `backend/src/sttts_server/protocol.py`。
//! 両者は必ず同期して変更すること。

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;

/// エンジン(TTS/ASR)の状態。phase は "idle" | "loading" | "ready" | "error"。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineState {
    pub phase: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// モデルエイリアス(例: "v4.1-small-mf")
    pub id: String,
    /// UI表示名
    pub label: String,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioDeviceInfo {
    pub index: i64,
    pub name: String,
    #[serde(default)]
    pub default_rate: Option<u32>,
    #[serde(default)]
    pub is_default: bool,
}

/// backend → GUI メッセージ。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendMessage {
    Hello {
        protocol: u32,
        mock: bool,
        #[serde(default)]
        python: Option<String>,
        #[serde(default)]
        backend_version: Option<String>,
        #[serde(default)]
        models: Vec<ModelInfo>,
    },
    Devices {
        inputs: Vec<AudioDeviceInfo>,
        outputs: Vec<AudioDeviceInfo>,
    },
    State {
        tts: EngineState,
        asr: EngineState,
        mic_running: bool,
    },
    Log {
        level: String,
        message: String,
    },
    MicLevel {
        rms: f32,
        db: f32,
    },
    AsrPartial {
        utterance: u64,
        text: String,
    },
    AsrFinal {
        utterance: u64,
        text: String,
    },
    SpeakAccepted {
        request: u64,
        /// "manual" | "auto"
        origin: String,
        #[serde(default)]
        tag: Option<String>,
    },
    TtsChunkStart {
        request: u64,
        chunk: u32,
        text: String,
    },
    TtsAudio {
        request: u64,
        chunk: u32,
        wav_base64: String,
        sample_rate: u32,
        duration_ms: u64,
        gen_ms: u64,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        seed: Option<i64>,
    },
    TtsChunkDone {
        request: u64,
        chunk: u32,
        gen_ms: u64,
    },
    SpeakDone {
        request: u64,
        chunks: u32,
        #[serde(default)]
        cancelled: bool,
        #[serde(default)]
        failed: bool,
    },
    Error {
        scope: String,
        message: String,
        recoverable: bool,
    },
    Pong {
        nonce: u64,
    },
}

/// GUI → backend メッセージ。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GuiMessage {
    Configure {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tts: Option<TtsConfig>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        asr: Option<AsrConfig>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio: Option<AudioConfig>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        voice: Option<VoiceConfig>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pipeline: Option<PipelineConfig>,
    },
    StartSession,
    StopSession,
    Speak {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caption: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ref_wavs: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tag: Option<String>,
    },
    CancelSpeak,
    Ping {
        nonce: u64,
    },
    Shutdown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TtsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// "auto" | "xpu" | "cpu"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_mode: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsrConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// CTranslate2 compute type: "int8" | "float32" ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_interval_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_device_index: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VoiceConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_wavs: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_ref: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// ASR確定文を自動でTTSに回すか
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_speak: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_min_chars: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_chunk_min_chars: Option<u32>,
}

/// 既知/未知をまとめて受けるラッパ。未知のtypeはUI側でログ表示に回す。
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnyMessage {
    Known(BackendMessage),
    Unknown(serde_json::Value),
}

impl AnyMessage {
    pub fn type_name(&self) -> String {
        match self {
            AnyMessage::Known(m) => match m {
                BackendMessage::Hello { .. } => "hello",
                BackendMessage::Devices { .. } => "devices",
                BackendMessage::State { .. } => "state",
                BackendMessage::Log { .. } => "log",
                BackendMessage::MicLevel { .. } => "mic_level",
                BackendMessage::AsrPartial { .. } => "asr_partial",
                BackendMessage::AsrFinal { .. } => "asr_final",
                BackendMessage::SpeakAccepted { .. } => "speak_accepted",
                BackendMessage::TtsChunkStart { .. } => "tts_chunk_start",
                BackendMessage::TtsAudio { .. } => "tts_audio",
                BackendMessage::TtsChunkDone { .. } => "tts_chunk_done",
                BackendMessage::SpeakDone { .. } => "speak_done",
                BackendMessage::Error { .. } => "error",
                BackendMessage::Pong { .. } => "pong",
            }
            .to_string(),
            AnyMessage::Unknown(v) => v
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("<no type>")
                .to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_messages() {
        let m: AnyMessage =
            serde_json::from_str(r#"{"type":"hello","protocol":1,"mock":true}"#).unwrap();
        match m {
            AnyMessage::Known(BackendMessage::Hello { protocol, mock, .. }) => {
                assert_eq!((protocol, mock), (1, true));
            }
            _ => panic!("expected hello"),
        }

        let m: AnyMessage = serde_json::from_str(
            r#"{"type":"tts_audio","request":1,"chunk":0,"wav_base64":"QUJD","sample_rate":48000,"duration_ms":500,"gen_ms":120}"#,
        )
        .unwrap();
        match m {
            AnyMessage::Known(BackendMessage::TtsAudio {
                wav_base64,
                sample_rate,
                path,
                ..
            }) => {
                assert_eq!(wav_base64, "QUJD");
                assert_eq!(sample_rate, 48000);
                assert_eq!(path, None);
            }
            _ => panic!("expected tts_audio"),
        }
    }

    #[test]
    fn unknown_type_falls_back() {
        let m: AnyMessage = serde_json::from_str(r#"{"type":"future_thing","x":1}"#).unwrap();
        assert_eq!(m.type_name(), "future_thing");
    }

    #[test]
    fn serialize_gui_speak_minimal() {
        let s = serde_json::to_string(&GuiMessage::Speak {
            text: "こんにちは".into(),
            caption: None,
            ref_wavs: None,
            seed: None,
            tag: None,
        })
        .unwrap();
        assert_eq!(s, r#"{"type":"speak","text":"こんにちは"}"#);
    }
}
