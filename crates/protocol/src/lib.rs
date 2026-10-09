//! sttts-gui の GUI(Rust)⇄ バックエンド(Python)間 stdio NDJSON プロトコル定義。
//!
//! 1メッセージ = 1行のJSONオブジェクト。`type` フィールドがタグ。
//! Python 側の対応実装は `backend/src/sttts_server/protocol.py`。
//! 両者は必ず同期して変更すること。

use std::collections::BTreeMap;

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
        /// partial デコード時間(ms)
        #[serde(default)]
        asr_ms: Option<u64>,
    },
    /// 計測用フィールドはすべて任意(古い backend とも互換)。
    /// `*_ms` の時刻は backend プロセスの monotonic 時計基準(GUI の時計とは比較しない)。
    AsrFinal {
        utterance: u64,
        text: String,
        /// 発話終了(話し終わり)推定時刻(backend monotonic ms)
        #[serde(default)]
        speech_end_ms: Option<f64>,
        /// 話し終わり → VAD が発話終了を確定するまで(ms)
        #[serde(default)]
        vad_wait_ms: Option<u64>,
        /// 確定デコード時間(ms)
        #[serde(default)]
        asr_ms: Option<u64>,
        /// 発話音声の長さ(ms)
        #[serde(default)]
        audio_ms: Option<u64>,
        /// 元音声から得た発話単位の表現。転写文には挿入しない。
        #[serde(default)]
        delivery: Option<DeliveryInfo>,
        #[serde(default)]
        pause_ms: Option<u64>,
    },
    SpeakAccepted {
        request: u64,
        /// "manual" | "auto"
        origin: String,
        #[serde(default)]
        tag: Option<String>,
        /// 自動発話の元になった ASR 発話 id
        #[serde(default)]
        utterance: Option<u64>,
        #[serde(default)]
        speech_end_ms: Option<f64>,
        #[serde(default)]
        delivery: Option<DeliveryInfo>,
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
        /// 先頭チャンクか
        #[serde(default)]
        first_chunk: bool,
        /// 投機的 TTS(確定前の partial から先行合成)の結果を流用したチャンクか
        #[serde(default)]
        speculative: bool,
        /// 合成時間 / 音声長
        #[serde(default)]
        rtf: Option<f64>,
        /// TTS キューでの待ち時間(ms)
        #[serde(default)]
        queue_wait_ms: Option<u64>,
        /// 先頭チャンクのみ: 発話受付 → 送出(ms)
        #[serde(default)]
        first_chunk_ms: Option<u64>,
        /// 先頭チャンクのみ: 話し終わり → 送出(ms)。自動発話のときだけ
        #[serde(default)]
        e2e_ms: Option<u64>,
        /// Irodori の段階別時間(ms)。例: predict_duration / sample_meanflow / decode_latent
        #[serde(default)]
        stages: Option<BTreeMap<String, f64>>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliveryInfo {
    #[serde(default)]
    pub emoji: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub duration_scale: Option<f64>,
    #[serde(default)]
    pub emotion: Option<String>,
    pub source: String,
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
        /// ASR 発話を確認してから話す場合に表現計画を保持する。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delivery: Option<DeliveryInfo>,
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
    /// "auto" | "cuda" | "xpu" | "cpu"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_mode: Option<String>,
    /// "auto" | "fp32" | "bf16"(auto: CUDA cc<8.0 → fp32)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_conditions: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_latent_cache: Option<bool>,
    /// Irodori の SamplingRequest 項目の上書き(cfg_scale_text / duration_scale 等。項目名は Irodori 側と同じ)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsrConfig {
    /// "kotoba" | "reazonspeech"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// "auto" | "cuda" | "cpu"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// CTranslate2 compute type: "auto" | "int8" | "float16" | "float32" ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_interval_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vad_min_silence_ms: Option<u64>,
    /// Gemini(engine = "gemini")の API キー。空文字は「GUI では未設定」
    /// (backend は環境変数 GEMINI_API_KEY / GOOGLE_API_KEY にフォールバック)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemini_api_key: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_max_chars: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_chunk_mora_min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_chunk_mora_max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_tts: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_stable_partials: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub performance_enabled: Option<bool>,
    /// "none" | "emotion2vec"。感情推定は CPU で ASR と並行実行する。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion_engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub performance_wait_ms: Option<u32>,
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
    fn parse_timing_fields() {
        let m: AnyMessage = serde_json::from_str(
            r#"{"type":"asr_final","utterance":3,"text":"こんにちは","speech_end_ms":12345.6,"vad_wait_ms":290,"asr_ms":180,"audio_ms":2100}"#,
        )
        .unwrap();
        match m {
            AnyMessage::Known(BackendMessage::AsrFinal { utterance, speech_end_ms, vad_wait_ms, asr_ms, audio_ms, .. }) => {
                assert_eq!(utterance, 3);
                assert_eq!(speech_end_ms, Some(12345.6));
                assert_eq!((vad_wait_ms, asr_ms, audio_ms), (Some(290), Some(180), Some(2100)));
            }
            _ => panic!("expected asr_final"),
        }

        let m: AnyMessage = serde_json::from_str(
            r#"{"type":"tts_audio","request":2,"chunk":0,"wav_base64":"","sample_rate":48000,"duration_ms":1200,"gen_ms":300,
               "first_chunk":true,"speculative":true,"rtf":0.25,"queue_wait_ms":3,"first_chunk_ms":310,"e2e_ms":820,
               "stages":{"predict_duration":20.5,"sample_meanflow":150.0},"path":null,"seed":7}"#,
        )
        .unwrap();
        match m {
            AnyMessage::Known(BackendMessage::TtsAudio { first_chunk, speculative, rtf, e2e_ms, first_chunk_ms, stages, queue_wait_ms, .. }) => {
                assert!(first_chunk && speculative);
                assert_eq!(rtf, Some(0.25));
                assert_eq!((e2e_ms, first_chunk_ms, queue_wait_ms), (Some(820), Some(310), Some(3)));
                assert_eq!(stages.unwrap()["sample_meanflow"], 150.0);
            }
            _ => panic!("expected tts_audio"),
        }
    }

    #[test]
    fn timing_fields_are_optional_for_old_backends() {
        let m: AnyMessage =
            serde_json::from_str(r#"{"type":"asr_final","utterance":1,"text":"x"}"#).unwrap();
        assert!(matches!(m, AnyMessage::Known(BackendMessage::AsrFinal { asr_ms: None, .. })));
        let m: AnyMessage = serde_json::from_str(
            r#"{"type":"speak_accepted","request":1,"origin":"auto","utterance":null,"speech_end_ms":null}"#,
        )
        .unwrap();
        assert!(matches!(m, AnyMessage::Known(BackendMessage::SpeakAccepted { utterance: None, .. })));
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
            delivery: None,
        })
        .unwrap();
        assert_eq!(s, r#"{"type":"speak","text":"こんにちは"}"#);
    }
}
