//! 実 ASR エンジンのアダプタ(各エンジンのクレートを `AsrEngine` に合わせる)。

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::asr::{AsrEngine, Progress, StreamCb};
use crate::config::{get, get_f64, get_i64, get_str};

pub fn create_real_asr(engine: &str, cfg: &Value, progress: Progress) -> Result<Arc<dyn AsrEngine>> {
    match engine {
        "kotoba" | "whisper" | "faster-whisper" => Ok(Arc::new(WhisperAsr::load(cfg, progress)?)),
        "gemini" | "gemini_live" => Ok(Arc::new(GeminiAsr::load(cfg, progress)?)),
        other => bail!("unknown asr.engine: {other:?} (choices: kotoba, nemotron, gemini, mock)"),
    }
}

/// 本番は GPU(wgpu)。CPU 推論は実装しない。
pub fn gpu_device() -> Result<irodori::Device> {
    #[cfg(feature = "gpu")]
    {
        irodori::try_gpu_device()
    }
    #[cfg(not(feature = "gpu"))]
    {
        bail!("GPU 対応なしでビルドされています(sttts-engine の gpu feature)")
    }
}

// ---------------------------------------------------------------- kotoba-whisper

/// kotoba-whisper-v2.0(純 Rust / burn)。確定は beam 幅 `asr.final_beam_size`、途中経過は greedy。
pub struct WhisperAsr {
    whisper: sttts_whisper::Whisper,
    final_beam: usize,
}

impl WhisperAsr {
    pub fn load(cfg: &Value, progress: Progress) -> Result<Self> {
        let repo = get_str(cfg, "asr", "model").filter(|m| !m.is_empty() && !m.ends_with("-faster")).unwrap_or(sttts_whisper::DEFAULT_REPO);
        let final_beam = get_i64(cfg, "asr", "final_beam_size", 2).max(1) as usize;
        let opts = sttts_whisper::WhisperOptions {
            repo: repo.to_string(),
            language: get_str(cfg, "asr", "language").unwrap_or("ja").to_string(),
            final_beam_size: final_beam,
            device: gpu_device()?,
        };
        progress(&format!("ASRモデル取得中: {repo}"));
        let whisper = sttts_whisper::Whisper::load(opts, progress)?;
        progress(&format!("ASR準備完了: {repo}"));
        Ok(Self { whisper, final_beam })
    }
}

impl AsrEngine for WhisperAsr {
    fn model_id(&self) -> String {
        self.whisper.model_id().to_string()
    }
    fn transcribe_utterance(&self, audio: &[f32]) -> Result<String> {
        self.whisper.transcribe(audio, self.final_beam)
    }
    fn transcribe_partial(&self, audio: &[f32]) -> Result<String> {
        self.whisper.transcribe(audio, 1)
    }
}

// ---------------------------------------------------------------- Gemini Live

/// Gemini Live API(クラウド)。VAD の開始〜終了に合わせて音声を送り、確定を待つ。
pub struct GeminiAsr {
    live: sttts_gemini::GeminiLive,
}

impl GeminiAsr {
    pub fn load(cfg: &Value, progress: Progress) -> Result<Self> {
        let key = match get(cfg, "asr", "gemini_api_key") {
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            _ => None,
        };
        let opts = sttts_gemini::GeminiOptions {
            model: get_str(cfg, "asr", "gemini_model").unwrap_or_default().to_string(),
            api_key: key,
            language: get_str(cfg, "asr", "language").unwrap_or("ja").to_string(),
            mode: get_str(cfg, "asr", "gemini_mode").unwrap_or("VERBATIM").to_string(),
            timeout_s: get_f64(cfg, "asr", "gemini_timeout_s", 20.0),
            endpoint: None,
        };
        Ok(Self { live: sttts_gemini::GeminiLive::load(opts, progress).context("Gemini の初期化")? })
    }
}

impl AsrEngine for GeminiAsr {
    fn model_id(&self) -> String {
        self.live.model_id().to_string()
    }
    fn transcribe_utterance(&self, audio: &[f32]) -> Result<String> {
        self.live.transcribe_utterance(audio)
    }
    fn transcribe_partial(&self, _audio: &[f32]) -> Result<String> {
        // ストリーミング型は途中経過を begin_stream のコールバックで返す。partial ジョブは使わない
        Ok(String::new())
    }
    fn unload(&self) {
        self.live.abort_all_streams();
    }
    fn is_streaming(&self) -> bool {
        true
    }
    fn begin_stream(&self, utterance: u64, on_partial: StreamCb) {
        if let Err(e) = self.live.begin_stream(utterance, on_partial) {
            eprintln!("[gemini] begin_stream: {e:#}");
        }
    }
    fn feed_stream(&self, utterance: u64, frame: &[f32]) {
        self.live.feed_stream(utterance, frame);
    }
    fn end_stream(&self, utterance: u64) {
        self.live.end_stream(utterance);
    }
    fn abort_stream(&self, utterance: u64) {
        self.live.abort_stream(utterance);
    }
    fn finish_stream(&self, utterance: u64, audio: &[f32]) -> Result<String> {
        self.live.finish_stream(utterance, audio)
    }
    fn abort_all_streams(&self) {
        self.live.abort_all_streams();
    }
}
