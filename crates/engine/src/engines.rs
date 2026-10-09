//! 実 ASR エンジンのアダプタ(各エンジンのクレートを `AsrEngine` に合わせる)。

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sttts_i18n::{tr, trf};

use crate::asr::{AsrEngine, Progress, StreamCb};
use crate::config::{get, get_f64, get_i64, get_str};

pub fn create_real_asr(engine: &str, cfg: &Value, progress: Progress) -> Result<Arc<dyn AsrEngine>> {
    match engine {
        "kotoba" | "whisper" | "faster-whisper" => Ok(Arc::new(WhisperAsr::load(cfg, progress)?)),
        "nemotron" => Ok(Arc::new(NemotronAsr::load(cfg, progress)?)),
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
        bail!(
            "{}",
            tr!(
                "Built without GPU support (the gpu feature of sttts-engine)",
                "GPU 対応なしでビルドされています(sttts-engine の gpu feature)",
                "构建时未启用 GPU 支持(sttts-engine 的 gpu feature)"
            )
        )
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
        progress(&trf!("Fetching the ASR model: {repo}", "ASRモデル取得中: {repo}", "正在获取 ASR 模型:{repo}"));
        let _gpu_load = crate::util::gpu_load_guard(); // TTS のロードと同時に GPU を初期化しない
        let whisper = sttts_whisper::Whisper::load(opts, progress)?;
        progress(&trf!("ASR ready: {repo}", "ASR準備完了: {repo}", "ASR 就绪:{repo}"));
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
        Ok(Self { live: sttts_gemini::GeminiLive::load(opts, progress).context(tr!("initializing Gemini", "Gemini の初期化", "初始化 Gemini"))? })
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

// ---------------------------------------------------------------- Nemotron (ONNX)

/// Nemotron 3.5 ASR(cache-aware FastConformer-RNNT / onnxruntime CPU)。句読点をネイティブに出力する。
/// 途中経過(partial)も確定も `transcribe` で、伸びていく発話は直前の続きから再開するので partial は軽い。
pub struct NemotronAsr {
    nemotron: sttts_nemotron::Nemotron,
}

impl NemotronAsr {
    pub fn load(cfg: &Value, progress: Progress) -> Result<Self> {
        let d = sttts_nemotron::NemotronOptions::default();
        let opts = sttts_nemotron::NemotronOptions {
            model_dir: get_str(cfg, "asr", "nemotron_model_dir").filter(|s| !s.is_empty()).map(Into::into),
            repo: get_str(cfg, "asr", "nemotron_repo").filter(|s| !s.is_empty()).map_or(d.repo, str::to_string),
            chunk_ms: get_i64(cfg, "asr", "nemotron_chunk_ms", i64::from(d.chunk_ms)).max(0) as u32,
            precision: get_str(cfg, "asr", "nemotron_precision").filter(|s| !s.is_empty()).map_or(d.precision, str::to_string),
            language: get_str(cfg, "asr", "language").unwrap_or("ja").to_string(),
            num_threads: get_i64(cfg, "asr", "nemotron_threads", d.num_threads as i64).max(0) as usize,
        };
        Ok(Self { nemotron: sttts_nemotron::Nemotron::load(opts, progress)? })
    }
}

impl AsrEngine for NemotronAsr {
    fn model_id(&self) -> String {
        self.nemotron.model_id().to_string()
    }
    fn transcribe_utterance(&self, audio: &[f32]) -> Result<String> {
        self.nemotron.transcribe(audio)
    }
    fn transcribe_partial(&self, audio: &[f32]) -> Result<String> {
        self.nemotron.transcribe(audio)
    }
    fn max_partial_seconds(&self) -> Option<f64> {
        None // 直前の続きから再開するので、長い発話でも partial は伸びた分しか計算しない
    }
}
