//! sttts-nemotron
//!
//! Nemotron 3.5 ASR streaming 0.6b(FastConformer-RNNT、ONNX fp16)の純 Rust 実装。
//! `ort`(CPU EP)でキャッシュ対応エンコーダ・予測ネットワーク・ジョイナを回し、
//! 特徴量抽出と貪欲デコードは Rust 側で行う。元実装は Python 版
//! Python 版 `nemotron_onnx_streaming.py`(Apache-2.0、`LICENSE` 参照。git 履歴の backend/ にある)。

mod engine;
mod mel;

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};

pub use mel::SAMPLE_RATE;

pub const DEFAULT_REPO: &str = "codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx";

/// HF パッケージに用意されているチャンク(他は自前 export が必要)
const KNOWN_CHUNKS: [u32; 1] = [320];

#[derive(Debug, Clone)]
pub struct NemotronOptions {
    /// 指定時はこのディレクトリを直接使う(HF から取得しない)
    pub model_dir: Option<PathBuf>,
    pub repo: String,
    pub chunk_ms: u32,
    pub precision: String,
    /// "ja" / "ja-JP" / "auto" など
    pub language: String,
    pub num_threads: usize,
}

impl Default for NemotronOptions {
    fn default() -> Self {
        Self {
            model_dir: None,
            repo: DEFAULT_REPO.into(),
            chunk_ms: 320,
            precision: "fp16".into(),
            language: "ja".into(),
            num_threads: 4,
        }
    }
}

pub struct Nemotron {
    engine: Mutex<engine::Engine>,
    model_id: String,
}

impl Nemotron {
    /// モデルを解決(必要なら HF から取得)してセッションを構築する。
    pub fn load(opts: NemotronOptions, progress: &dyn Fn(&str)) -> Result<Self> {
        let repo = if opts.repo.is_empty() { DEFAULT_REPO.to_string() } else { opts.repo.clone() };
        let precision = if opts.precision.is_empty() { "fp16".to_string() } else { opts.precision.clone() };
        let mut chunk_ms = if opts.chunk_ms == 0 { 320 } else { opts.chunk_ms };
        let num_threads = if opts.num_threads == 0 { 4 } else { opts.num_threads };
        let language = engine::normalize_language(&opts.language);

        let (model_dir, model_id) = match &opts.model_dir {
            Some(d) => (d.clone(), repo),
            None => {
                let dir = resolve_dir(&repo, chunk_ms, &precision, progress)?;
                // 指定チャンクのグラフが無ければ HF パッケージ既定の 320ms に戻す
                if !KNOWN_CHUNKS.contains(&chunk_ms) && !has_chunk(&dir, chunk_ms) {
                    chunk_ms = 320;
                }
                let id = dir.display().to_string();
                (dir, id)
            }
        };
        progress(&format!("Nemotron ONNX 構築中: chunk={chunk_ms}ms {precision}"));
        let engine = engine::Engine::new(&model_dir, &language, chunk_ms, &precision, num_threads)?;
        progress(&format!("ASR準備完了: Nemotron 3.5 ASR (chunk={chunk_ms}ms)"));
        Ok(Self { engine: Mutex::new(engine), model_id })
    }

    /// 解決後のモデル(ディレクトリ、または `model_dir` 指定時はリポジトリ名)
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// 16kHz mono f32 の発話全体 → テキスト(reset → accept_waveform → finish)
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut e = self.engine.lock().map_err(|_| anyhow!("ASR エンジンのロックが壊れています"))?;
        Ok(e.transcribe(audio)?.trim().to_string())
    }
}

fn has_chunk(dir: &std::path::Path, chunk_ms: u32) -> bool {
    let prefix = format!("encoder_{chunk_ms}ms");
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix)))
        .unwrap_or(false)
}

/// 必要なファイル(指定チャンク・精度のグラフと外部データ、トークン、設定)
fn wanted(chunk_ms: u32, precision: &str) -> impl Fn(&str) -> bool {
    let enc = format!("encoder_{chunk_ms}ms");
    let suffix = if precision == "fp32" { String::new() } else { format!("_{precision}") };
    move |f: &str| {
        let base = f.rsplit('/').next().unwrap_or(f);
        if base == "tokens.txt" || base == "nemotron_onnx_config.json" || base.starts_with("LICENSE") || base == "NOTICE" {
            return true;
        }
        // 精度付きのグラフ(外部データ .data を含む)と、その fp32 フォールバック
        let is_graph = |stem: &str| {
            base.starts_with(&format!("{stem}{suffix}.onnx")) || (suffix.is_empty() && base.starts_with(&format!("{stem}.onnx")))
        };
        is_graph(&format!("{enc}_first")) || is_graph(&enc) || is_graph("decoder") || is_graph("joiner")
            || (!suffix.is_empty() && (base == "decoder.onnx" || base == "joiner.onnx"))
    }
}

fn resolve_dir(repo: &str, chunk_ms: u32, precision: &str, progress: &dyn Fn(&str)) -> Result<PathBuf> {
    let required = ["tokens.txt", "nemotron_onnx_config.json", "joiner.onnx"];
    let snap = match sttts_hub::find_snapshot(repo, &required) {
        Some(s) => s,
        None => {
            progress(&format!("ASRモデル取得中: {repo}"));
            sttts_hub::snapshot_download(repo, &wanted(chunk_ms, precision), progress)?
        }
    };
    sttts_hub::materialize_snapshot(&snap).with_context(|| format!("snapshot の展開: {}", snap.display()))
}
