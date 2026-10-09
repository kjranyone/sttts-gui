//! kotoba-whisper-v2.0(distil-whisper large-v3 系)の純 Rust 推論。burn の上に実装し、CPU(flex)は
//! 参照・テスト用、実運用は wgpu で GPU を使う(`irodori` と同じデバイス構成を使う)。
//! PyTorch / Python / CTranslate2 には依存しない。
//!
//! 構成(数値は transformers の `WhisperForConditionalGeneration` と段階ごとに一致させる。参照出力は
//! `backend/scripts/dump_whisper_ref.py` が `target/whisper-ref/` に書く):
//! - [`mel`]: ログメル特徴(`WhisperFeatureExtractor` 互換、CPU)
//! - [`model`]: encoder(Conv1d x2 + 32 層)と decoder(2 層、self / cross-attention の KV キャッシュ)
//! - [`decode`]: greedy / beam search
//! - [`fetch`]: HF キャッシュ上の場所と、無ければダウンロード
//!
//! 30 秒を超える音声は 30 秒ずつ順次デコードして連結する(窓をまたぐ文脈の引き継ぎはしない:
//! 元の faster-whisper 設定の `condition_on_previous_text=False` と同じ)。

// バイト列 → 数値の変換は `chunks_exact` + `from_le_bytes` と書いたほうが読みやすい(速度は同じ)
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod decode;
pub mod fetch;
pub mod mel;
pub mod model;

use std::path::Path;

use anyhow::{Context, Result, anyhow};
pub use irodori::{Device, Tensor};
use irodori::weights::Weights;

use crate::decode::DecodeSpec;
use crate::mel::{MelExtractor, N_SAMPLES, SAMPLE_RATE};
use crate::model::Model;

/// 既定のモデル
pub const DEFAULT_REPO: &str = "kotoba-tech/kotoba-whisper-v2.0";

pub struct WhisperOptions {
    /// HF のリポジトリ(例: [`DEFAULT_REPO`])
    pub repo: String,
    /// 言語コード(例: `ja`)
    pub language: String,
    /// 確定(final)デコードの beam 幅。`transcribe` に渡す値の目安として保持する
    pub final_beam_size: usize,
    /// 実行デバイス。`irodori::device::cpu_device()`(参照・テスト用)か `irodori::gpu_device()`
    pub device: Device,
}

pub struct Whisper {
    opts: WhisperOptions,
    model: Model,
    tokenizer: tokenizers::Tokenizer,
    mel: MelExtractor,
    spec: DecodeSpec,
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parse {}", path.display()))
}

fn id_list(v: &serde_json::Value, key: &str) -> Vec<usize> {
    v[key].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).unwrap_or_default()
}

impl Whisper {
    /// HF キャッシュ(無ければダウンロード)からモデルを読み込む。`progress` に進捗の文言を渡す。
    pub fn load(opts: WhisperOptions, progress: &dyn Fn(&str)) -> Result<Self> {
        let dir = fetch::ensure(&opts.repo, progress)?;
        Self::load_from(opts, &dir, progress)
    }

    /// スナップショットのディレクトリ(`model.safetensors` / `tokenizer.json` / `generation_config.json`)から読み込む。
    pub fn load_from(opts: WhisperOptions, dir: &Path, progress: &dyn Fn(&str)) -> Result<Self> {
        progress("tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let tok_id = |t: &str| -> Result<i64> {
            tokenizer.token_to_id(t).map(i64::from).ok_or_else(|| anyhow!("tokenizer に {t} がありません"))
        };
        let gen_cfg = read_json(&dir.join("generation_config.json"))?;
        let eos = tok_id("<|endoftext|>")?;
        let spec = DecodeSpec {
            prompt: vec![
                tok_id("<|startoftranscript|>")?,
                tok_id(&format!("<|{}|>", opts.language))?,
                tok_id("<|transcribe|>")?,
                tok_id("<|notimestamps|>")?,
            ],
            eos,
            suppress: id_list(&gen_cfg, "suppress_tokens"),
            begin_suppress: id_list(&gen_cfg, "begin_suppress_tokens"),
        };
        progress("重みを読み込み中");
        let weights = Weights::open(dir.join("model.safetensors"))?;
        let model = Model::load(&weights, &opts.device, progress)?;
        progress("準備完了");
        Ok(Self { opts, model, tokenizer, mel: MelExtractor::new(), spec })
    }

    /// 読み込んだモデルの識別子(リポジトリ名)
    pub fn model_id(&self) -> &str {
        &self.opts.repo
    }

    pub fn options(&self) -> &WhisperOptions {
        &self.opts
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn mel(&self) -> &MelExtractor {
        &self.mel
    }

    /// 16kHz モノ f32 の音声を文字起こしする(`beam_size` 1 で greedy)。30 秒を超えるときは
    /// 30 秒ずつ順次デコードして連結する。
    pub fn transcribe(&self, audio: &[f32], beam_size: usize) -> Result<String> {
        let mut out = String::new();
        for (i, win) in audio.chunks(N_SAMPLES).enumerate() {
            // 末尾の極端に短い切れ端(0.1 秒未満)は無音扱いで読み飛ばす
            if win.len() < SAMPLE_RATE / 10 && (i > 0 || win.is_empty()) {
                continue;
            }
            out.push_str(&self.transcribe_window(win, beam_size)?);
        }
        Ok(out)
    }

    fn transcribe_window(&self, win: &[f32], beam_size: usize) -> Result<String> {
        let mel = self.mel.log_mel(win);
        let enc = self.model.encode(&mel);
        let cross = self.model.cross_kv(&enc);
        let ids = decode::generate(&self.model, &self.spec, &cross, beam_size)?;
        self.decode_text(&ids)
    }

    /// トークン列 → テキスト(特殊トークンは除く、前後の空白は削る)
    pub fn decode_text(&self, ids: &[i64]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let s = self.tokenizer.decode(&ids, true).map_err(|e| anyhow!("detokenize: {e}"))?;
        Ok(s.trim().to_string())
    }

    pub fn spec(&self) -> &DecodeSpec {
        &self.spec
    }
}
