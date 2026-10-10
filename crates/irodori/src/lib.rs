//! Irodori-TTS の純 Rust 推論。burn の上に実装し、CPU(flex)は参照・テスト用、
//! 実運用は wgpu(Vulkan)で GPU を使う。PyTorch / Python には依存しない。
//!
//! 構成(数値は PyTorch 実装 `irodori_tts` と段階ごとに一致させる。参照出力は
//! `tools/reference/dump_irodori_ref.py` が `target/irodori-ref/` に書く):
//! - [`weights`] / [`config`]: safetensors の読み込みとモデル設定
//! - [`text`] / [`tokenizer`] / [`modernbert`] / [`condition`]: テキスト・キャプション条件
//! - [`dit`] / [`duration`]: MeanFlow DiT、話者エンコーダ、長さ予測
//! - [`sampler`]: MeanFlow オイラーサンプラ
//! - [`pth`] / [`codec`]: DACVAE(潜在 ⇄ 波形)
//! - [`watermark`]: SilentCipher 透かし
//! - [`pipeline`]: 上記をつないだ `synthesize`

// バイト列 → 数値の変換は `chunks_exact` + `from_le_bytes` と書いたほうが読みやすい(速度は同じ)
#![allow(clippy::chunks_exact_to_as_chunks)]

pub use burn::tensor::{Device, Tensor};
pub use sttts_hub as hub;

#[cfg(feature = "_gpu")]
pub use device::{gpu_device, try_gpu_device};

pub mod codec;
pub mod condition;
pub mod config;
pub mod device;
pub mod dit;
pub mod duration;
pub mod modernbert;
pub mod nn;
pub mod pipeline;
pub mod pth;
pub mod sampler;
pub mod t5gemma;
pub mod testing;
pub mod text;
pub mod tokenizer;
pub mod watermark;
pub mod weights;
