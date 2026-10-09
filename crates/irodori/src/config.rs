//! チェックポイント(safetensors のメタデータ)から読むモデル設定。

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::weights::Weights;

/// `config_json`。使うものだけを定義し、未知の項目は無視する。
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub flow_parameterization: String,
    pub latent_dim: usize,
    pub latent_patch_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    #[serde(default)]
    pub text_mlp_ratio: Option<f64>,
    #[serde(default)]
    pub speaker_mlp_ratio: Option<f64>,
    pub text_vocab_size: usize,
    pub text_add_bos: bool,
    pub text_encoder_type: String,
    #[serde(default)]
    pub pretrained_projector_type: Option<String>,
    #[serde(default)]
    pub pretrained_projector_hidden_ratio: Option<f64>,
    pub text_dim: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub use_caption_condition: bool,
    #[serde(default)]
    pub caption_add_bos: Option<bool>,
    #[serde(default)]
    pub caption_dim: Option<usize>,
    pub use_speaker_condition: bool,
    #[serde(default)]
    pub speaker_dim: Option<usize>,
    #[serde(default)]
    pub speaker_layers: Option<usize>,
    #[serde(default)]
    pub speaker_heads: Option<usize>,
    #[serde(default)]
    pub speaker_patch_size: Option<usize>,
    pub timestep_embed_dim: usize,
    pub adaln_rank: usize,
    pub norm_eps: f64,
    pub use_duration_predictor: bool,
    #[serde(default)]
    pub duration_aux_dim: Option<usize>,
    #[serde(default)]
    pub duration_hidden_dim: Option<usize>,
    #[serde(default)]
    pub duration_layers: Option<usize>,
    #[serde(default)]
    pub duration_attention_heads: Option<usize>,
    #[serde(default)]
    pub duration_architecture: Option<String>,
    #[serde(default)]
    pub duration_token_init_frames: Option<f64>,
    #[serde(default)]
    pub duration_speaker_fusion: Option<String>,
    #[serde(default)]
    pub duration_caption_fusion: Option<String>,
    #[serde(default)]
    pub duration_caption_pooling: Option<String>,
    pub max_text_len: usize,
    #[serde(default)]
    pub max_caption_len: Option<usize>,
    #[serde(default)]
    pub ref_max_seconds: Option<f64>,
}

impl ModelConfig {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let json = w.metadata("config_json").context("config_json metadata missing")?;
        serde_json::from_str(json).context("parse config_json")
    }

    /// DiT が扱う潜在の次元(パッチ化後)
    pub fn patched_latent_dim(&self) -> usize {
        self.latent_dim * self.latent_patch_size
    }

    pub fn is_meanflow(&self) -> bool {
        self.flow_parameterization.eq_ignore_ascii_case("meanflow")
    }
}
