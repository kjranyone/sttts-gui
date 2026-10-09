//! テキスト・キャプション条件。`TextToLatentRFDiT.encode_conditions` のテキスト/キャプション側
//! (話者側は `dit` の担当)。
//!
//! `PretrainedConditionProjector`(residual_mlp)は ModernBERT の出力
//! `h` から `Linear(h) + down(silu(up(rms_norm(h))))` を作り、マスクを掛ける。続けて
//! `text_norm` / `caption_norm`(RMSNorm)を通す。
//!
//! キャプションが空文字のときの扱いは原典 `inference_runtime` と同じで、トークナイズ後に
//! マスクを全て false にする([`caption_inputs`])。全マスク行はバックボーンの出力が 0 になり、
//! 射影・ノルム後も 0 のまま。

use anyhow::{Context, Result, bail};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Tensor, TensorData};

use crate::config::ModelConfig;
use crate::modernbert::{ModernBert, linear};
use crate::tokenizer::{Encoded, Tokenizer};
use crate::weights::Weights;

fn add_bias(x: Tensor<3>, b: &Tensor<1>) -> Tensor<3> {
    let n = b.dims()[0];
    x + b.clone().reshape([1, 1, n])
}

/// RMSNorm(最終軸、`x * rsqrt(mean(x^2) + eps) * weight`)
fn rms_norm(x: Tensor<3>, weight: &Tensor<1>, eps: f64) -> Tensor<3> {
    let n = weight.dims()[0];
    let r = (x.clone().square().mean_dim(2) + eps).sqrt().recip();
    x * r * weight.clone().reshape([1, 1, n])
}

fn mask_tensor(mask: &[Vec<bool>], dev: &Device) -> Tensor<3> {
    let (b, s) = (mask.len(), mask[0].len());
    let v: Vec<f32> = mask.iter().flatten().map(|&m| if m { 1.0 } else { 0.0 }).collect();
    Tensor::<3>::from_data(TensorData::new(v, [b, s, 1]), dev)
}

/// `PretrainedConditionProjector`(`residual_mlp`)
pub struct ConditionProjector {
    proj_w: Tensor<2>,
    proj_b: Tensor<1>,
    res_norm: Tensor<1>,
    res_up_w: Tensor<2>,
    res_up_b: Tensor<1>,
    res_down_w: Tensor<2>,
    res_down_b: Tensor<1>,
    eps: f64,
}

impl ConditionProjector {
    /// `prefix` は `text_encoder` / `caption_encoder`
    pub fn load(w: &Weights, prefix: &str, eps: f64, dev: &Device) -> Result<Self> {
        let t2 = |n: &str| -> Result<Tensor<2>> { Ok(crate::ops::store(w.tensor::<2>(&format!("{prefix}.{n}"), dev)?.transpose())) };
        let t1 = |n: &str| w.tensor::<1>(&format!("{prefix}.{n}"), dev);
        Ok(Self {
            proj_w: t2("projector.weight")?,
            proj_b: t1("projector.bias")?,
            res_norm: t1("residual_norm.weight")?,
            res_up_w: t2("residual_up.weight")?,
            res_up_b: t1("residual_up.bias")?,
            res_down_w: t2("residual_down.weight")?,
            res_down_b: t1("residual_down.bias")?,
            eps,
        })
    }

    /// `state`: バックボーン出力 `[B,S,backbone_dim]`、`mask`: `[B,S,1]`(0/1)
    pub fn forward(&self, state: Tensor<3>, mask: Tensor<3>) -> Tensor<3> {
        let projected = add_bias(linear(state.clone(), &self.proj_w), &self.proj_b);
        let r = rms_norm(state, &self.res_norm, self.eps);
        let r = silu(add_bias(linear(r, &self.res_up_w), &self.res_up_b));
        let r = add_bias(linear(r, &self.res_down_w), &self.res_down_b);
        (projected + r) * mask
    }
}

/// ModernBERT(text/caption 共有)+ 射影 + ノルム。
pub struct TextConditioner {
    pub cfg: ModelConfig,
    backbone: ModernBert,
    text_encoder: ConditionProjector,
    text_norm: Tensor<1>,
    caption: Option<(ConditionProjector, Tensor<1>)>,
    eps: f64,
    device: Device,
}

impl TextConditioner {
    pub fn load(w: &Weights, dev: &Device) -> Result<Self> {
        let cfg = ModelConfig::from_weights(w)?;
        if cfg.text_encoder_type != "pretrained" {
            bail!("unsupported text_encoder_type {:?}", cfg.text_encoder_type);
        }
        if cfg.pretrained_projector_type.as_deref() != Some("residual_mlp") {
            bail!("unsupported pretrained_projector_type {:?}", cfg.pretrained_projector_type);
        }
        let eps = cfg.norm_eps;
        let backbone = ModernBert::load(w, dev)?;
        let text_encoder = ConditionProjector::load(w, "text_encoder", eps, dev)?;
        let text_norm = w.tensor::<1>("text_norm.weight", dev).context("text_norm")?;
        let caption = if cfg.use_caption_condition {
            Some((
                ConditionProjector::load(w, "caption_encoder", eps, dev)?,
                w.tensor::<1>("caption_norm.weight", dev).context("caption_norm")?,
            ))
        } else {
            None
        };
        Ok(Self { cfg, backbone, text_encoder, text_norm, caption, eps, device: dev.clone() })
    }

    pub fn backbone(&self) -> &ModernBert {
        &self.backbone
    }

    pub fn caption_add_bos(&self) -> bool {
        self.cfg.caption_add_bos.unwrap_or(self.cfg.text_add_bos)
    }

    pub fn max_caption_len(&self) -> usize {
        self.cfg.max_caption_len.unwrap_or(self.cfg.max_text_len)
    }

    /// `text_state = text_norm(text_encoder(backbone, ids, mask))`。`[B,S,text_dim]`。
    /// マスクは呼び出し側が持つ(`encode_conditions.out1` はトークナイザのマスクそのもの)。
    pub fn encode_text(&self, ids: &[Vec<i64>], mask: &[Vec<bool>]) -> Tensor<3> {
        let state = self.backbone.forward(ids, mask);
        let proj = self.text_encoder.forward(state, mask_tensor(mask, &self.device));
        rms_norm(proj, &self.text_norm, self.eps)
    }

    /// `caption_state = caption_norm(caption_encoder(backbone, ids, mask))`。`[B,S,caption_dim]`。
    /// キャプション条件が無効なモデルでは None。空キャプションは [`caption_inputs`] でマスク全 false にしておく。
    pub fn encode_caption(&self, ids: &[Vec<i64>], mask: &[Vec<bool>]) -> Option<Tensor<3>> {
        let (proj, norm) = self.caption.as_ref()?;
        let state = self.backbone.forward(ids, mask);
        let p = proj.forward(state, mask_tensor(mask, &self.device));
        Some(rms_norm(p, norm, self.eps))
    }
}

/// 原典 runtime のキャプション入力: `strip()` してトークナイズし、空ならマスクを全て false にする。
pub fn caption_inputs(
    tok: &Tokenizer,
    caption: Option<&str>,
    batch: usize,
    max_len: usize,
    add_bos: bool,
) -> Result<Encoded> {
    let text = caption.unwrap_or("").trim().to_string();
    let texts = vec![text.clone(); batch];
    let (ids, mut mask) = tok.batch_encode(&texts, max_len, add_bos)?;
    if text.is_empty() {
        for row in &mut mask {
            row.fill(false);
        }
    }
    Ok((ids, mask))
}
