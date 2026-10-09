//! ModernBERT-ja-310m(テキスト/キャプション共有のバックボーン)。
//! 原典: `irodori_tts/model.py::PretrainedTextBackbone` と transformers の `modeling_modernbert.py`
//! (`ModernBertModel`、eager 注意)。重みは `pretrained_text_backbone.backbone.*`。
//!
//! 構成は safetensors メタデータ `text_encoder_config_json` から読む。25 層、全注意と窓付き注意が
//! `layer_types` に従って並び、RoPE の theta は種別ごとに違う。マスクは (キー側のパディング) と、
//! 窓付きでは `|i-j| <= local_attention/2`。

use anyhow::{Context, Result, bail};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Device, Tensor, TensorData};
use serde_json::Value;

use crate::weights::Weights;

const PREFIX: &str = "pretrained_text_backbone.backbone";
/// 全てマスクされた行でも NaN にならないよう、-inf ではなく有限の大きな負数を使う。
const MASK_NEG: f32 = -1.0e9;

struct Layer {
    attn_norm: Option<Tensor<1>>,
    /// `[hidden, 3*hidden]`(転置済み)
    wqkv: Tensor<2>,
    wo: Tensor<2>,
    mlp_norm: Tensor<1>,
    /// `[hidden, 2*intermediate]`
    wi: Tensor<2>,
    /// `[intermediate, hidden]`
    mlp_wo: Tensor<2>,
    sliding: bool,
}

pub struct ModernBert {
    hidden: usize,
    heads: usize,
    intermediate: usize,
    eps: f64,
    /// 窓付き注意の半幅(`local_attention / 2`)
    half_window: usize,
    theta_full: f64,
    theta_sliding: f64,
    /// 埋め込み表 `[vocab, hidden]`。GPU に置かない(約 300MB。引く行は数百行だけなので CPU で引いて送る)
    tok_embeddings: Vec<f32>,
    vocab: usize,
    emb_norm: Tensor<1>,
    layers: Vec<Layer>,
    final_norm: Tensor<1>,
    device: Device,
}

/// `x[..., in] @ w[in, out]`(`w` は転置済み)。
pub(crate) fn linear(x: Tensor<3>, w: &Tensor<2>) -> Tensor<3> {
    let [b, s, i] = x.dims();
    let o = w.dims()[1];
    x.reshape([b * s, i]).matmul(w.clone()).reshape([b, s, o])
}

/// バイアスなし LayerNorm(最終軸)。
fn layer_norm(x: Tensor<3>, weight: &Tensor<1>, eps: f64) -> Tensor<3> {
    let mean = x.clone().mean_dim(2);
    let xc = x - mean;
    let var = xc.clone().square().mean_dim(2);
    let n = weight.dims()[0];
    xc / (var + eps).sqrt() * weight.clone().reshape([1, 1, n])
}

fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    let hd = x.dims()[3];
    let x1 = x.clone().narrow(3, 0, hd / 2);
    let x2 = x.narrow(3, hd / 2, hd / 2);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

fn rope_table(seq: usize, head_dim: usize, theta: f64, dev: &Device) -> (Tensor<4>, Tensor<4>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq * head_dim);
    let mut sin = Vec::with_capacity(seq * head_dim);
    for p in 0..seq {
        let mut c = vec![0f32; head_dim];
        let mut s = vec![0f32; head_dim];
        for j in 0..half {
            let inv = 1.0 / theta.powf((2 * j) as f64 / head_dim as f64);
            let a = p as f64 * inv;
            c[j] = a.cos() as f32;
            c[j + half] = c[j];
            s[j] = a.sin() as f32;
            s[j + half] = s[j];
        }
        cos.extend(c);
        sin.extend(s);
    }
    let shape = [1, 1, seq, head_dim];
    (
        Tensor::<4>::from_data(TensorData::new(cos, shape), dev),
        Tensor::<4>::from_data(TensorData::new(sin, shape), dev),
    )
}

impl ModernBert {
    pub fn load(w: &Weights, dev: &Device) -> Result<Self> {
        let cfg: Value = serde_json::from_str(
            w.metadata("text_encoder_config_json").context("text_encoder_config_json metadata missing")?,
        )
        .context("parse text_encoder_config_json")?;
        let get_usize = |k: &str| -> Result<usize> {
            cfg[k].as_u64().map(|v| v as usize).with_context(|| format!("text_encoder_config_json.{k}"))
        };
        let hidden = get_usize("hidden_size")?;
        let intermediate = get_usize("intermediate_size")?;
        let n_layers = get_usize("num_hidden_layers")?;
        let heads = get_usize("num_attention_heads")?;
        let local_attention = get_usize("local_attention")?;
        let eps = cfg["norm_eps"].as_f64().context("norm_eps")?;
        if cfg["hidden_activation"].as_str() != Some("gelu") {
            bail!("unsupported hidden_activation {:?}", cfg["hidden_activation"]);
        }
        if cfg["attention_bias"].as_bool() != Some(false) || cfg["mlp_bias"].as_bool() != Some(false) {
            bail!("biased ModernBERT is not supported");
        }
        let theta = |t: &str| -> Result<f64> {
            cfg["rope_parameters"][t]["rope_theta"].as_f64().with_context(|| format!("rope_theta of {t}"))
        };
        let layer_types: Vec<String> = cfg["layer_types"]
            .as_array()
            .context("layer_types")?
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect();
        if layer_types.len() != n_layers {
            bail!("layer_types has {} entries for {n_layers} layers", layer_types.len());
        }

        let t1 = |name: &str| w.tensor::<1>(&format!("{PREFIX}.{name}"), dev);
        let t2t = |name: &str| -> Result<Tensor<2>> {
            Ok(w.tensor::<2>(&format!("{PREFIX}.{name}"), dev)?.transpose())
        };
        let mut layers = Vec::with_capacity(n_layers);
        for (i, ty) in layer_types.iter().enumerate() {
            layers.push(Layer {
                attn_norm: if i == 0 { None } else { Some(t1(&format!("layers.{i}.attn_norm.weight"))?) },
                wqkv: t2t(&format!("layers.{i}.attn.Wqkv.weight"))?,
                wo: t2t(&format!("layers.{i}.attn.Wo.weight"))?,
                mlp_norm: t1(&format!("layers.{i}.mlp_norm.weight"))?,
                wi: t2t(&format!("layers.{i}.mlp.Wi.weight"))?,
                mlp_wo: t2t(&format!("layers.{i}.mlp.Wo.weight"))?,
                sliding: match ty.as_str() {
                    "sliding_attention" => true,
                    "full_attention" => false,
                    other => bail!("unknown layer type {other}"),
                },
            });
        }
        let (emb_shape, emb) = w.f32_vec(&format!("{PREFIX}.embeddings.tok_embeddings.weight"))?;
        let vocab = emb_shape[0];
        Ok(Self {
            hidden,
            heads,
            intermediate,
            eps,
            half_window: local_attention / 2,
            theta_full: theta("full_attention")?,
            theta_sliding: theta("sliding_attention")?,
            tok_embeddings: emb,
            vocab,
            emb_norm: t1("embeddings.norm.weight")?,
            layers,
            final_norm: t1("final_norm.weight")?,
            device: dev.clone(),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// `[B][S]` の ID とマスクから `last_hidden_state * mask` を返す(`[B, S, hidden]`)。
    pub fn forward(&self, ids: &[Vec<i64>], mask: &[Vec<bool>]) -> Tensor<3> {
        let b = ids.len();
        let s = ids[0].len();
        let dev = &self.device;
        let hd = self.hidden / self.heads;

        let mut rows = Vec::with_capacity(b * s * self.hidden);
        for &id in ids.iter().flatten() {
            let id = (id.max(0) as usize).min(self.vocab - 1);
            rows.extend_from_slice(&self.tok_embeddings[id * self.hidden..(id + 1) * self.hidden]);
        }
        let x = Tensor::<3>::from_data(TensorData::new(rows, [b, s, self.hidden]), dev);
        let mut x = layer_norm(x, &self.emb_norm, self.eps);

        // 加算マスク [B,1,S,S](全注意 / 窓付き)
        let build = |sliding: bool| -> Tensor<4> {
            let mut m = vec![0f32; b * s * s];
            for bi in 0..b {
                for q in 0..s {
                    for k in 0..s {
                        let in_window = !sliding || q.abs_diff(k) <= self.half_window;
                        if !(mask[bi][k] && in_window) {
                            m[(bi * s + q) * s + k] = MASK_NEG;
                        }
                    }
                }
            }
            Tensor::<4>::from_data(TensorData::new(m, [b, 1, s, s]), dev)
        };
        let mask_full = build(false);
        let mask_sliding = build(true);
        let (cos_f, sin_f) = rope_table(s, hd, self.theta_full, dev);
        let (cos_s, sin_s) = rope_table(s, hd, self.theta_sliding, dev);
        let scale = (hd as f64).powf(-0.5);

        for l in &self.layers {
            let (m, cos, sin) =
                if l.sliding { (&mask_sliding, &cos_s, &sin_s) } else { (&mask_full, &cos_f, &sin_f) };
            let h = match &l.attn_norm {
                Some(wn) => layer_norm(x.clone(), wn, self.eps),
                None => x.clone(),
            };
            let qkv = linear(h, &l.wqkv).reshape([b, s, 3, self.heads, hd]);
            let part = |i: usize| -> Tensor<4> {
                qkv.clone().narrow(2, i, 1).reshape([b, s, self.heads, hd]).swap_dims(1, 2)
            };
            let (q, k, v) = (part(0), part(1), part(2));
            let q = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
            let k = k.clone() * cos.clone() + rotate_half(k) * sin.clone();
            let scores = q.matmul(k.swap_dims(2, 3)) * scale + m.clone();
            let attn = softmax(scores, 3).matmul(v); // [B,H,S,hd]
            let attn = attn.swap_dims(1, 2).reshape([b, s, self.hidden]);
            x = x + linear(attn, &l.wo);

            let h = layer_norm(x.clone(), &l.mlp_norm, self.eps);
            let wi = linear(h, &l.wi);
            let input = wi.clone().narrow(2, 0, self.intermediate);
            let gate = wi.narrow(2, self.intermediate, self.intermediate);
            x = x + linear(gelu(input) * gate, &l.mlp_wo);
        }
        let x = layer_norm(x, &self.final_norm, self.eps);

        let mflat: Vec<f32> = mask.iter().flatten().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        let mt = Tensor::<3>::from_data(TensorData::new(mflat, [b, s, 1]), dev);
        x * mt
    }
}
