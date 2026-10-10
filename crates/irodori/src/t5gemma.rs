//! T5Gemma 2 のテキストエンコーダ(Irodori-TTS v4 Large のテキスト/キャプション共有のバックボーン)。
//! 原典: `irodori_tts/model.py::PretrainedTextBackbone` と transformers の `modeling_t5gemma2.py`
//! (`T5Gemma2TextEncoder`)。重みは `pretrained_text_backbone.backbone.*`(デコーダと画像側は含まない)。
//!
//! Gemma 3 系の双方向エンコーダ: 埋め込み × √hidden(EOI トークンだけは専用の埋め込み)、各層は
//! 前後に RMSNorm(`1 + w` を掛ける)を挟んだ自己注意と GeGLU(tanh 近似の GELU)。注意は GQA で、
//! q / k にも RMSNorm、RoPE は層の種別(全注意 / 窓付き)ごとに theta と倍率が違う。窓付きは
//! キーが `q - (W+1)/2 < k < q + W/2 + 1` の範囲(原典 `sliding_window_mask_function(is_causal=False)`)。

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Tensor, TensorData};
use serde_json::Value;

use crate::nn::Linear;
use crate::weights::Weights;

const PREFIX: &str = "pretrained_text_backbone.backbone";
/// 無効なキーに足す値(全てマスクされた行でも NaN にならない有限値)
const MASK_NEG: f32 = -1.0e9;

/// Gemma の RMSNorm: `x / sqrt(mean(x²) + eps) * (1 + w)`
struct RmsNorm {
    /// `1 + w`
    scale: Tensor<1>,
    eps: f64,
}

impl RmsNorm {
    fn load(w: &Weights, name: &str, eps: f64, dev: &Device) -> Result<Self> {
        Ok(Self { scale: w.tensor::<1>(name, dev)?.add_scalar(1.0), eps })
    }

    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let n = self.scale.dims()[0];
        let mut shape = [1usize; D];
        shape[D - 1] = n;
        let rms = x.clone().square().mean_dim(D - 1).add_scalar(self.eps).sqrt();
        x.div(rms).mul(self.scale.clone().reshape(shape))
    }
}

struct Layer {
    pre_attn: RmsNorm,
    post_attn: RmsNorm,
    pre_ff: RmsNorm,
    post_ff: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    sliding: bool,
}

pub struct T5Gemma2Encoder {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// 注意の倍率(`query_pre_attn_scalar ** -0.5`)
    scaling: f64,
    sliding_window: usize,
    /// (theta, 位置の倍率)。全注意は linear スケーリング(位置を factor で割る)
    rope_full: (f64, f64),
    rope_sliding: (f64, f64),
    /// 埋め込み表 `[vocab, hidden]`。GPU に置かない(約 1.2GB。引く行は数百行だけなので CPU で引いて送る)
    tok_embeddings: Vec<f32>,
    eoi_embedding: Vec<f32>,
    eoi_token: i64,
    vocab: usize,
    layers: Vec<Layer>,
    norm: RmsNorm,
    device: Device,
}

/// `gelu_pytorch_tanh`(tanh 近似の GELU)
fn gelu_tanh(x: Tensor<3>) -> Tensor<3> {
    let c = (2.0f64 / std::f64::consts::PI).sqrt();
    let inner = x.clone().add(x.clone().powi_scalar(3).mul_scalar(0.044715)).mul_scalar(c);
    x.mul(inner.tanh().add_scalar(1.0)).mul_scalar(0.5)
}

fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    let hd = x.dims()[3];
    let x1 = x.clone().narrow(3, 0, hd / 2);
    let x2 = x.narrow(3, hd / 2, hd / 2);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

/// RoPE の cos / sin `[1, 1, seq, head_dim]`(`emb = cat(freqs, freqs)`)
fn rope_table(seq: usize, head_dim: usize, (theta, factor): (f64, f64), dev: &Device) -> (Tensor<4>, Tensor<4>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq * head_dim);
    let mut sin = Vec::with_capacity(seq * head_dim);
    for p in 0..seq {
        let mut c = vec![0f32; head_dim];
        let mut s = vec![0f32; head_dim];
        for j in 0..half {
            let inv = 1.0 / theta.powf((2 * j) as f64 / head_dim as f64) / factor;
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
    (Tensor::<4>::from_data(TensorData::new(cos, shape), dev), Tensor::<4>::from_data(TensorData::new(sin, shape), dev))
}

impl T5Gemma2Encoder {
    /// `text_encoder_config_json` が T5Gemma 2 のチェックポイントか
    pub fn is_t5gemma2(w: &Weights) -> bool {
        w.metadata("text_encoder_config_json")
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            .is_some_and(|v| v["model_type"].as_str() == Some("t5gemma2"))
    }

    pub fn load(w: &Weights, dev: &Device) -> Result<Self> {
        let root: Value = serde_json::from_str(w.metadata("text_encoder_config_json").context("text_encoder_config_json metadata missing")?)
            .context("parse text_encoder_config_json")?;
        let cfg = &root["encoder"]["text_config"];
        let get = |k: &str| -> Result<usize> { cfg[k].as_u64().map(|v| v as usize).with_context(|| format!("text_config.{k}")) };
        let hidden = get("hidden_size")?;
        let heads = get("num_attention_heads")?;
        let kv_heads = get("num_key_value_heads")?;
        let head_dim = get("head_dim")?;
        let n_layers = get("num_hidden_layers")?;
        let sliding_window = get("sliding_window")?;
        let eps = cfg["rms_norm_eps"].as_f64().context("rms_norm_eps")?;
        let scaling = cfg["query_pre_attn_scalar"].as_f64().context("query_pre_attn_scalar")?.powf(-0.5);
        ensure!(heads.is_multiple_of(kv_heads), "num_attention_heads must be a multiple of num_key_value_heads");
        ensure!(cfg["hidden_activation"].as_str() == Some("gelu_pytorch_tanh"), "unsupported hidden_activation {:?}", cfg["hidden_activation"]);
        ensure!(cfg["attention_bias"].as_bool() == Some(false), "biased attention is not supported");
        ensure!(cfg["attn_logit_softcapping"].is_null(), "attention logit softcapping is not supported");
        let rope = |t: &str| -> Result<(f64, f64)> {
            let p = &cfg["rope_parameters"][t];
            let theta = p["rope_theta"].as_f64().with_context(|| format!("rope_theta of {t}"))?;
            match p["rope_type"].as_str() {
                Some("default") => Ok((theta, 1.0)),
                Some("linear") => Ok((theta, p["factor"].as_f64().with_context(|| format!("factor of {t}"))?)),
                other => bail!("unsupported rope_type {other:?} for {t}"),
            }
        };
        let eoi_token = root["eoi_token_index"].as_i64().context("eoi_token_index")?;
        let layer_types: Vec<&str> = cfg["layer_types"].as_array().context("layer_types")?.iter().map(|v| v.as_str().unwrap_or_default()).collect();
        ensure!(layer_types.len() == n_layers, "layer_types has {} entries for {n_layers} layers", layer_types.len());

        let norm = |name: &str| RmsNorm::load(w, &format!("{PREFIX}.{name}"), eps, dev);
        let lin = |name: &str| Linear::weight(w, &format!("{PREFIX}.{name}.weight"), dev);
        let mut layers = Vec::with_capacity(n_layers);
        for (i, ty) in layer_types.iter().enumerate() {
            let p = format!("layers.{i}");
            layers.push(Layer {
                pre_attn: norm(&format!("{p}.pre_self_attn_layernorm.weight"))?,
                post_attn: norm(&format!("{p}.post_self_attn_layernorm.weight"))?,
                pre_ff: norm(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                post_ff: norm(&format!("{p}.post_feedforward_layernorm.weight"))?,
                q: lin(&format!("{p}.self_attn.q_proj"))?,
                k: lin(&format!("{p}.self_attn.k_proj"))?,
                v: lin(&format!("{p}.self_attn.v_proj"))?,
                o: lin(&format!("{p}.self_attn.o_proj"))?,
                q_norm: norm(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: norm(&format!("{p}.self_attn.k_norm.weight"))?,
                gate: lin(&format!("{p}.mlp.gate_proj"))?,
                up: lin(&format!("{p}.mlp.up_proj"))?,
                down: lin(&format!("{p}.mlp.down_proj"))?,
                sliding: match *ty {
                    "sliding_attention" => true,
                    "full_attention" => false,
                    other => bail!("unknown layer type {other}"),
                },
            });
        }
        let (emb_shape, tok_embeddings) = w.f32_vec(&format!("{PREFIX}.embed_tokens.weight"))?;
        ensure!(emb_shape == [emb_shape[0], hidden], "embed_tokens shape {emb_shape:?}");
        let (_, eoi_embedding) = w.f32_vec(&format!("{PREFIX}.embed_tokens.eoi_embedding"))?;
        Ok(Self {
            hidden,
            heads,
            kv_heads,
            head_dim,
            scaling,
            sliding_window,
            rope_full: rope("full_attention")?,
            rope_sliding: rope("sliding_attention")?,
            vocab: emb_shape[0],
            tok_embeddings,
            eoi_embedding,
            eoi_token,
            layers,
            norm: norm("norm.weight")?,
            device: dev.clone(),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// `[B][S]` の ID とマスクから `last_hidden_state * mask` を返す(`[B, S, hidden]`)。
    pub fn forward(&self, ids: &[Vec<i64>], mask: &[Vec<bool>]) -> Tensor<3> {
        let (b, s) = (ids.len(), ids[0].len());
        let dev = &self.device;
        let (hd, h) = (self.head_dim, self.hidden);

        // 埋め込み × √hidden(原典は f32 の倍率を掛ける)。EOI は倍率なしの専用埋め込み
        let scale = (h as f64).sqrt() as f32;
        let mut rows = Vec::with_capacity(b * s * h);
        for &id in ids.iter().flatten() {
            if id == self.eoi_token {
                rows.extend_from_slice(&self.eoi_embedding);
            } else {
                let id = (id.max(0) as usize).min(self.vocab - 1);
                rows.extend(self.tok_embeddings[id * h..(id + 1) * h].iter().map(|v| v * scale));
            }
        }
        let mut x = Tensor::<3>::from_data(TensorData::new(rows, [b, s, h]), dev);

        // 加算マスク [B,1,S,S](キー側のパディング。窓付きは範囲外のキーも)
        let (left, right) = (self.sliding_window.div_ceil(2), self.sliding_window / 2 + 1);
        let build = |sliding: bool| -> Tensor<4> {
            let mut m = vec![0f32; b * s * s];
            for (bi, row) in mask.iter().enumerate() {
                for q in 0..s {
                    for (k, &valid) in row.iter().enumerate() {
                        let in_window = !sliding || if q >= k { q - k < left } else { k - q < right };
                        if !(valid && in_window) {
                            m[(bi * s + q) * s + k] = MASK_NEG;
                        }
                    }
                }
            }
            Tensor::<4>::from_data(TensorData::new(m, [b, 1, s, s]), dev)
        };
        let (mask_full, mask_sliding) = (build(false), build(true));
        let (cos_f, sin_f) = rope_table(s, hd, self.rope_full, dev);
        let (cos_s, sin_s) = rope_table(s, hd, self.rope_sliding, dev);
        let rep = self.heads / self.kv_heads;

        for l in &self.layers {
            let (m, cos, sin) = if l.sliding { (&mask_sliding, &cos_s, &sin_s) } else { (&mask_full, &cos_f, &sin_f) };
            let residual = x.clone();
            let hs = l.pre_attn.forward(x);
            let q = l.q_norm.forward(l.q.forward3(hs.clone()).reshape([b, s, self.heads, hd])).swap_dims(1, 2);
            let k = l.k_norm.forward(l.k.forward3(hs.clone()).reshape([b, s, self.kv_heads, hd]));
            let v = l.v.forward3(hs).reshape([b, s, self.kv_heads, hd]);
            // GQA: キー・値の各ヘッドを rep 回くり返す(ヘッド h はキー・値のヘッド h / rep)
            let expand = |t: Tensor<4>| -> Tensor<4> {
                t.reshape([b, s, self.kv_heads, 1, hd]).expand([b, s, self.kv_heads, rep, hd]).reshape([b, s, self.heads, hd]).swap_dims(1, 2)
            };
            let (k, v) = (expand(k), expand(v));
            let q = q.clone().mul(cos.clone()).add(rotate_half(q).mul(sin.clone()));
            let k = k.clone().mul(cos.clone()).add(rotate_half(k).mul(sin.clone()));
            let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scaling).add(m.clone());
            let attn = softmax(scores, 3).matmul(v).swap_dims(1, 2).reshape([b, s, self.heads * hd]);
            x = residual.add(l.post_attn.forward(l.o.forward3(attn)));

            let residual = x.clone();
            let hs = l.pre_ff.forward(x);
            let ff = l.down.forward3(gelu_tanh(l.gate.forward3(hs.clone())).mul(l.up.forward3(hs)));
            x = residual.add(l.post_ff.forward(ff));
        }
        let x = self.norm.forward(x);
        let mflat: Vec<f32> = mask.iter().flatten().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        x.mul(Tensor::<3>::from_data(TensorData::new(mflat, [b, s, 1]), dev))
    }
}
