//! MeanFlow DiT(`TextToLatentRFDiT` の推論経路)と話者エンコーダ(`ReferenceLatentEncoder`)。
//!
//! 原典: `irodori_tts/model.py`。重みはチェックポイントの名前そのまま(`blocks.*` など)。
//! テキスト/キャプションの状態(ModernBERT + projector + norm)は別モジュールが作り、ここへ渡す。
//!
//! 規約:
//! - マスクは f32 の `Tensor<2>`(1.0 = 有効、0.0 = パディング)。形は `[batch, seq]`。
//! - 注意のマスクは加算バイアス(無効キー = -1e9)で適用し、ヘッド・バッチ方向へは `expand` で共有する。
//! - 線形層は `[out, in]` の重みをロード時に 1 回転置して保持する。

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, Tensor, TensorData};

use crate::config::ModelConfig;
use crate::weights::Weights;

/// 無効キーに足すバイアス(softmax 後に厳密に 0 になる)
const MASK_NEG: f64 = -1.0e9;

// ---------------------------------------------------------------------------------------------
// 基本部品
// ---------------------------------------------------------------------------------------------

struct Linear {
    /// `[in, out]`(ロード時に転置済み)
    wt: Tensor<2>,
    b: Option<Tensor<1>>,
}

impl Linear {
    fn load(w: &Weights, prefix: &str, bias: bool, dev: &Device) -> Result<Self> {
        let wt = w.tensor::<2>(&format!("{prefix}.weight"), dev)?.transpose();
        let b = if bias { Some(w.tensor::<1>(&format!("{prefix}.bias"), dev)?) } else { None };
        Ok(Self { wt, b })
    }

    fn forward2(&self, x: Tensor<2>) -> Tensor<2> {
        let y = x.matmul(self.wt.clone());
        match &self.b {
            Some(b) => {
                let n = b.dims()[0];
                y.add(b.clone().reshape([1, n]))
            }
            None => y,
        }
    }

    fn forward3(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, s, i] = x.dims();
        let o = self.wt.dims()[1];
        self.forward2(x.reshape([b * s, i])).reshape([b, s, o])
    }
}

/// RMSNorm(重みは放送できる形 `[1, .., 1, dims..]` で保持)
struct RmsNorm<const D: usize> {
    w: Tensor<D>,
    eps: f64,
}

fn load_flat(w: &Weights, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let (s, data) = w.f32_vec(name)?;
    let n: usize = s.iter().product();
    ensure!(n == shape.iter().product::<usize>(), "{name}: shape {s:?} does not match {shape:?}");
    Ok(data)
}

impl<const D: usize> RmsNorm<D> {
    fn from_weights(w: &Weights, name: &str, shape: [usize; D], eps: f64, dev: &Device) -> Result<Self> {
        let data = load_flat(w, name, &shape)?;
        Ok(Self { w: Tensor::<D>::from_data(TensorData::new(data, shape.to_vec()), dev), eps })
    }

    fn forward(&self, x: Tensor<D>) -> Tensor<D> {
        let ms = x.clone().square().mean_dim(D - 1);
        let inv = ms.add_scalar(self.eps).sqrt().recip();
        x.mul(inv).mul(self.w.clone())
    }
}

/// RoPE のテーブル(複素数 `x_{2i} + i x_{2i+1}` に `e^{i pos θ_i}` を掛ける原典の方式)
struct Rope {
    /// `[1, S, 1, Dh/2, 1]`
    cos: Tensor<5>,
    sin: Tensor<5>,
}

impl Rope {
    fn new(head_dim: usize, seq_len: usize, dev: &Device) -> Self {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(seq_len * half);
        let mut sin = Vec::with_capacity(seq_len * half);
        let freqs: Vec<f32> =
            (0..half).map(|i| 1.0f32 / 10000f32.powf((2 * i) as f32 / head_dim as f32)).collect();
        for p in 0..seq_len {
            for f in &freqs {
                let a = (p as f32 * f) as f64;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        let shape = vec![1, seq_len, 1, half, 1];
        Self {
            cos: Tensor::<5>::from_data(TensorData::new(cos, shape.clone()), dev),
            sin: Tensor::<5>::from_data(TensorData::new(sin, shape), dev),
        }
    }

    /// `x`: `[B, S, H, Dh]`
    fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, s, h, d] = x.dims();
        let x5 = x.reshape([b, s, h, d / 2, 2]);
        let a = x5.clone().narrow(4, 0, 1);
        let c = x5.narrow(4, 1, 1);
        let ra = a.clone().mul(self.cos.clone()).sub(c.clone().mul(self.sin.clone()));
        let rb = a.mul(self.sin.clone()).add(c.mul(self.cos.clone()));
        Tensor::cat(vec![ra, rb], 4).reshape([b, s, h, d])
    }

    /// 前半のヘッドだけ回し、後半はそのまま(JointAttention の `_apply_rotary_half`)
    fn apply_half(&self, x: Tensor<4>) -> Tensor<4> {
        let h = x.dims()[2];
        let half = h / 2;
        let rot = self.apply(x.clone().narrow(2, 0, half));
        let pass = x.narrow(2, half, h - half);
        Tensor::cat(vec![rot, pass], 2)
    }
}

/// `[B, Sk]` の有効マスク → 加算バイアス `[B, H, Sq, Sk]`(放送のみ。実体化しない)
fn mask_to_bias(mask: Tensor<2>, heads: usize, sq: usize) -> Tensor<4> {
    let [b, sk] = mask.dims();
    mask.sub_scalar(1.0).mul_scalar(-MASK_NEG).reshape([b, 1, 1, sk]).expand([b, heads, sq, sk])
}

/// `[B, H, S, Dh]` → `[B, S, H*Dh]`
fn from_heads(x: Tensor<4>) -> Tensor<3> {
    let [b, h, s, d] = x.dims();
    x.swap_dims(1, 2).reshape([b, s, h * d])
}

fn sdpa(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, bias: Tensor<4>) -> Tensor<4> {
    attention(q, k, v, None, Some(bias), AttentionModuleOptions::default())
}

struct SwiGlu {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGlu {
    fn load(w: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            w1: Linear::load(w, &format!("{prefix}.w1"), false, dev)?,
            w2: Linear::load(w, &format!("{prefix}.w2"), false, dev)?,
            w3: Linear::load(w, &format!("{prefix}.w3"), false, dev)?,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let a = silu(self.w1.forward3(x.clone()));
        self.w2.forward3(a.mul(self.w3.forward3(x)))
    }
}

// ---------------------------------------------------------------------------------------------
// 話者エンコーダ(ReferenceLatentEncoder)
// ---------------------------------------------------------------------------------------------

struct SelfAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    gate: Linear,
    q_norm: RmsNorm<4>,
    k_norm: RmsNorm<4>,
    heads: usize,
}

impl SelfAttention {
    fn load(w: &Weights, prefix: &str, heads: usize, head_dim: usize, eps: f64, dev: &Device) -> Result<Self> {
        let lin = |n: &str| Linear::load(w, &format!("{prefix}.{n}"), false, dev);
        Ok(Self {
            wq: lin("wq")?,
            wk: lin("wk")?,
            wv: lin("wv")?,
            wo: lin("wo")?,
            gate: lin("gate")?,
            q_norm: RmsNorm::from_weights(w, &format!("{prefix}.q_norm.weight"), [1, 1, heads, head_dim], eps, dev)?,
            k_norm: RmsNorm::from_weights(w, &format!("{prefix}.k_norm.weight"), [1, 1, heads, head_dim], eps, dev)?,
            heads,
        })
    }

    fn forward(&self, x: Tensor<3>, bias: Tensor<4>, rope: &Rope) -> Tensor<3> {
        let [b, s, d] = x.dims();
        let hd = d / self.heads;
        let split = |t: Tensor<3>| t.reshape([b, s, self.heads, hd]);
        let q = rope.apply(self.q_norm.forward(split(self.wq.forward3(x.clone()))));
        let k = rope.apply(self.k_norm.forward(split(self.wk.forward3(x.clone()))));
        let v = split(self.wv.forward3(x.clone()));
        let y = from_heads(sdpa(q.swap_dims(1, 2), k.swap_dims(1, 2), v.swap_dims(1, 2), bias));
        self.wo.forward3(y.mul(sigmoid(self.gate.forward3(x))))
    }
}

struct TextBlock {
    attention_norm: RmsNorm<3>,
    attention: SelfAttention,
    mlp_norm: RmsNorm<3>,
    mlp: SwiGlu,
}

struct SpeakerEncoder {
    in_proj: Linear,
    blocks: Vec<TextBlock>,
    heads: usize,
    head_dim: usize,
    patch: usize,
}

impl SpeakerEncoder {
    fn load(w: &Weights, cfg: &ModelConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.speaker_dim.context("speaker_dim")?;
        let heads = cfg.speaker_heads.context("speaker_heads")?;
        let layers = cfg.speaker_layers.context("speaker_layers")?;
        let head_dim = dim / heads;
        let eps = cfg.norm_eps;
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("speaker_encoder.blocks.{i}");
            blocks.push(TextBlock {
                attention_norm: RmsNorm::from_weights(w, &format!("{p}.attention_norm.weight"), [1, 1, dim], eps, dev)?,
                attention: SelfAttention::load(w, &format!("{p}.attention"), heads, head_dim, eps, dev)?,
                mlp_norm: RmsNorm::from_weights(w, &format!("{p}.mlp_norm.weight"), [1, 1, dim], eps, dev)?,
                mlp: SwiGlu::load(w, &format!("{p}.mlp"), dev)?,
            });
        }
        Ok(Self {
            in_proj: Linear::load(w, "speaker_encoder.in_proj", true, dev)?,
            blocks,
            heads,
            head_dim,
            patch: cfg.speaker_patch_size.context("speaker_patch_size")?,
        })
    }

    /// `latent`: `[B, T, Dp]`(パッチ化後の次元でさらに系列方向に `speaker_patch_size` でまとめた後)、
    /// `mask`: `[B, T]`
    fn forward(&self, latent: Tensor<3>, mask: Tensor<2>) -> Tensor<3> {
        let [b, s, _] = latent.dims();
        let mask3 = mask.clone().reshape([b, s, 1]);
        let mut x = self.in_proj.forward3(latent).div_scalar(6.0).mul(mask3.clone());
        let rope = Rope::new(self.head_dim, s, &x.device());
        let bias = prefix_mask_bias(mask, self.heads, s);
        for blk in &self.blocks {
            let a = blk.attention.forward(blk.attention_norm.forward(x.clone()), bias.clone(), &rope);
            x = x.add(a);
            x = x.clone().add(blk.mlp.forward(blk.mlp_norm.forward(x)));
            x = x.mul(mask3.clone());
        }
        x
    }
}

/// 自己注意用のキーマスク。全無効の行は先頭キーを有効にする(原典の `safe_mask` と同じ)
fn prefix_mask_bias(mask: Tensor<2>, heads: usize, sq: usize) -> Tensor<4> {
    let [b, s] = mask.dims();
    let dev = mask.device();
    let any = mask.clone().max_dim(1); // [B,1]
    let none = any.neg().add_scalar(1.0);
    let mut e0 = vec![0f32; s];
    e0[0] = 1.0;
    let e0 = Tensor::<2>::from_data(TensorData::new(e0, vec![1, s]), &dev);
    let safe = mask.add(e0.mul(none));
    debug_assert_eq!(safe.dims(), [b, s]);
    mask_to_bias(safe, heads, sq)
}

// ---------------------------------------------------------------------------------------------
// DiT 本体
// ---------------------------------------------------------------------------------------------

struct LowRankAdaLn {
    down: [Linear; 3], // shift, scale, gate
    up: [Linear; 3],
    eps: f64,
}

impl LowRankAdaLn {
    fn load(w: &Weights, prefix: &str, eps: f64, dev: &Device) -> Result<Self> {
        let l = |n: &str, bias| Linear::load(w, &format!("{prefix}.{n}"), bias, dev);
        Ok(Self {
            down: [l("shift_down", false)?, l("scale_down", false)?, l("gate_down", false)?],
            up: [l("shift_up", true)?, l("scale_up", true)?, l("gate_up", true)?],
            eps,
        })
    }

    /// `cond`: `[B, 3*dim]`。x に依らない部分(shift, scale, tanh(gate))を `[B,1,dim]` で返す。
    fn modulation(&self, cond: &Tensor<2>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let [b, n] = cond.dims();
        let d = n / 3;
        let parts = cond.clone().chunk(3, 1);
        let mut out = parts.into_iter().enumerate().map(|(i, p)| {
            let y = self.up[i].forward2(self.down[i].forward2(silu(p.clone()))).add(p);
            y.reshape([b, 1, d])
        });
        let shift = out.next().unwrap();
        let scale = out.next().unwrap();
        let gate = out.next().unwrap().tanh();
        (shift, scale, gate)
    }

    fn norm_modulate(&self, x: Tensor<3>, shift: Tensor<3>, scale: Tensor<3>) -> Tensor<3> {
        let inv = x.clone().square().mean_dim(2).add_scalar(self.eps).sqrt().recip();
        x.mul(inv).mul(scale.add_scalar(1.0)).add(shift)
    }
}

/// 層ごとの条件 K/V(text → speaker → caption の順に系列方向へ連結済み)。`[B, H, Sctx, Dh]`
pub struct LayerKv {
    pub k: Tensor<4>,
    pub v: Tensor<4>,
}

/// `build_context_kv_cache` の結果(条件が固定の間、全ステップで使い回す)
pub struct ContextKv {
    pub layers: Vec<LayerKv>,
}

/// エンコード済み条件(`encode_conditions` の出力に相当)。マスクは f32(1.0 / 0.0)。
pub struct Conditions {
    pub text_state: Tensor<3>,
    pub text_mask: Tensor<2>,
    pub speaker_state: Option<Tensor<3>>,
    pub speaker_mask: Option<Tensor<2>>,
    pub caption_state: Option<Tensor<3>>,
    pub caption_mask: Option<Tensor<2>>,
}

impl Conditions {
    pub fn batch(&self) -> usize {
        self.text_state.dims()[0]
    }

    /// K/V の連結順(text, speaker, caption)と同じ順のキーマスク `[B, Sctx]`
    fn context_mask(&self) -> Tensor<2> {
        let mut parts = vec![self.text_mask.clone()];
        if let (Some(_), Some(m)) = (&self.speaker_state, &self.speaker_mask) {
            parts.push(m.clone());
        }
        if let (Some(_), Some(m)) = (&self.caption_state, &self.caption_mask) {
            parts.push(m.clone());
        }
        Tensor::cat(parts, 1)
    }
}

struct JointAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wk_text: Linear,
    wv_text: Linear,
    wk_speaker: Option<Linear>,
    wv_speaker: Option<Linear>,
    wk_caption: Option<Linear>,
    wv_caption: Option<Linear>,
    gate: Linear,
    wo: Linear,
    q_norm: RmsNorm<4>,
    k_norm: RmsNorm<4>,
    heads: usize,
}

impl JointAttention {
    fn load(w: &Weights, prefix: &str, cfg: &ModelConfig, dev: &Device) -> Result<Self> {
        let heads = cfg.num_heads;
        let hd = cfg.model_dim / heads;
        let eps = cfg.norm_eps;
        let lin = |n: &str| Linear::load(w, &format!("{prefix}.{n}"), false, dev);
        let opt = |n: &str| -> Result<Option<Linear>> {
            if w.contains(&format!("{prefix}.{n}.weight")) { Ok(Some(lin(n)?)) } else { Ok(None) }
        };
        Ok(Self {
            wq: lin("wq")?,
            wk: lin("wk")?,
            wv: lin("wv")?,
            wk_text: lin("wk_text")?,
            wv_text: lin("wv_text")?,
            wk_speaker: opt("wk_speaker")?,
            wv_speaker: opt("wv_speaker")?,
            wk_caption: opt("wk_caption")?,
            wv_caption: opt("wv_caption")?,
            gate: lin("gate")?,
            wo: lin("wo")?,
            q_norm: RmsNorm::from_weights(w, &format!("{prefix}.q_norm.weight"), [1, 1, heads, hd], eps, dev)?,
            k_norm: RmsNorm::from_weights(w, &format!("{prefix}.k_norm.weight"), [1, 1, heads, hd], eps, dev)?,
            heads,
        })
    }

    /// 条件側の K(RMSNorm 済み)と V を `[B,H,S,Dh]` で返し、連結する
    fn project_context(&self, c: &Conditions) -> Result<LayerKv> {
        let proj = |lk: &Linear, lv: &Linear, s: &Tensor<3>| {
            let [b, n, _] = s.dims();
            let hd = lk.wt.dims()[1] / self.heads;
            let k = self.k_norm.forward(lk.forward3(s.clone()).reshape([b, n, self.heads, hd]));
            let v = lv.forward3(s.clone()).reshape([b, n, self.heads, hd]);
            (k.swap_dims(1, 2), v.swap_dims(1, 2))
        };
        let (k, v) = proj(&self.wk_text, &self.wv_text, &c.text_state);
        let (mut ks, mut vs) = (vec![k], vec![v]);
        if let (Some(lk), Some(lv)) = (&self.wk_speaker, &self.wv_speaker) {
            let s = c.speaker_state.as_ref().context("speaker_state is required (model has speaker conditioning)")?;
            let (k, v) = proj(lk, lv, s);
            ks.push(k);
            vs.push(v);
        }
        if let (Some(lk), Some(lv)) = (&self.wk_caption, &self.wv_caption) {
            let s = c.caption_state.as_ref().context("caption_state is required (model has caption conditioning)")?;
            let (k, v) = proj(lk, lv, s);
            ks.push(k);
            vs.push(v);
        }
        Ok(LayerKv { k: Tensor::cat(ks, 2), v: Tensor::cat(vs, 2) })
    }

    fn forward(&self, x: Tensor<3>, kv: &LayerKv, bias: Tensor<4>, rope: &Rope) -> Tensor<3> {
        let [b, s, d] = x.dims();
        let hd = d / self.heads;
        let split = |t: Tensor<3>| t.reshape([b, s, self.heads, hd]);
        let q = rope.apply_half(self.q_norm.forward(split(self.wq.forward3(x.clone()))));
        let ks = rope.apply_half(self.k_norm.forward(split(self.wk.forward3(x.clone()))));
        let vs = split(self.wv.forward3(x.clone()));
        let k = Tensor::cat(vec![ks.swap_dims(1, 2), kv.k.clone()], 2);
        let v = Tensor::cat(vec![vs.swap_dims(1, 2), kv.v.clone()], 2);
        let y = from_heads(sdpa(q.swap_dims(1, 2), k, v, bias));
        self.wo.forward3(y.mul(sigmoid(self.gate.forward3(x))))
    }
}

struct DiffusionBlock {
    attention: JointAttention,
    mlp: SwiGlu,
    attention_adaln: LowRankAdaLn,
    mlp_adaln: LowRankAdaLn,
}

struct CondMlp {
    l0: Linear,
    l1: Linear,
    l2: Linear,
}

impl CondMlp {
    fn load(w: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            l0: Linear::load(w, &format!("{prefix}.0"), false, dev)?,
            l1: Linear::load(w, &format!("{prefix}.2"), false, dev)?,
            l2: Linear::load(w, &format!("{prefix}.4"), false, dev)?,
        })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = silu(self.l0.forward2(x));
        let x = silu(self.l1.forward2(x));
        self.l2.forward2(x)
    }
}

pub struct Dit {
    pub cfg: ModelConfig,
    device: Device,
    blocks: Vec<DiffusionBlock>,
    cond_module: CondMlp,
    delta_cond_module: Option<CondMlp>,
    in_proj: Linear,
    out_norm: RmsNorm<3>,
    out_proj: Linear,
    speaker_encoder: Option<SpeakerEncoder>,
    speaker_norm: Option<RmsNorm<3>>,
}

/// タイムステップ埋め込み(原典 `get_timestep_embedding`)。`[B] → [B, dim]`
fn timestep_embedding(t: &[f32], dim: usize, dev: &Device) -> Tensor<2> {
    let half = dim / 2;
    let freqs: Vec<f32> =
        (0..half).map(|i| 1000.0f32 * (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
    let mut data = Vec::with_capacity(t.len() * dim);
    for &tv in t {
        let args: Vec<f32> = freqs.iter().map(|f| tv * f).collect();
        data.extend(args.iter().map(|&a| (a as f64).cos() as f32));
        data.extend(args.iter().map(|&a| (a as f64).sin() as f32));
    }
    Tensor::<2>::from_data(TensorData::new(data, vec![t.len(), dim]), dev)
}

fn host_vec(t: Tensor<1>) -> Result<Vec<f32>> {
    t.into_data().convert::<f32>().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))
}

impl Dit {
    pub fn load(w: &Weights, cfg: &ModelConfig, device: &Device) -> Result<Self> {
        ensure!(cfg.is_meanflow(), "only MeanFlow checkpoints are supported");
        ensure!(cfg.model_dim % cfg.num_heads == 0 && (cfg.model_dim / cfg.num_heads) % 2 == 0, "bad head dim");
        ensure!((cfg.num_heads % 2) == 0, "num_heads must be even for half-head RoPE");
        let eps = cfg.norm_eps;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("blocks.{i}");
            blocks.push(DiffusionBlock {
                attention: JointAttention::load(w, &format!("{p}.attention"), cfg, device)?,
                mlp: SwiGlu::load(w, &format!("{p}.mlp"), device)?,
                attention_adaln: LowRankAdaLn::load(w, &format!("{p}.attention_adaln"), eps, device)?,
                mlp_adaln: LowRankAdaLn::load(w, &format!("{p}.mlp_adaln"), eps, device)?,
            });
        }
        let (speaker_encoder, speaker_norm) = if cfg.use_speaker_condition {
            let dim = cfg.speaker_dim.context("speaker_dim")?;
            (
                Some(SpeakerEncoder::load(w, cfg, device)?),
                Some(RmsNorm::from_weights(w, "speaker_norm.weight", [1, 1, dim], eps, device)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            cfg: cfg.clone(),
            device: device.clone(),
            blocks,
            cond_module: CondMlp::load(w, "cond_module", device)?,
            delta_cond_module: Some(CondMlp::load(w, "delta_cond_module", device)?),
            in_proj: Linear::load(w, "in_proj", true, device)?,
            out_norm: RmsNorm::from_weights(w, "out_norm.weight", [1, 1, cfg.model_dim], eps, device)?,
            out_proj: Linear::load(w, "out_proj", true, device)?,
            speaker_encoder,
            speaker_norm,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn has_speaker_condition(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// `encode_conditions` の話者側。`reference` は (パッチ化済み潜在 `[B,T,latent_dim*latent_patch]`、
    /// 有効マスク `[B,T]`)。`None` は参照なし(`no_ref`)で、原典と同じく全無効の `[B, 2, speaker_dim]`
    /// を返す(平均トークン + `speaker_patch_size` でまとめた 1 トークン、どちらも無効)。
    pub fn encode_speaker(
        &self,
        reference: Option<(Tensor<3>, Tensor<2>)>,
        batch: usize,
    ) -> Result<(Tensor<3>, Tensor<2>)> {
        let enc = self.speaker_encoder.as_ref().context("model has no speaker conditioning")?;
        let norm = self.speaker_norm.as_ref().unwrap();
        let dim = self.cfg.speaker_dim.unwrap();
        let Some((latent, mask)) = reference else {
            // ゼロ潜在 + 全無効マスクを通すと全層で 0 になる(原典で確認済み)ので計算を省く
            let st = Tensor::<3>::zeros([batch, 2, dim], &self.device);
            let mk = Tensor::<2>::zeros([batch, 2], &self.device);
            return Ok((st, mk));
        };
        let [b, t, d] = latent.dims();
        ensure!(mask.dims() == [b, t], "ref mask shape {:?} != {:?}", mask.dims(), [b, t]);
        let p = enc.patch;
        let usable = (t / p) * p;
        if usable == 0 {
            bail!("reference too short for speaker_patch_size={p}: {t} frames");
        }
        let (latent, mask) = if p > 1 {
            let l = latent.narrow(1, 0, usable).reshape([b, usable / p, d * p]);
            let m = mask.narrow(1, 0, usable).reshape([b, usable / p, p]).min_dim(2).reshape([b, usable / p]);
            (l, m)
        } else {
            (latent, mask)
        };
        let s = mask.dims()[1];
        let state = enc.forward(latent, mask.clone());
        let state = norm.forward(state);
        // 先頭に有効トークンの平均を 1 つ足す
        let mask3 = mask.clone().reshape([b, s, 1]);
        let denom = mask3.clone().sum_dim(1).clamp_min(1.0);
        let mean_tok = state.clone().mul(mask3).sum_dim(1).div(denom);
        let has_any = mask.clone().max_dim(1);
        Ok((Tensor::cat(vec![mean_tok, state], 1), Tensor::cat(vec![has_any, mask], 1)))
    }

    /// 層ごとの text / speaker / caption の K,V を事前計算する(条件が固定の間は全ステップで共通)
    pub fn build_context_kv_cache(&self, c: &Conditions) -> Result<ContextKv> {
        let layers = self.blocks.iter().map(|b| b.attention.project_context(c)).collect::<Result<Vec<_>>>()?;
        Ok(ContextKv { layers })
    }

    /// `x_t`: `[B, S, latent_dim*latent_patch]`、`t` / `delta_t`: `[B]`。速度 `[B, S, ..]` を返す。
    /// `kv` が `None` のときは都度 `build_context_kv_cache` する。
    pub fn forward_with_encoded_conditions(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        delta_t: Tensor<1>,
        cond: &Conditions,
        kv: Option<&ContextKv>,
    ) -> Result<Tensor<3>> {
        let [b, s, _] = x_t.dims();
        let dev = &self.device;
        let ted = self.cfg.timestep_embed_dim;
        let tv = host_vec(t)?;
        let dv = host_vec(delta_t)?;
        ensure!(tv.len() == b && dv.len() == b, "t / delta_t must have batch size {b}");
        let mut cond_embed = self.cond_module.forward(timestep_embedding(&tv, ted, dev));
        let delta = self.delta_cond_module.as_ref().context("MeanFlow delta_cond_module missing")?;
        cond_embed = cond_embed.add(delta.forward(timestep_embedding(&dv, ted, dev)));

        let built;
        let kv = match kv {
            Some(k) => k,
            None => {
                built = self.build_context_kv_cache(cond)?;
                &built
            }
        };
        ensure!(kv.layers.len() == self.blocks.len(), "context kv cache layer count mismatch");

        let heads = self.cfg.num_heads;
        // 連結キー = [自己(全有効), text, speaker, caption]
        let ctx_mask = cond.context_mask();
        let key_mask = Tensor::cat(vec![Tensor::<2>::ones([b, s], dev), ctx_mask], 1);
        let bias = mask_to_bias(key_mask, heads, s);

        let rope = Rope::new(self.cfg.model_dim / heads, s, dev);
        let mut x = self.in_proj.forward3(x_t);
        for (blk, lkv) in self.blocks.iter().zip(&kv.layers) {
            let (shift, scale, gate) = blk.attention_adaln.modulation(&cond_embed);
            let h = blk.attention_adaln.norm_modulate(x.clone(), shift, scale);
            x = x.add(gate.mul(blk.attention.forward(h, lkv, bias.clone(), &rope)));

            let (shift, scale, gate) = blk.mlp_adaln.modulation(&cond_embed);
            let h = blk.mlp_adaln.norm_modulate(x.clone(), shift, scale);
            x = x.add(gate.mul(blk.mlp.forward(h)));
        }
        let x = self.out_norm.forward(x);
        Ok(self.out_proj.forward3(x))
    }
}
