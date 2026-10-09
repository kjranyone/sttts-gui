//! Whisper(distil-large-v3 系: encoder 32 層 / decoder 2 層 / d_model 1280 / 128 メル)の順伝播。
//! 原典: transformers の `modeling_whisper.py`(eager 注意、事前正規化、GELU)。重みは HF の名前そのまま
//! (`model.encoder.*` / `model.decoder.*`)。出力側の射影は埋め込み表と共有(`proj_out` は無い)。
//!
//! 規約:
//! - 活性は `Tensor<2>`(`[系列, 1280]`、バッチ 1)で持ち、注意の直前だけ `[1, ヘッド, 系列, 64]` にする。
//! - 線形層は `[out, in]` の重みをロード時に 1 回転置して保持する(irodori と同じ)。
//! - 重みは f32 に展開して持つ(f16 の行列積は精度が足りなかった: docs/irodori-rs.md)。
//! - Conv1d は im2col を作らず、タップごとの行列積の和にする(畳み込みの一時バッファを大きくしない)。
//! - 巨大な埋め込み表(51866 x 1280)は CPU に置いて入力の行だけ引く。出力側の射影は GPU にも置くが、
//!   単一の巨大確保を避けるため行方向に `PROJ_CHUNK` ずつ分けた複数のテンソルにする。

use anyhow::{Context, Result, ensure};
use burn::tensor::activation::gelu;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, Tensor, TensorData};
use irodori::weights::Weights;

use crate::mel::{N_FRAMES, N_MELS};

const D: usize = 1280;
const HEADS: usize = 20;
const HEAD_DIM: usize = D / HEADS;
const ENC_LAYERS: usize = 32;
const DEC_LAYERS: usize = 2;
/// encoder の出力長(conv2 の stride 2 で 3000 → 1500)
pub const ENC_LEN: usize = N_FRAMES / 2;
pub const VOCAB: usize = 51866;
pub const MAX_TARGET: usize = 448;
const LN_EPS: f64 = 1e-5;
/// 無効キーに足すバイアス(softmax 後に厳密に 0 になる)
const MASK_NEG: f32 = -1.0e9;
/// 注意の一時テンソル `[ヘッド, 問い, 1500]` が大きくなりすぎないよう、問いをこの行数ずつ処理する
const Q_CHUNK: usize = 500;
/// 出力射影を分ける行数(1 つあたり約 66MB)
const PROJ_CHUNK: usize = 13_000;

// ---------------------------------------------------------------------------------------------
// 基本部品
// ---------------------------------------------------------------------------------------------

struct Linear {
    /// `[in, out]`(転置済み)
    wt: Tensor<2>,
    b: Option<Tensor<1>>,
}

impl Linear {
    fn load(w: &Weights, prefix: &str, bias: bool, dev: &Device) -> Result<Self> {
        let wt = w.tensor::<2>(&format!("{prefix}.weight"), dev)?.transpose();
        let b = if bias { Some(w.tensor::<1>(&format!("{prefix}.bias"), dev)?) } else { None };
        Ok(Self { wt, b })
    }

    /// q / k / v を 1 つの `[1280, 3840]` にまとめる(k にはバイアスが無いので 0)。
    fn load_qkv(w: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        let mut data = Vec::with_capacity(3 * D * D);
        for n in ["q_proj", "k_proj", "v_proj"] {
            data.extend(w.f32_vec(&format!("{prefix}.{n}.weight"))?.1);
        }
        let wt = Tensor::<2>::from_data(TensorData::new(data, [3 * D, D]), dev).transpose();
        let mut bias = w.f32_vec(&format!("{prefix}.q_proj.bias"))?.1;
        bias.extend(std::iter::repeat_n(0f32, D));
        bias.extend(w.f32_vec(&format!("{prefix}.v_proj.bias"))?.1);
        Ok(Self { wt, b: Some(Tensor::<1>::from_data(TensorData::new(bias, [3 * D]), dev)) })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let y = x.matmul(self.wt.clone());
        match &self.b {
            Some(b) => {
                let n = b.dims()[0];
                y.add(b.clone().reshape([1, n]))
            }
            None => y,
        }
    }
}

struct LayerNorm {
    w: Tensor<2>,
    b: Tensor<2>,
}

impl LayerNorm {
    fn load(w: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        let get = |n: &str| -> Result<Tensor<2>> {
            let t = w.tensor::<1>(&format!("{prefix}.{n}"), dev)?;
            let len = t.dims()[0];
            Ok(t.reshape([1, len]))
        };
        Ok(Self { w: get("weight")?, b: get("bias")? })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mean = x.clone().mean_dim(1);
        let xc = x - mean;
        let var = xc.clone().square().mean_dim(1);
        xc / (var + LN_EPS).sqrt() * self.w.clone() + self.b.clone()
    }
}

/// `[S, 1280]` → `[1, ヘッド, S, 64]`
fn to_heads(x: Tensor<2>) -> Tensor<4> {
    let s = x.dims()[0];
    x.reshape([1, s, HEADS, HEAD_DIM]).swap_dims(1, 2)
}

/// `[1, ヘッド, S, 64]` → `[S, 1280]`
fn from_heads(x: Tensor<4>) -> Tensor<2> {
    let s = x.dims()[2];
    x.swap_dims(1, 2).reshape([s, D])
}

/// 注意(スケールは `1/sqrt(64)`)。問いを `Q_CHUNK` 行ずつに分けて、スコア行列を大きくしない。
fn sdpa(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, bias: Option<Tensor<4>>) -> Tensor<4> {
    let s = q.dims()[2];
    if s <= Q_CHUNK {
        return attention(q, k, v, None, bias, AttentionModuleOptions::default());
    }
    let mut outs = Vec::new();
    let mut start = 0;
    while start < s {
        let n = Q_CHUNK.min(s - start);
        let qc = q.clone().narrow(2, start, n);
        let bc = bias.clone().map(|b| b.narrow(2, start, n));
        outs.push(attention(qc, k.clone(), v.clone(), None, bc, AttentionModuleOptions::default()));
        start += n;
    }
    Tensor::cat(outs, 2)
}

/// 行ベクトルの表 `[rows, D]`(CPU)から `ids` の行を集めて `[ids.len(), D]`
fn gather_rows(table: &[f32], ids: impl Iterator<Item = usize>, out: &mut [f32]) {
    for (i, id) in ids.enumerate() {
        out[i * D..(i + 1) * D].copy_from_slice(&table[id * D..(id + 1) * D]);
    }
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

struct EncLayer {
    attn_ln: LayerNorm,
    qkv: Linear,
    out: Linear,
    ffn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncLayer {
    fn load(w: &Weights, i: usize, dev: &Device) -> Result<Self> {
        let p = format!("model.encoder.layers.{i}");
        Ok(Self {
            attn_ln: LayerNorm::load(w, &format!("{p}.self_attn_layer_norm"), dev)?,
            qkv: Linear::load_qkv(w, &format!("{p}.self_attn"), dev)?,
            out: Linear::load(w, &format!("{p}.self_attn.out_proj"), true, dev)?,
            ffn_ln: LayerNorm::load(w, &format!("{p}.final_layer_norm"), dev)?,
            fc1: Linear::load(w, &format!("{p}.fc1"), true, dev)?,
            fc2: Linear::load(w, &format!("{p}.fc2"), true, dev)?,
        })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let n = x.dims()[0];
        let qkv = self.qkv.forward(self.attn_ln.forward(x.clone()));
        let q = to_heads(qkv.clone().narrow(1, 0, D));
        let k = to_heads(qkv.clone().narrow(1, D, D));
        let v = to_heads(qkv.narrow(1, 2 * D, D));
        debug_assert_eq!(n, ENC_LEN);
        let x = x + self.out.forward(from_heads(sdpa(q, k, v, None)));
        let h = gelu(self.fc1.forward(self.ffn_ln.forward(x.clone())));
        x + self.fc2.forward(h)
    }
}

struct Encoder {
    /// conv1 のタップごとの重み `[128, 1280]` x 3
    conv1: [Tensor<2>; 3],
    conv1_b: Tensor<2>,
    /// conv2 のタップごとの重み `[1280, 1280]` x 3
    conv2: [Tensor<2>; 3],
    conv2_b: Tensor<2>,
    pos: Tensor<2>,
    layers: Vec<EncLayer>,
    ln: LayerNorm,
}

/// `[out, in, 3]` の畳み込み重みを、タップごとの `[in, out]` 3 枚にする。
fn conv_taps(w: &Weights, name: &str, dev: &Device) -> Result<[Tensor<2>; 3]> {
    let (shape, data) = w.f32_vec(name)?;
    ensure!(shape.len() == 3 && shape[2] == 3, "{name}: unexpected shape {shape:?}");
    let (o, i) = (shape[0], shape[1]);
    let tap = |k: usize| {
        let mut t = vec![0f32; i * o];
        for oo in 0..o {
            for ii in 0..i {
                t[ii * o + oo] = data[(oo * i + ii) * 3 + k];
            }
        }
        Tensor::<2>::from_data(TensorData::new(t, [i, o]), dev)
    };
    Ok([tap(0), tap(1), tap(2)])
}

impl Encoder {
    fn load(w: &Weights, dev: &Device, progress: &dyn Fn(&str)) -> Result<Self> {
        let bias = |n: &str| -> Result<Tensor<2>> {
            Ok(w.tensor::<1>(n, dev)?.reshape([1, D]))
        };
        let mut layers = Vec::with_capacity(ENC_LAYERS);
        for i in 0..ENC_LAYERS {
            if i % 8 == 0 {
                progress(&format!("encoder 層 {i}/{ENC_LAYERS}"));
            }
            layers.push(EncLayer::load(w, i, dev).with_context(|| format!("encoder layer {i}"))?);
        }
        Ok(Self {
            conv1: conv_taps(w, "model.encoder.conv1.weight", dev)?,
            conv1_b: bias("model.encoder.conv1.bias")?,
            conv2: conv_taps(w, "model.encoder.conv2.weight", dev)?,
            conv2_b: bias("model.encoder.conv2.bias")?,
            pos: w.tensor::<2>("model.encoder.embed_positions.weight", dev)?,
            layers,
            ln: LayerNorm::load(w, "model.encoder.layer_norm", dev)?,
        })
    }

    /// `mel`: `[N_MELS * N_FRAMES]`(メル軸が外側)→ `[1500, 1280]`
    fn forward(&self, mel: &[f32], dev: &Device) -> Tensor<2> {
        // [128, 3000] → [3000, 128]
        let mut tr = vec![0f32; N_FRAMES * N_MELS];
        for m in 0..N_MELS {
            for t in 0..N_FRAMES {
                tr[t * N_MELS + m] = mel[m * N_FRAMES + t];
            }
        }
        let x = Tensor::<2>::from_data(TensorData::new(tr, [N_FRAMES, N_MELS]), dev);
        // conv1: k=3, stride 1, pad 1
        let xp = Tensor::cat(
            vec![Tensor::<2>::zeros([1, N_MELS], dev), x, Tensor::<2>::zeros([1, N_MELS], dev)],
            0,
        );
        let mut h = xp.clone().narrow(0, 0, N_FRAMES).matmul(self.conv1[0].clone());
        for k in 1..3 {
            h = h + xp.clone().narrow(0, k, N_FRAMES).matmul(self.conv1[k].clone());
        }
        let h = gelu(h + self.conv1_b.clone());
        // conv2: k=3, stride 2, pad 1。出力 t の入力位置は 2t + k(パディング込み)
        let hp = Tensor::cat(vec![Tensor::<2>::zeros([1, D], dev), h, Tensor::<2>::zeros([1, D], dev)], 0);
        let tap = |k: usize| {
            hp.clone().narrow(0, k, N_FRAMES).reshape([ENC_LEN, 2, D]).narrow(1, 0, 1).reshape([ENC_LEN, D])
        };
        let mut h = tap(0).matmul(self.conv2[0].clone());
        for k in 1..3 {
            h = h + tap(k).matmul(self.conv2[k].clone());
        }
        let mut h = gelu(h + self.conv2_b.clone()) + self.pos.clone();
        for l in &self.layers {
            h = l.forward(h);
        }
        self.ln.forward(h)
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------

struct DecLayer {
    sa_ln: LayerNorm,
    sa_qkv: Linear,
    sa_out: Linear,
    ca_ln: LayerNorm,
    ca_q: Linear,
    ca_k: Linear,
    ca_v: Linear,
    ca_out: Linear,
    ffn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl DecLayer {
    fn load(w: &Weights, i: usize, dev: &Device) -> Result<Self> {
        let p = format!("model.decoder.layers.{i}");
        let lin = |n: &str, bias: bool| Linear::load(w, &format!("{p}.{n}"), bias, dev);
        Ok(Self {
            sa_ln: LayerNorm::load(w, &format!("{p}.self_attn_layer_norm"), dev)?,
            sa_qkv: Linear::load_qkv(w, &format!("{p}.self_attn"), dev)?,
            sa_out: lin("self_attn.out_proj", true)?,
            ca_ln: LayerNorm::load(w, &format!("{p}.encoder_attn_layer_norm"), dev)?,
            ca_q: lin("encoder_attn.q_proj", true)?,
            ca_k: lin("encoder_attn.k_proj", false)?,
            ca_v: lin("encoder_attn.v_proj", true)?,
            ca_out: lin("encoder_attn.out_proj", true)?,
            ffn_ln: LayerNorm::load(w, &format!("{p}.final_layer_norm"), dev)?,
            fc1: lin("fc1", true)?,
            fc2: lin("fc2", true)?,
        })
    }
}

/// 層ごとの cross-attention の K / V(`[1, ヘッド, 1500, 64]`)。encoder 出力ごとに 1 回だけ作る。
pub struct CrossKv {
    layers: Vec<(Tensor<4>, Tensor<4>)>,
}

/// 層ごとの self-attention の K / V キャッシュ(`[1, ヘッド, 長さ, 64]`)。複製は安価(テンソルの参照)。
#[derive(Clone, Default)]
pub struct SelfCache {
    layers: Vec<Option<(Tensor<4>, Tensor<4>)>>,
    /// 処理済みトークン数
    len: usize,
}

impl SelfCache {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct Model {
    enc: Encoder,
    dec_layers: Vec<DecLayer>,
    dec_ln: LayerNorm,
    /// トークン埋め込み `[VOCAB, D]`(CPU)
    tok_emb: Vec<f32>,
    /// 位置埋め込み `[448, D]`(CPU)
    pos_emb: Vec<f32>,
    /// 出力射影 `[PROJ_CHUNK 行ずつ, D]`(転置して使う)
    proj: Vec<Tensor<2>>,
    device: Device,
}

impl Model {
    pub fn load(w: &Weights, dev: &Device, progress: &dyn Fn(&str)) -> Result<Self> {
        let enc = Encoder::load(w, dev, progress)?;
        progress("decoder");
        let dec_layers = (0..DEC_LAYERS)
            .map(|i| DecLayer::load(w, i, dev).with_context(|| format!("decoder layer {i}")))
            .collect::<Result<Vec<_>>>()?;
        let (shape, tok_emb) = w.f32_vec("model.decoder.embed_tokens.weight")?;
        ensure!(shape == [VOCAB, D], "embed_tokens shape {shape:?}");
        let (_, pos_emb) = w.f32_vec("model.decoder.embed_positions.weight")?;
        let proj = tok_emb
            .chunks(PROJ_CHUNK * D)
            .map(|c| Tensor::<2>::from_data(TensorData::new(c.to_vec(), [c.len() / D, D]), dev))
            .collect();
        Ok(Self {
            enc,
            dec_layers,
            dec_ln: LayerNorm::load(w, "model.decoder.layer_norm", dev)?,
            tok_emb,
            pos_emb,
            proj,
            device: dev.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// ログメル `[N_MELS * N_FRAMES]` → encoder 出力 `[1500, 1280]`
    pub fn encode(&self, mel: &[f32]) -> Tensor<2> {
        self.enc.forward(mel, &self.device)
    }

    pub fn cross_kv(&self, enc: &Tensor<2>) -> CrossKv {
        let layers = self
            .dec_layers
            .iter()
            .map(|l| (to_heads(l.ca_k.forward(enc.clone())), to_heads(l.ca_v.forward(enc.clone()))))
            .collect();
        CrossKv { layers }
    }

    pub fn new_cache(&self) -> SelfCache {
        SelfCache { layers: vec![None; DEC_LAYERS], len: 0 }
    }

    /// `tokens` を位置 `cache.len()` から続けて処理し、最後のトークンの logits(`[VOCAB]`)を返す。
    /// `cache` は更新される。
    pub fn decode_step(&self, tokens: &[i64], cache: &mut SelfCache, cross: &CrossKv) -> Result<Vec<f32>> {
        let s = tokens.len();
        let pos0 = cache.len;
        ensure!(s > 0 && pos0 + s <= MAX_TARGET, "decoder position out of range ({pos0} + {s})");
        let mut x = vec![0f32; s * D];
        let mut p = vec![0f32; s * D];
        gather_rows(&self.tok_emb, tokens.iter().map(|&t| t as usize), &mut x);
        gather_rows(&self.pos_emb, pos0..pos0 + s, &mut p);
        for (a, b) in x.iter_mut().zip(&p) {
            *a += b;
        }
        let mut h = Tensor::<2>::from_data(TensorData::new(x, [s, D]), &self.device);
        // 複数トークンを一度に処理するときだけ因果マスクが要る
        let total = pos0 + s;
        let bias = (s > 1).then(|| {
            let mut m = vec![0f32; s * total];
            for i in 0..s {
                for j in pos0 + i + 1..total {
                    m[i * total + j] = MASK_NEG;
                }
            }
            Tensor::<4>::from_data(TensorData::new(m, [1, 1, s, total]), &self.device)
        });
        for (li, l) in self.dec_layers.iter().enumerate() {
            let qkv = l.sa_qkv.forward(l.sa_ln.forward(h.clone()));
            let q = to_heads(qkv.clone().narrow(1, 0, D));
            let mut k = to_heads(qkv.clone().narrow(1, D, D));
            let mut v = to_heads(qkv.narrow(1, 2 * D, D));
            if let Some((pk, pv)) = cache.layers[li].take() {
                k = Tensor::cat(vec![pk, k], 2);
                v = Tensor::cat(vec![pv, v], 2);
            }
            cache.layers[li] = Some((k.clone(), v.clone()));
            h = h + l.sa_out.forward(from_heads(sdpa(q, k, v, bias.clone())));

            let q = to_heads(l.ca_q.forward(l.ca_ln.forward(h.clone())));
            let (ck, cv) = &cross.layers[li];
            h = h + l.ca_out.forward(from_heads(sdpa(q, ck.clone(), cv.clone(), None)));

            let f = gelu(l.fc1.forward(l.ffn_ln.forward(h.clone())));
            h = h + l.fc2.forward(f);
        }
        cache.len = total;
        let last = self.dec_ln.forward(h.narrow(0, s - 1, 1));
        let parts: Vec<Tensor<2>> = self.proj.iter().map(|p| last.clone().matmul(p.clone().transpose())).collect();
        let logits = Tensor::cat(parts, 1);
        let data = logits.into_data().convert::<f32>();
        data.try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("logits: {e:?}"))
    }
}
