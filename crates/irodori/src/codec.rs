//! DACVAE コーデック(`Aratako/Semantic-DACVAE-Japanese-32dim`、48 kHz、hop 1920、潜在 32 次元)。
//!
//! 原典は Meta の `dacvae`(`backend/.venv/Lib/site-packages/dacvae/`)と `irodori_tts/codec.py`。
//! - [`DacVae::decode_latent`]: 潜在 `[B,T,32]` → 波形 `[B,1,T*1920]`。デコーダは Snake 活性化 +
//!   ConvTranspose1d + 残差ブロック。Irodori は `decoder.alpha = 0` とし、透かし枝は
//!   `wm_model.encoder_block.forward_no_conv`(Snake → conv(96→1, k7) → Tanh)だけを通す
//!   (`codec.py` の `_watermark_passthrough`)。ELU/因果畳み込みの枝は透かし用なので使わない。
//! - [`DacVae::decode_latent_windowed`]: 時間窓ごとに(前後の文脈付きで)デコードする版。GPU で
//!   メモリと並列度を調整するためのもの。
//! - [`DacVae::encode_waveform`]: 参照音声用。モノ化 → リサンプル(torchaudio の sinc_interp_hann
//!   と同じ) → ラウドネス正規化(BS.1770 / pyloudnorm 相当) → 反射パディング → エンコーダ →
//!   `quantizer.in_proj` の平均成分(`deterministic_encode=True`)。
//!
//! 重みは [`crate::pth`] で読み、`weight_norm` は読み込み時に畳み込む。fp32 前提。

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Device, Tensor, TensorData};

use crate::pth::Pth;

/// Irodori が参照音声に使う既定のラウドネス目標(LUFS 相当)
pub const DEFAULT_NORMALIZE_DB: f32 = -16.0;

/// HF キャッシュ内の既定の重み(`weights.pth`)。無ければ None。
pub fn default_weights_path() -> Option<PathBuf> {
    crate::hub::find_snapshot("models--Aratako--Semantic-DACVAE-Japanese-32dim", "weights.pth").map(|d| d.join("weights.pth"))
}

// ---------------------------------------------------------------------------------------------
// 層
// ---------------------------------------------------------------------------------------------


/// im2col を作る出力列数の上限(1 回の行列積あたり)。`[k*C, 列]` が数十 MB に収まる大きさ
const CHUNK_COLS: usize = 8192;

struct Conv {
    w: Tensor<3>,
    /// `[Cout, k*C]`(im2col 用に並べ替え済み。層ごとに 1 回だけ作る)
    w2: Tensor<2>,
    b: Option<Tensor<1>>,
    opts: ConvOptions<1>,
    stride: usize,
    pad: usize,
    dil: usize,
}

impl Conv {
    fn load(p: &Pth, name: &str, stride: usize, pad: usize, dil: usize, dev: &Device) -> Result<Self> {
        let w = p.tensor::<3>(&format!("{name}.weight"), dev)?;
        let [co, c, k] = w.dims();
        let w2 = w.clone().swap_dims(1, 2).reshape([co, k * c]);
        Ok(Self {
            w,
            w2,
            b: if p.contains(&format!("{name}.bias")) {
                Some(p.tensor::<1>(&format!("{name}.bias"), dev)?)
            } else {
                None
            },
            opts: ConvOptions::new([stride], [pad], [dil], 1),
            stride,
            pad,
            dil,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, c, l] = x.dims();
        let [co, _, k] = self.w.dims();
        let (stride, pad, dil) = (self.stride, self.pad, self.dil);
        if batch != 1 || stride != 1 {
            return conv1d(x, self.w.clone(), self.b.clone(), self.opts.clone());
        }
        // burn の conv1d は、カーネル幅 1 の畳み込みで GPU(wgpu)のカーネル探索が終わらないことがあり、
        // それ以外も行列積より 2〜5 倍遅い。stride 1・batch 1(デコーダ・エンコーダの全層)は
        // im2col + 行列積で計算する: ずらしたスライスを積んで [Cout, k*C] x [k*C, L] にする。
        let lout = l + 2 * pad - dil * (k - 1);
        let w2 = self.w2.clone();
        if k == 1 {
            let y = w2.matmul(x.reshape([c, l])).reshape([1, co, lout]);
            return match &self.b {
                Some(b) => y + b.clone().reshape([1, co, 1]),
                None => y,
            };
        }
        let dev = x.device();
        let xp = if pad > 0 {
            Tensor::cat(vec![Tensor::<3>::zeros([1, c, pad], &dev), x, Tensor::<3>::zeros([1, c, pad], &dev)], 2)
        } else {
            x
        };
        // im2col の列を一度に作ると数百 MB になり、GPU のメモリプールが肥大して以降の処理が遅くなる。
        // 出力を CHUNK_COLS 列ずつに分けて、小さな行列積にする。
        let mut pieces = Vec::new();
        let mut t0 = 0;
        while t0 < lout {
            let n = CHUNK_COLS.min(lout - t0);
            let parts: Vec<Tensor<3>> = (0..k).map(|j| xp.clone().narrow(2, t0 + j * dil, n)).collect();
            let cols = Tensor::cat(parts, 1).reshape([k * c, n]);
            pieces.push(w2.clone().matmul(cols).reshape([1, co, n]));
            t0 += n;
        }
        let y = if pieces.len() == 1 { pieces.pop().unwrap() } else { Tensor::cat(pieces, 2) };
        match &self.b {
            Some(b) => y + b.clone().reshape([1, co, 1]),
            None => y,
        }
    }
}

struct ConvT {
    w: Tensor<3>,
    b: Option<Tensor<1>>,
    opts: ConvTransposeOptions<1>,
}

impl ConvT {
    /// `NormConvTranspose1d`(pad_mode="none"): padding = (s+1)/2、output_padding = s%2
    fn load(p: &Pth, name: &str, stride: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            w: p.tensor::<3>(&format!("{name}.weight"), dev)?,
            b: Some(p.tensor::<1>(&format!("{name}.bias"), dev)?),
            opts: ConvTransposeOptions::new([stride], [stride.div_ceil(2)], [stride % 2], [1], 1),
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        conv_transpose1d(x, self.w.clone(), self.b.clone(), self.opts.clone())
    }
}

/// Snake: `x + 1/(alpha+1e-9) * sin(alpha*x)^2`(alpha は `[1,C,1]`)
struct Snake {
    alpha: Tensor<3>,
    inv: Tensor<3>,
}

impl Snake {
    fn load(p: &Pth, name: &str, dev: &Device) -> Result<Self> {
        let alpha = p.tensor::<3>(name, dev)?;
        let inv = (alpha.clone() + 1e-9).recip();
        Ok(Self { alpha, inv })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let s = (x.clone() * self.alpha.clone()).sin();
        x + self.inv.clone() * (s.clone() * s)
    }
}

/// `ResidualUnit`(Snake → conv(k, dilation) → Snake → conv(k=1)) + スキップ
struct ResUnit {
    s0: Snake,
    c0: Conv,
    s1: Snake,
    c1: Conv,
}

impl ResUnit {
    fn load(p: &Pth, name: &str, dilation: usize, dev: &Device) -> Result<Self> {
        let w0 = p.get(&format!("{name}.block.1.weight"))?;
        let k = w0.shape[2];
        Ok(Self {
            s0: Snake::load(p, &format!("{name}.block.0.alpha"), dev)?,
            c0: Conv::load(p, &format!("{name}.block.1"), 1, (k - 1) * dilation / 2, dilation, dev)?,
            s1: Snake::load(p, &format!("{name}.block.2.alpha"), dev)?,
            c1: Conv::load(p, &format!("{name}.block.3"), 1, 0, 1, dev)?,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let a = self.c0.forward(self.s0.forward(x.clone()));
        let y = self.c1.forward(self.s1.forward(a));
        y + x
    }
}

struct DecoderBlock {
    snake: Snake,
    up: ConvT,
    units: [ResUnit; 3],
}

impl DecoderBlock {
    fn load(p: &Pth, name: &str, stride: usize, dev: &Device) -> Result<Self> {
        // block.0 Snake, block.1 ConvT, block.4/5/8 = ResidualUnit(dilation 1/3/9)。
        // block.2/3/6/7/9..11 は透かし用の ELU/因果枝なので使わない。
        Ok(Self {
            snake: Snake::load(p, &format!("{name}.block.0.alpha"), dev)?,
            up: ConvT::load(p, &format!("{name}.block.1"), stride, dev)?,
            units: [
                ResUnit::load(p, &format!("{name}.block.4"), 1, dev)?,
                ResUnit::load(p, &format!("{name}.block.5"), 3, dev)?,
                ResUnit::load(p, &format!("{name}.block.8"), 9, dev)?,
            ],
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mut x = self.up.forward(self.snake.forward(x));
        for u in &self.units {
            x = u.forward(x);
        }
        x
    }
}

struct Decoder {
    first: Conv,
    blocks: Vec<DecoderBlock>,
    wm_snake: Snake,
    wm_conv: Conv,
}

impl Decoder {
    fn load(p: &Pth, rates: &[usize], dev: &Device) -> Result<Self> {
        let k0 = p.get("decoder.model.0.weight")?.shape[2];
        let mut blocks = Vec::new();
        for (i, &s) in rates.iter().enumerate() {
            blocks.push(DecoderBlock::load(p, &format!("decoder.model.{}", i + 1), s, dev)?);
        }
        let wk = p.get("decoder.wm_model.encoder_block.pre.1.weight")?.shape[2];
        Ok(Self {
            first: Conv::load(p, "decoder.model.0", 1, (k0 - 1) / 2, 1, dev)?,
            blocks,
            wm_snake: Snake::load(p, "decoder.wm_model.encoder_block.pre.0.alpha", dev)?,
            wm_conv: Conv::load(p, "decoder.wm_model.encoder_block.pre.1", 1, (wk - 1) / 2, 1, dev)?,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mut x = self.first.forward(x);
        for b in &self.blocks {
            x = b.forward(x);
        }
        // `_watermark_passthrough`: encoder_block.forward_no_conv = Snake → conv(→1ch) → Tanh
        self.wm_conv.forward(self.wm_snake.forward(x)).tanh()
    }
}

struct EncoderBlock {
    units: [ResUnit; 3],
    snake: Snake,
    down: Conv,
}

impl EncoderBlock {
    fn load(p: &Pth, name: &str, stride: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            units: [
                ResUnit::load(p, &format!("{name}.block.0"), 1, dev)?,
                ResUnit::load(p, &format!("{name}.block.1"), 3, dev)?,
                ResUnit::load(p, &format!("{name}.block.2"), 9, dev)?,
            ],
            snake: Snake::load(p, &format!("{name}.block.3.alpha"), dev)?,
            // kernel = 2*stride、pad = ceil(stride / 2)(DAC と同じ。奇数のストライドでも長さが合う)
            down: Conv::load(p, &format!("{name}.block.4"), stride, stride.div_ceil(2), 1, dev)?,
        })
    }

    fn forward(&self, mut x: Tensor<3>) -> Tensor<3> {
        for u in &self.units {
            x = u.forward(x);
        }
        self.down.forward(self.snake.forward(x))
    }
}

struct Encoder {
    first: Conv,
    blocks: Vec<EncoderBlock>,
    snake: Snake,
    last: Conv,
}

impl Encoder {
    fn load(p: &Pth, rates: &[usize], dev: &Device) -> Result<Self> {
        let n = rates.len();
        let mut blocks = Vec::new();
        for (i, &s) in rates.iter().enumerate() {
            blocks.push(EncoderBlock::load(p, &format!("encoder.block.{}", i + 1), s, dev)?);
        }
        let k0 = p.get("encoder.block.0.weight")?.shape[2];
        let kl = p.get(&format!("encoder.block.{}.weight", n + 2))?.shape[2];
        Ok(Self {
            first: Conv::load(p, "encoder.block.0", 1, (k0 - 1) / 2, 1, dev)?,
            blocks,
            snake: Snake::load(p, &format!("encoder.block.{}.alpha", n + 1), dev)?,
            last: Conv::load(p, &format!("encoder.block.{}", n + 2), 1, (kl - 1) / 2, 1, dev)?,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mut x = self.first.forward(x);
        for b in &self.blocks {
            x = b.forward(x);
        }
        self.last.forward(self.snake.forward(x))
    }
}

// ---------------------------------------------------------------------------------------------
// モデル
// ---------------------------------------------------------------------------------------------

pub struct DacVae {
    pub sample_rate: u32,
    /// 潜在 1 フレームあたりのサンプル数(1920)
    pub hop: usize,
    /// 潜在次元(32)
    pub latent_dim: usize,
    device: Device,
    encoder: Encoder,
    in_proj: Conv,
    out_proj: Conv,
    decoder: Decoder,
}

fn rates(meta: &serde_json::Value, key: &str, default: &[usize]) -> Vec<usize> {
    meta["kwargs"][key]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
        .unwrap_or_else(|| default.to_vec())
}

impl DacVae {
    /// `weights.pth` を読む。
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let mut p = Pth::load(path).with_context(|| format!("load {}", path.display()))?;
        p.fold_weight_norm()?;
        Self::from_pth(&p, device)
    }

    /// HF キャッシュの既定の重みを読む。
    pub fn load_default(device: &Device) -> Result<Self> {
        let path = default_weights_path().context("DACVAE weights.pth not found in the HF cache")?;
        Self::load(path, device)
    }

    /// `weight_norm` を畳み込み済みの [`Pth`] から構築する。
    pub fn from_pth(p: &Pth, device: &Device) -> Result<Self> {
        let meta = &p.metadata;
        let enc_rates = rates(meta, "encoder_rates", &[2, 8, 10, 12]);
        let dec_rates = rates(meta, "decoder_rates", &[12, 10, 8, 2]);
        let sample_rate = meta["kwargs"]["sample_rate"].as_u64().unwrap_or(48000) as u32;
        let hop: usize = enc_rates.iter().product();
        let dec_hop: usize = dec_rates.iter().product();
        if hop != dec_hop {
            bail!("encoder hop {hop} != decoder hop {dec_hop}");
        }
        let in_proj = Conv::load(p, "quantizer.in_proj", 1, 0, 1, device)?;
        let out_proj = Conv::load(p, "quantizer.out_proj", 1, 0, 1, device)?;
        let latent_dim = p.get("quantizer.out_proj.weight")?.shape[1];
        Ok(Self {
            sample_rate,
            hop,
            latent_dim,
            device: device.clone(),
            encoder: Encoder::load(p, &enc_rates, device)?,
            in_proj,
            out_proj,
            decoder: Decoder::load(p, &dec_rates, device)?,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 潜在 `[B,T,D]` → 波形 `[B,1,T*hop]`(全体一括)。
    pub fn decode_latent(&self, z: Tensor<3>) -> Tensor<3> {
        let z = z.swap_dims(1, 2);
        self.decoder.forward(self.out_proj.forward(z))
    }

    /// 時間窓ごとに分割してデコードする。`window` フレームを 1 窓の「採用範囲」とし、前後に
    /// `context` フレームの文脈を付けた(固定長 `window + 2*context` の)窓をデコードして、採用範囲だけを切り出して連結する。
    /// 先頭・末尾は全体版と同じゼロ詰めになる(文脈は信号の端でクリップ)。
    /// 受容野より `context` が大きければ全体版とほぼ一致する。
    pub fn decode_latent_windowed(&self, z: Tensor<3>, window: usize, context: usize) -> Tensor<3> {
        let [_, t, _] = z.dims();
        let window = window.max(1);
        // どの窓も同じ長さ(window + 2*context フレーム)でデコードする。GPU のカーネルは形状の
        // 整列クラスごとに作り直されるので、窓の長さを固定すると初回以外はコンパイルが起きない。
        // 端の窓は文脈を内側にずらして取る(文脈が受容野以上なら全体デコードと同じ結果)。
        let win = window + 2 * context;
        if t <= win {
            return self.decode_latent(z);
        }
        let mut parts = Vec::new();
        let mut s = 0;
        while s < t {
            let e = (s + window).min(t);
            let a = s.saturating_sub(context).min(t - win);
            let out = self.decode_latent(z.clone().narrow(1, a, win));
            parts.push(out.narrow(2, (s - a) * self.hop, (e - s) * self.hop));
            s = e;
        }
        Tensor::cat(parts, 2)
    }

    /// 参照音声 `[B,C,N]`(サンプルレート `sample_rate`)→ 潜在 `[B,T,D]`。
    ///
    /// `codec.py::encode_waveform` と同じ前処理: チャネル平均 → 48 kHz へリサンプル →
    /// `normalize_db` が `Some` ならラウドネス正規化(+ピーク 1.0 への制限)、`None` で
    /// `ensure_max` ならピークが 1.0 を超える場合のみ縮小 → hop の倍数へ反射パディング →
    /// エンコーダ → `in_proj` の平均成分。
    pub fn encode_waveform(
        &self,
        wav: Tensor<3>,
        sample_rate: u32,
        normalize_db: Option<f32>,
        ensure_max: bool,
    ) -> Result<Tensor<3>> {
        let [b, c, n] = wav.dims();
        let data = wav.try_into_data().map_err(|e| anyhow::anyhow!("GPU からの読み戻しに失敗しました: {e:?}"))?.convert::<f32>().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut all: Vec<f32> = Vec::new();
        let mut len = 0usize;
        for bi in 0..b {
            let item = &data[bi * c * n..(bi + 1) * c * n];
            let mut mono = vec![0f32; n];
            for ch in 0..c {
                for (m, &x) in mono.iter_mut().zip(&item[ch * n..(ch + 1) * n]) {
                    *m += x;
                }
            }
            if c != 1 {
                let inv = 1.0 / c as f32;
                mono.iter_mut().for_each(|m| *m *= inv);
            }
            let mut x = if sample_rate != self.sample_rate {
                resample(&mono, sample_rate, self.sample_rate)?
            } else {
                mono
            };
            match normalize_db {
                Some(db) => {
                    normalize_loudness(&mut x, self.sample_rate, db as f64);
                    limit_peak(&mut x);
                }
                None if ensure_max => limit_peak(&mut x),
                None => {}
            }
            // `_pad`: hop の倍数へ反射パディング(右側)
            let rem = x.len() % self.hop;
            if rem != 0 {
                let p = self.hop - rem;
                if p >= x.len() {
                    bail!("waveform too short ({} samples) for reflect padding of {p}", x.len());
                }
                let l = x.len();
                for k in 0..p {
                    x.push(x[l - 2 - k]);
                }
            }
            len = x.len();
            all.extend_from_slice(&x);
        }
        let t = Tensor::<3>::from_data(TensorData::new(all, [b, 1, len]), &self.device);
        let z = self.in_proj.forward(self.encoder.forward(t));
        let mean = z.narrow(1, 0, self.latent_dim);
        Ok(mean.swap_dims(1, 2))
    }
}

// ---------------------------------------------------------------------------------------------
// 前処理(ホスト側): ピーク制限、ラウドネス、リサンプル
// ---------------------------------------------------------------------------------------------

/// ピークが 1.0 を超えるときだけ 1.0 に収まるよう縮小(audiotools `ensure_max_of_audio`)
fn limit_peak(x: &mut [f32]) {
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak.is_finite() && peak > 1.0 {
        let g = 1.0 / peak;
        x.iter_mut().for_each(|v| *v *= g);
    }
}

/// RBJ biquad(pyloudnorm `IIRfilter`)。係数は a0 で正規化済み。
fn biquad(kind: &str, g_db: f64, q: f64, fc: f64, rate: f64) -> ([f64; 3], [f64; 3]) {
    use std::f64::consts::PI;
    let a = 10f64.powf(g_db / 40.0);
    let w0 = 2.0 * PI * (fc / rate);
    let alpha = w0.sin() / (2.0 * q);
    let c = w0.cos();
    let (b0, b1, b2, a0, a1, a2) = match kind {
        "high_shelf" => (
            a * ((a + 1.0) + (a - 1.0) * c + 2.0 * a.sqrt() * alpha),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - 2.0 * a.sqrt() * alpha),
            (a + 1.0) - (a - 1.0) * c + 2.0 * a.sqrt() * alpha,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - 2.0 * a.sqrt() * alpha,
        ),
        _ => (
            (1.0 + c) / 2.0,
            -(1.0 + c),
            (1.0 + c) / 2.0,
            1.0 + alpha,
            -2.0 * c,
            1.0 - alpha,
        ),
    };
    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

fn lfilter(x: &[f64], b: &[f64; 3], a: &[f64; 3]) -> Vec<f64> {
    // 直接形 II 転置
    let (mut z1, mut z2) = (0f64, 0f64);
    x.iter()
        .map(|&v| {
            let y = b[0] * v + z1;
            z1 = b[1] * v - a[1] * y + z2;
            z2 = b[2] * v - a[2] * y;
            y
        })
        .collect()
}

/// ITU-R BS.1770-4 の統合ラウドネス(モノ、K 特性、400 ms ブロック、75% オーバーラップ、
/// 絶対 -70 / 相対 -10 のゲート)。audiotools `Meter.integrated_loudness` と同じ手順
/// (最終ブロックはゼロ詰めで全サンプルを覆う)。
pub fn integrated_loudness(x: &[f32], rate: u32) -> f64 {
    let r = rate as f64;
    let (b1, a1) = biquad("high_shelf", 4.0, 1.0 / 2f64.sqrt(), 1500.0, r);
    let (b2, a2) = biquad("high_pass", 0.0, 0.5, 38.0, r);
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let y = lfilter(&lfilter(&xf, &b1, &a1), &b2, &a2);

    let t_g = 0.400f64;
    let kernel = (t_g * r) as usize;
    let stride = (t_g * r * (1.0 - 0.75)) as usize;
    let n_frames = (y.len().max(kernel) - kernel).div_ceil(stride) + 1;
    let z: Vec<f64> = (0..n_frames)
        .map(|j| {
            let s = j * stride;
            let e = (s + kernel).min(y.len());
            let sum: f64 = if s < e { y[s..e].iter().map(|v| v * v).sum() } else { 0.0 };
            sum / (t_g * r)
        })
        .collect();
    let l: Vec<f64> = z.iter().map(|&v| -0.691 + 10.0 * v.log10()).collect();

    let gate = |thr_r: Option<f64>| -> f64 {
        let (mut sum, mut cnt) = (0f64, 0usize);
        for (&zz, &ll) in z.iter().zip(&l) {
            if ll > -70.0 && thr_r.is_none_or(|g| ll > g) {
                sum += zz;
                cnt += 1;
            }
        }
        sum / cnt as f64
    };
    let z_abs = gate(None);
    let gamma_r = -0.691 + 10.0 * z_abs.log10() - 10.0;
    let mut z_rel = gate(Some(gamma_r));
    if z_rel.is_nan() {
        z_rel = 0.0;
    }
    -0.691 + 10.0 * z_rel.log10()
}

/// `AudioSignal.normalize(db)`:ラウドネスを `db` に合わせる(0.5 秒未満は測定のみゼロ詰め、
/// 測定値は -70 で下限)。ピーク制限は呼び出し側([`limit_peak`])。
pub fn normalize_loudness(x: &mut [f32], rate: u32, db: f64) {
    let dur = x.len() as f64 / rate as f64;
    let measured = if dur < 0.5 {
        let mut p = x.to_vec();
        p.resize(x.len() + ((0.5 - dur) * rate as f64) as usize, 0.0);
        integrated_loudness(&p, rate)
    } else {
        integrated_loudness(x, rate)
    };
    let reference = measured.max(-70.0);
    let gain = ((db - reference) * (10f64.ln() / 20.0)).exp() as f32;
    x.iter_mut().for_each(|v| *v *= gain);
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// torchaudio `functional.resample`(`sinc_interp_hann`、lowpass_filter_width=6、rolloff=0.99)。
/// カーネル表の大きさの上限(f32 の個数。256MB)。互いに素に近い比(47999→48000 など)は表が巨大になる
const MAX_RESAMPLE_TABLE: usize = 64 << 20;

/// torchaudio の `sinc_interp_hann` 互換のリサンプル(`lowpass_filter_width = 6`、`rolloff = 0.99`)。
pub fn resample(x: &[f32], orig_freq: u32, new_freq: u32) -> Result<Vec<f32>> {
    if orig_freq == new_freq {
        return Ok(x.to_vec());
    }
    const LPW: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let g = gcd(orig_freq as u64, new_freq as u64);
    let orig = (orig_freq as u64 / g) as usize;
    let new = (new_freq as u64 / g) as usize;
    let base = (orig.min(new) as f64) * ROLLOFF;
    let width = (LPW * orig as f64 / base).ceil() as usize;
    let klen = 2 * width + orig;
    anyhow::ensure!(
        new.saturating_mul(klen) <= MAX_RESAMPLE_TABLE,
        "resample {orig_freq} -> {new_freq} Hz: 比が複雑すぎる(カーネル表 {} 個)。44100 / 48000 など一般的なレートに揃えてください",
        new.saturating_mul(klen)
    );
    // kernels[i][k]
    let mut kernels = vec![0f32; new * klen];
    let scale = base / orig as f64;
    for i in 0..new {
        for k in 0..klen {
            let idx = (k as f64 - width as f64) / orig as f64;
            let mut t = (-(i as f64) / new as f64 + idx) * base;
            t = t.clamp(-LPW, LPW);
            let window = (t * std::f64::consts::PI / LPW / 2.0).cos().powi(2);
            let tp = t * std::f64::consts::PI;
            let sinc = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            kernels[i * klen + k] = (sinc * window * scale) as f32;
        }
    }
    let n = x.len();
    let target = (new * n).div_ceil(orig);
    let frames = n / orig + 1;
    let mut out = vec![0f32; frames * new];
    for m in 0..frames {
        for i in 0..new {
            let mut acc = 0f32;
            let kr = &kernels[i * klen..(i + 1) * klen];
            // padded[m*orig + k] = x[m*orig + k - width]
            for (k, &kv) in kr.iter().enumerate() {
                let p = (m * orig + k) as isize - width as isize;
                if p >= 0 && (p as usize) < n {
                    acc += kv * x[p as usize];
                }
            }
            out[m * new + i] = acc;
        }
    }
    out.truncate(target);
    Ok(out)
}
