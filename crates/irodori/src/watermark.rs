//! SilentCipher(Sony、44.1kHz モデル)による電子透かしの埋め込み(エンコード側のみ)。
//!
//! 原典: `silentcipher/server.py` の `Model.encode_wav`(`calc_sdr=False`)と
//! `irodori_tts/watermark.py`。処理手順:
//!
//! 1. 入力が 44.1kHz でなければ torchaudio 互換(`sinc_interp_hann`)でリサンプル
//! 2. 電力を VCTK 平均(0.0028372)に正規化
//! 3. 右に `4096 - n % 4096` サンプルのゼロ詰め → STFT(n_fft=4096, hop=2048, Hann, center/reflect)
//!    → 振幅 `carrier` [1,1,2049,T] と位相
//! 4. `enc_c`(ゲート付き畳み込み 3 層 + BatchNorm)で carrier を符号化、
//!    ペイロード(5 バイト → 2bit × 20 + 終端 0 の 21 シンボル、one-hot 5 次元、フレーム方向にタイル)を
//!    `Linear(5→1024)`(+ 2049 バンドへゼロ埋め)で変換し、`[enc, carrier×32, msg×32]` の 96ch を `dec_c` へ
//! 5. `dec_c` の出力を abs → 1024 バンド以上を 0 → バンド毎 RMS で割り `10^(SDR/20)` で割る
//!    → carrier の全体 RMS を掛ける → `relu(carrier - message)` が新しい振幅
//! 6. 元の位相で iSTFT → 末尾のパディングを除去 → 電力を元に戻す → 元のレートへリサンプル
//!
//! 注意: 原典は `.eval()` を呼ばないため BatchNorm は **学習モード(入力のバッチ統計)** で動く。
//! ここでも同じく入力から統計を取る(running_mean/var は使わない)。

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use burn::tensor::activation::sigmoid;
use burn::tensor::{Device, Tensor, TensorData};
use realfft::RealFftPlanner;

use crate::codec::resample;
use crate::pth::Pth;

/// Irodori が埋め込むペイロード("IRDTS")
pub const IRODORI_PAYLOAD: [u8; 5] = [73, 82, 68, 84, 83];

const AVERAGE_ENERGY_VCTK: f64 = 0.002837200844477648;
const BN_EPS: f32 = 1e-5;

/// `hparams.yaml` から使う項目
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub message_band_size: usize,
    pub message_dim: usize,
    pub message_len: usize,
    pub message_sdr: f32,
}

impl WatermarkConfig {
    fn parse(text: &str) -> Result<Self> {
        let mut m: HashMap<&str, &str> = HashMap::new();
        for line in text.lines() {
            if let Some((k, v)) = line.split_once(':') {
                m.insert(k.trim(), v.trim());
            }
        }
        let get = |k: &str| m.get(k).copied().ok_or_else(|| anyhow!("hparams.yaml: missing {k}"));
        let flag = |k: &str| -> Result<bool> { Ok(get(k)? == "true") };
        // この実装が対応する構成(44.1k モデル)以外は拒否する
        ensure!(flag("ensure_negative_message")?, "unsupported hparams: ensure_negative_message=false");
        ensure!(flag("utterance_level_normalization")?, "unsupported hparams: utterance_level_normalization=false");
        ensure!(!flag("frame_level_normalization")?, "unsupported hparams: frame_level_normalization");
        ensure!(!flag("no_normalization")?, "unsupported hparams: no_normalization");
        ensure!(get("n_messages")? == "1", "unsupported hparams: n_messages != 1");
        ensure!(get("enc_n_layers")? == "3" && get("dec_c_n_layers")? == "4", "unsupported layer counts");
        Ok(Self {
            sample_rate: get("SR")?.parse()?,
            n_fft: get("N_FFT")?.parse()?,
            hop: get("HOP_LENGTH")?.parse()?,
            message_band_size: get("message_band_size")?.parse()?,
            message_dim: get("message_dim")?.parse()?,
            message_len: get("message_len")?.parse()?,
            message_sdr: get("message_sdr")?.parse()?,
        })
    }
}

/// 1 回の行列積で出力する位置数(固定長。最後の塊は零詰め)。大きすぎると一時メモリが膨らみ、
/// 小さすぎると演算の数が増える。`[k*k*C, CHUNK]` の im2col が数十 MB に収まる大きさ。
const CHUNK: usize = 16384;

/// 活性を「縁取り付きの格子を行優先で平らにした」形 `[C, buf_len]` で持つための寸法。
///
/// 格子は高さ `h`(周波数ビン)× 幅 `w`(フレーム)で、各行の幅を `wp = w + 2` に取り、行間の
/// 2 列分と先頭 `wp + 1`・末尾の余りを零にしておく。位置 `(i, j)` は `q = (i + 1) * wp + j + 1`。
/// こうすると 3x3 近傍は平らな配列上の一次元のずらし(`dy * wp + dx`)になるので、畳み込みは
/// 連続した切り出しを積んだ行列と重みの行列積になり、4D のまま切り出して詰め直すより
/// GPU でのコピーが連続で済み、一時メモリも少ない。
struct Geometry {
    h: usize,
    w: usize,
    wp: usize,
    nchunks: usize,
    buf_len: usize,
    /// `[1, nchunks * CHUNK]`: 有効な出力位置(行内の `j < w` かつ `p < ncols`)が 1
    mask: Tensor<2>,
    /// 有効位置の数(`h * w`)。BatchNorm の N
    n_valid: f32,
}

impl Geometry {
    fn new(h: usize, w: usize, device: &Device) -> Self {
        let wp = w + 2;
        let ncols = h * wp;
        let nchunks = ncols.div_ceil(CHUNK);
        let buf_len = nchunks * CHUNK + 2 * wp + 2;
        let mut m = vec![0f32; nchunks * CHUNK];
        for (p, v) in m.iter_mut().enumerate() {
            if p < ncols && p % wp < w {
                *v = 1.0;
            }
        }
        let mask = Tensor::<2>::from_data(TensorData::new(m, [1, nchunks * CHUNK]), device);
        Self { h, w, wp, nchunks, buf_len, mask, n_valid: (h * w) as f32 }
    }

    /// 1 チャネルの値 `[h * w]`(行優先)を縁取り付きの平らな配列 `[1, buf_len]` にして GPU へ送る
    fn place(&self, vals: &[f32], device: &Device) -> Tensor<2> {
        let mut buf = vec![0f32; self.buf_len];
        for i in 0..self.h {
            let q = (i + 1) * self.wp + 1;
            buf[q..q + self.w].copy_from_slice(&vals[i * self.w..(i + 1) * self.w]);
        }
        Tensor::<2>::from_data(TensorData::new(buf, [1, self.buf_len]), device)
    }

    /// 平らな出力 `[1, buf_len]` から `[h * w]`(行優先)を取り出す
    fn extract(&self, flat: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.h * self.w];
        for i in 0..self.h {
            let q = (i + 1) * self.wp + 1;
            out[i * self.w..(i + 1) * self.w].copy_from_slice(&flat[q..q + self.w]);
        }
        out
    }
}

/// ゲート付き畳み込み + BatchNorm(`silentcipher.model.Layer`)
struct Layer {
    /// conv と gate を出力チャネル方向に連結して im2col 用に並べた重み `[2*Cout, k*k*Cin]`(並びは dy, dx, c)
    w2: Tensor<2>,
    /// conv と gate の偏り `[2*Cout, 1]`
    b: Tensor<2>,
    out_ch: usize,
    bn_w: Tensor<2>,
    bn_b: Tensor<2>,
    k: usize,
    pad: usize,
}

impl Layer {
    fn load(sd: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let conv_w = sd.tensor::<4>(&format!("{prefix}.conv.weight"), device)?;
        let gate_w = sd.tensor::<4>(&format!("{prefix}.gate.weight"), device)?;
        let conv_b = sd.tensor::<1>(&format!("{prefix}.conv.bias"), device)?;
        let gate_b = sd.tensor::<1>(&format!("{prefix}.gate.bias"), device)?;
        let k = conv_w.dims()[2];
        let out_ch = conv_w.dims()[0];
        ensure!(k == 1 || k == 3, "unsupported kernel size {k}");
        let w = Tensor::cat(vec![conv_w, gate_w], 0);
        let [co2, c, kh, kw] = w.dims();
        let b = Tensor::cat(vec![conv_b, gate_b], 0);
        Ok(Self {
            w2: w.permute([0, 2, 3, 1]).reshape([co2, kh * kw * c]),
            b: b.reshape([co2, 1]),
            out_ch,
            bn_w: sd.tensor::<1>(&format!("{prefix}.bn.weight"), device)?.reshape([out_ch, 1]),
            bn_b: sd.tensor::<1>(&format!("{prefix}.bn.bias"), device)?.reshape([out_ch, 1]),
            k,
            pad: k / 2,
        })
    }

    /// 近傍 (dy, dx) の平らな配列上のずらし量
    fn taps(&self, wp: usize) -> Vec<usize> {
        let base = 1 - self.pad;
        (0..self.k).flat_map(|dy| (0..self.k).map(move |dx| (dy + base) * wp + dx + base)).collect()
    }

    /// `x`: 縁取り付きの平らな活性 `[Cin, buf_len]` → 同じ形式の `[Cout, buf_len]`。
    /// BatchNorm は学習モード(この入力の全有効位置でチャネルごとに平均・偏分散を取る)。
    fn forward(&self, x: &Tensor<2>, g: &Geometry) -> Tensor<2> {
        let co = self.out_ch;
        let taps = self.taps(g.wp);
        let mut ys = Vec::with_capacity(g.nchunks);
        let mut sum = Tensor::<2>::zeros([co, 1], &x.device());
        for ci in 0..g.nchunks {
            let base = ci * CHUNK;
            let mut parts: Vec<Tensor<2>> = taps.iter().map(|&o| x.clone().narrow(1, base + o, CHUNK)).collect();
            let cols = if parts.len() == 1 { parts.pop().unwrap() } else { Tensor::cat(parts, 0) };
            let cg = self.w2.clone().matmul(cols) + self.b.clone();
            let y = cg.clone().narrow(0, 0, co) * sigmoid(cg.narrow(0, co, co));
            let m = g.mask.clone().narrow(1, base, CHUNK);
            sum = sum + (y.clone() * m).sum_dim(1);
            ys.push(y);
        }
        let mean = sum.div_scalar(g.n_valid);
        let mut sq = Tensor::<2>::zeros([co, 1], &x.device());
        for (ci, y) in ys.iter().enumerate() {
            let m = g.mask.clone().narrow(1, ci * CHUNK, CHUNK);
            let d = (y.clone() - mean.clone()) * m;
            sq = sq + (d.clone() * d).sum_dim(1);
        }
        let inv = (sq.div_scalar(g.n_valid) + BN_EPS).sqrt().recip();
        let a = self.bn_w.clone() * inv;
        let shift = self.bn_b.clone() - mean * a.clone();
        let pieces: Vec<Tensor<2>> = ys
            .into_iter()
            .enumerate()
            .map(|(ci, y)| (y * a.clone() + shift.clone()) * g.mask.clone().narrow(1, ci * CHUNK, CHUNK))
            .collect();
        let o = if pieces.len() == 1 { pieces.into_iter().next().unwrap() } else { Tensor::cat(pieces, 1) };
        let z = Tensor::<2>::zeros([co, g.wp + 1], &x.device());
        Tensor::cat(vec![z.clone(), o, z], 1)
    }
}

pub struct Watermarker {
    device: Device,
    cfg: WatermarkConfig,
    enc: Vec<Layer>,
    dec: Vec<Layer>,
    /// `Linear(message_dim → message_band_size)` の重み [band, dim] と bias
    lin_w: Vec<f32>,
    lin_b: Vec<f32>,
}

impl Watermarker {
    /// `ckpt_dir` = `<snapshot>/44_1_khz/73999_iteration`(`enc_c.ckpt` `dec_c.ckpt` `hparams.yaml`)
    pub fn load(ckpt_dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = ckpt_dir.as_ref();
        let cfg = WatermarkConfig::parse(
            &std::fs::read_to_string(dir.join("hparams.yaml")).context("read hparams.yaml")?,
        )?;
        let enc_sd = StateDict::open(dir.join("enc_c.ckpt"))?;
        let dec_sd = StateDict::open(dir.join("dec_c.ckpt"))?;
        let enc = (0..3).map(|i| Layer::load(&enc_sd, &format!("main.{i}"), device)).collect::<Result<_>>()?;
        let dec = (0..4).map(|i| Layer::load(&dec_sd, &format!("main.{i}"), device)).collect::<Result<_>>()?;
        let (shape, lin_w) = enc_sd.f32_vec("linear.weight")?;
        ensure!(shape == [cfg.message_band_size, cfg.message_dim], "linear.weight shape {shape:?}");
        let (_, lin_b) = enc_sd.f32_vec("linear.bias")?;
        Ok(Self { device: device.clone(), cfg, enc, dec, lin_w, lin_b })
    }

    pub fn config(&self) -> &WatermarkConfig {
        &self.cfg
    }

    /// モノラル波形へ `payload`(5 バイト)を埋め込む。SDR は既定(47dB)。
    pub fn encode(&self, audio: &[f32], sample_rate: u32, payload: &[u8]) -> Result<Vec<f32>> {
        self.encode_with_sdr(audio, sample_rate, payload, self.cfg.message_sdr)
    }

    pub fn encode_with_sdr(&self, audio: &[f32], sample_rate: u32, payload: &[u8], sdr_db: f32) -> Result<Vec<f32>> {
        ensure!(payload.len() * 4 + 1 == self.cfg.message_len, "payload must be {} bytes", (self.cfg.message_len - 1) / 4);
        if audio.is_empty() {
            return Ok(audio.to_vec());
        }
        let sr = self.cfg.sample_rate;
        let y: Vec<f32> = if sample_rate != sr { resample(audio, sample_rate, sr)? } else { audio.to_vec() };
        let n = y.len();
        let power = (y.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64) as f32;
        if power == 0.0 {
            return Ok(audio.to_vec()); // 無音は埋め込まない(原典と同じ)
        }
        let norm = ((AVERAGE_ENERGY_VCTK as f32) / power).sqrt();
        let y: Vec<f32> = y.iter().map(|&v| v * norm).collect();

        let (n_fft, hop) = (self.cfg.n_fft, self.cfg.hop);
        let nb = n_fft / 2 + 1;
        let (mag, phase, frames) = stft(&y, n_fft, hop);

        // メッセージ(one-hot をフレーム方向へタイル)
        let msg_syms = encode_symbols(payload, self.cfg.message_len);
        let dim = self.cfg.message_dim;
        let band = self.cfg.message_band_size;
        let mut msg = vec![0f32; nb * frames];
        for t in 0..frames {
            let sym = msg_syms[t % self.cfg.message_len];
            for j in 0..band {
                msg[j * frames + t] = self.lin_w[j * dim + sym] + self.lin_b[j];
            }
        }

        let dev = &self.device;
        let geo = Geometry::new(nb, frames, dev);
        let carrier = geo.place(&mag, dev);
        let mut h = carrier.clone();
        for l in &self.enc {
            h = l.forward(&h, &geo);
        }
        let msg_t = geo.place(&msg, dev);
        let merged = Tensor::cat(
            vec![h, carrier.expand([32, geo.buf_len]), msg_t.expand([32, geo.buf_len])],
            0,
        );
        let mut d = merged;
        for l in &self.dec {
            d = l.forward(&d, &geo);
        }
        let flat = d.try_into_data().map_err(|e| anyhow!("GPU からの読み戻しに失敗しました: {e:?}"))?.convert::<f32>().try_to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let mut info = geo.extract(&flat);
        ensure!(info.len() == nb * frames, "unexpected decoder output size");

        // CarrierDecoder.forward の後処理
        let scale = 10f32.powf(sdr_db / 20.0);
        for v in info.iter_mut() {
            *v = v.abs();
        }
        for v in info[band * frames..].iter_mut() {
            *v = 0.0;
        }
        for t in 0..frames {
            let mut s = 0f32;
            for f in 0..nb {
                let v = info[f * frames + t];
                s += v * v;
            }
            let rms = (s / nb as f32).sqrt();
            for f in 0..nb {
                info[f * frames + t] = info[f * frames + t] / rms / scale;
            }
        }
        // utterance_level_normalization
        let car_rms = (mag.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / mag.len() as f64).sqrt() as f32;
        let new_mag: Vec<f32> = mag
            .iter()
            .zip(&info)
            .map(|(&c, &i)| ((-(i * car_rms)) + c).max(0.0)) // ensure_negative_message: relu(-info + carrier)
            .collect();

        let mut out = istft(&new_mag, &phase, frames, n_fft, hop);
        // STFT 前のゼロ詰めを除去
        out.truncate(n);
        let back = (power / AVERAGE_ENERGY_VCTK as f32).sqrt();
        for v in out.iter_mut() {
            *v *= back;
        }
        if sample_rate != sr {
            let mut r = resample(&out, sr, sample_rate)?;
            r.truncate(audio.len());
            return Ok(r);
        }
        Ok(out)
    }
}

/// ペイロードバイト列 → 2bit シンボル(+1)+ 終端 0。原典の `binary_encode` + `letters_encoding`。
fn encode_symbols(payload: &[u8], message_len: usize) -> Vec<usize> {
    let mut syms = Vec::with_capacity(message_len);
    for b in payload {
        for k in (0..4).rev() {
            syms.push((((b >> (2 * k)) & 3) as usize) + 1);
        }
    }
    syms.push(0);
    debug_assert_eq!(syms.len(), message_len);
    syms
}

fn hann(n: usize) -> Vec<f32> {
    // torch.hann_window(periodic=True)
    (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32).collect()
}

/// 原典の `STFT.transform`。戻り値は (振幅, 位相, フレーム数)。配列は [bin, frame] の行優先。
fn stft(x: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let n = x.len();
    let padded_len = n + (n_fft - n % n_fft);
    let mut xp = x.to_vec();
    xp.resize(padded_len, 0.0);
    // center=True, pad_mode=reflect
    let half = n_fft / 2;
    let mut ext = Vec::with_capacity(padded_len + n_fft);
    for i in 0..half {
        ext.push(xp[half - i]);
    }
    ext.extend_from_slice(&xp);
    for j in 0..half {
        ext.push(xp[padded_len - 2 - j]);
    }
    let frames = (ext.len() - n_fft) / hop + 1;
    let nb = n_fft / 2 + 1;
    let win = hann(n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut inbuf = r2c.make_input_vec();
    let mut spec = r2c.make_output_vec();
    let mut mag = vec![0f32; nb * frames];
    let mut phase = vec![0f32; nb * frames];
    for t in 0..frames {
        for i in 0..n_fft {
            inbuf[i] = ext[t * hop + i] * win[i];
        }
        r2c.process(&mut inbuf, &mut spec).expect("rfft");
        for (f, c) in spec.iter().enumerate() {
            mag[f * frames + t] = (c.re * c.re + c.im * c.im).sqrt();
            phase[f * frames + t] = c.im.atan2(c.re);
        }
    }
    (mag, phase, frames)
}

/// 原典の `STFT.inverse`(トリム前)。`torch.istft(center=True)` 相当。長さ `hop * (frames - 1)`。
fn istft(mag: &[f32], phase: &[f32], frames: usize, n_fft: usize, hop: usize) -> Vec<f32> {
    let nb = n_fft / 2 + 1;
    let win = hann(n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let c2r = planner.plan_fft_inverse(n_fft);
    let mut spec = c2r.make_input_vec();
    let mut buf = c2r.make_output_vec();
    let total = n_fft + hop * (frames - 1);
    let mut y = vec![0f32; total];
    let mut env = vec![0f32; total];
    for t in 0..frames {
        for f in 0..nb {
            let m = mag[f * frames + t];
            let p = phase[f * frames + t];
            spec[f].re = m * p.cos();
            spec[f].im = m * p.sin();
        }
        spec[0].im = 0.0;
        spec[nb - 1].im = 0.0;
        c2r.process(&mut spec, &mut buf).expect("irfft");
        for i in 0..n_fft {
            y[t * hop + i] += buf[i] / n_fft as f32 * win[i];
            env[t * hop + i] += win[i] * win[i];
        }
    }
    let half = n_fft / 2;
    (half..total - half).map(|i| y[i] / env[i]).collect()
}

// ---------------------------------------------------------------------------
// チェックポイント(`.ckpt` = PyTorch の zip + pickle。`pth` の読み取りを使う)
// ---------------------------------------------------------------------------

/// SilentCipher の ckpt。キーには DataParallel 由来の `module.` が付いている。
struct StateDict(Pth);

impl StateDict {
    fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self(Pth::load(path)?))
    }

    fn key(&self, name: &str) -> String {
        let prefixed = format!("module.{name}");
        if self.0.contains(&prefixed) { prefixed } else { name.to_string() }
    }

    fn f32_vec(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let t = self.0.get(&self.key(name))?;
        Ok((t.shape.clone(), t.data.clone()))
    }

    fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>> {
        self.0.tensor::<D>(&self.key(name), device)
    }
}
