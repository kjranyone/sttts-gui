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
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use burn::tensor::activation::sigmoid;
use burn::tensor::{Device, Tensor, TensorData};
use realfft::RealFftPlanner;

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

/// ゲート付き畳み込み + BatchNorm(`silentcipher.model.Layer`)
struct Layer {
    /// conv と gate を出力チャネル方向に連結した重み `[2*Cout, Cin, k, k]` と偏り(1 回の行列積で両方を計算する)
    /// im2col 用に並べ替え済みの重み `[2*Cout, kh*kw*Cin]`(f16 モードでは f16)
    w2: Tensor<2>,
    b: Tensor<1>,
    out_ch: usize,
    k: usize,
    bn_w: Tensor<1>,
    bn_b: Tensor<1>,
    pad: usize,
}

/// im2col を作る出力列数の上限(1 回の行列積あたり)
const CONV_CHUNK_COLS: usize = 65536;

/// 3x3 などの 2D 畳み込み(stride 1・batch 1・「same」パディング)を行列積で計算する。
///
/// 入力を零で縁取って `[C, (H+2p)*(W+2p)]` に平らにすると、近傍 (dy, dx) は一次元の連続したずらし
/// (`dy*(W+2p) + dx` 位置の切り出し)になる。これを `k*k` 本積んだ `[k*k*C, 列]` と重みの行列積で
/// 畳み込みになる。2D のまま切り出して詰め直す方式に比べて、GPU での詰め直し(strided コピー)が
/// 連続コピーだけで済み、数倍速い。出力の各行末には縁取り分の無効な列が `2p` 個できるので最後に捨てる。
/// 列は固定長 `CONV_CHUNK_COLS` の塊に分け(最後の塊は零詰め)、GPU のカーネル形状を発話長に依存させない。
fn conv2d_mm(x: Tensor<4>, w2: Tensor<2>, k: usize, b: Tensor<1>, pad: usize) -> Tensor<4> {
    let [n, c, h, wd] = x.dims();
    let co = w2.dims()[0];
    let (kh, kw) = (k, k);
    assert_eq!(n, 1, "conv2d_mm は batch 1 のみ");
    assert!(kh == kw && kh == 2 * pad + 1, "conv2d_mm は kernel = 2*pad+1 の正方のみ");
    let dev = x.device();
    let trace = std::env::var_os("IRODORI_TRACE_WM").is_some();
    let t0 = std::time::Instant::now();
    let lap = |what: &str| {
        if trace {
            let _ = dev.sync();
            eprintln!("[wm]        conv2d_mm {what}: {:.3}s", t0.elapsed().as_secs_f64());
        }
    };
    let (hp, wp) = (h + 2 * pad, wd + 2 * pad);
    let zr = Tensor::<4>::zeros([1, c, pad, wd], &dev);
    let x = Tensor::cat(vec![zr.clone(), x, zr], 2);
    let zc = Tensor::<4>::zeros([1, c, hp, pad], &dev);
    let x = Tensor::cat(vec![zc.clone(), x, zc], 3);
    lap("pad cats");
    let flat = x.reshape([c, hp * wp]);
    lap("flatten");

    let ncols = h * wp; // 出力位置(各行は wp 列。末尾 2*pad 列は無効)
    let chunk = CONV_CHUNK_COLS.min(ncols.div_ceil(256) * 256);
    let nchunks = ncols.div_ceil(chunk);
    let halo = (kh - 1) * wp + (kw - 1);
    let need = nchunks * chunk + halo;
    let flat = if need > hp * wp { Tensor::cat(vec![flat, Tensor::<2>::zeros([c, need - hp * wp], &dev)], 1) } else { flat };

    lap("zero tail + w2");
    let mut pieces = Vec::with_capacity(nchunks);
    for ci in 0..nchunks {
        let base = ci * chunk;
        let mut parts = Vec::with_capacity(kh * kw);
        for dy in 0..kh {
            for dx in 0..kw {
                parts.push(flat.clone().narrow(1, base + dy * wp + dx, chunk));
            }
        }
        pieces.push(crate::ops::matmul_lw(w2.clone(), Tensor::cat(parts, 0)));
    }
    lap("chunk loop");
    let y = if pieces.len() == 1 { pieces.pop().unwrap() } else { Tensor::cat(pieces, 1) };
    lap("cat pieces");
    let y = y.narrow(1, 0, ncols).reshape([co, h, wp]).narrow(2, 0, wd);
    let out = (y + b.reshape([co, 1, 1])).reshape([1, co, h, wd]);
    lap("bias+crop");
    out
}

impl Layer {
    fn load(sd: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let conv_w = sd.tensor::<4>(&format!("{prefix}.conv.weight"), device)?;
        let gate_w = sd.tensor::<4>(&format!("{prefix}.gate.weight"), device)?;
        let conv_b = sd.tensor::<1>(&format!("{prefix}.conv.bias"), device)?;
        let gate_b = sd.tensor::<1>(&format!("{prefix}.gate.bias"), device)?;
        let k = conv_w.dims()[2];
        let out_ch = conv_w.dims()[0];
        Ok(Self {
            w2: {
                let w = Tensor::cat(vec![conv_w, gate_w], 0);
                let [co, c, kh, kw] = w.dims();
                crate::ops::store(w.permute([0, 2, 3, 1]).reshape([co, kh * kw * c]))
            },
            b: Tensor::cat(vec![conv_b, gate_b], 0),
            out_ch,
            k,
            bn_w: sd.tensor(&format!("{prefix}.bn.weight"), device)?,
            bn_b: sd.tensor(&format!("{prefix}.bn.bias"), device)?,
            pad: k / 2,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let trace = std::env::var_os("IRODORI_TRACE_WM").is_some();
        let dev = x.device();
        let t0 = std::time::Instant::now();
        let lap = |what: &str| {
            if trace {
                let _ = dev.sync();
                eprintln!("[wm]    {what}: {:.3}s", t0.elapsed().as_secs_f64());
            }
        };
        let cg = conv2d_mm(x, self.w2.clone(), self.k, self.b.clone(), self.pad);
        lap("conv");
        let c = cg.clone().narrow(1, 0, self.out_ch);
        let g = cg.narrow(1, self.out_ch, self.out_ch);
        let y = c * sigmoid(g);
        lap("gate");
        // BatchNorm2d(学習モード): チャネル毎に (N,H,W) で平均・偏分散
        let [n, ch, h, w] = y.dims();
        let flat = y.swap_dims(0, 1).reshape([ch, n * h * w]);
        lap("bn flatten");
        let mean = flat.clone().mean_dim(1);
        let centered = flat - mean;
        lap("bn mean+center");
        let var = centered.clone().powf_scalar(2.0).mean_dim(1);
        let norm = centered / (var + BN_EPS).sqrt();
        lap("bn var+norm");
        let out = norm * self.bn_w.clone().reshape([ch, 1]) + self.bn_b.clone().reshape([ch, 1]);
        let out = out.reshape([ch, n, h, w]).swap_dims(0, 1);
        lap("bn affine");
        out
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
        let trace = std::env::var_os("IRODORI_TRACE_WM").is_some();
        let mut t0 = std::time::Instant::now();
        let mut lap = |name: &str| {
            if trace {
                eprintln!("[wm] {name}: {:.3}s", t0.elapsed().as_secs_f64());
            }
            t0 = std::time::Instant::now();
        };
        let sr = self.cfg.sample_rate;
        let y: Vec<f32> = if sample_rate != sr { resample(audio, sample_rate, sr) } else { audio.to_vec() };
        lap("resample in");
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
        lap("stft");

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
        let carrier = Tensor::<4>::from_data(TensorData::new(mag.clone(), [1, 1, nb, frames]), dev);
        let mut h = carrier.clone();
        for l in &self.enc {
            h = l.forward(h);
        }
        if trace {
            let _ = self.device.sync();
            lap("encoder");
        }
        let msg_t = Tensor::<4>::from_data(TensorData::new(msg, [1, 1, nb, frames]), dev);
        let merged = Tensor::cat(
            vec![h, carrier.clone().expand([1, 32, nb, frames]), msg_t.expand([1, 32, nb, frames])],
            1,
        );
        let mut d = merged;
        for l in &self.dec {
            d = l.forward(d);
        }
        if trace {
            let _ = self.device.sync();
            lap("decoder");
        }
        let mut info = d.into_data().convert::<f32>().try_to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        lap("  readback");
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
        lap("  post: rms loops");
        // utterance_level_normalization
        let car_rms = (mag.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / mag.len() as f64).sqrt() as f32;
        let new_mag: Vec<f32> = mag
            .iter()
            .zip(&info)
            .map(|(&c, &i)| ((-(i * car_rms)) + c).max(0.0)) // ensure_negative_message: relu(-info + carrier)
            .collect();

        lap("readback+post");
        let mut out = istft(&new_mag, &phase, frames, n_fft, hop);
        lap("istft");
        // STFT 前のゼロ詰めを除去
        out.truncate(n);
        let back = (power / AVERAGE_ENERGY_VCTK as f32).sqrt();
        for v in out.iter_mut() {
            *v *= back;
        }
        if sample_rate != sr {
            let mut r = resample(&out, sr, sample_rate);
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

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `torchaudio.functional.resample` 既定設定(`sinc_interp_hann`、幅 6、rolloff 0.99)の移植。
pub fn resample(x: &[f32], orig_sr: u32, new_sr: u32) -> Vec<f32> {
    let g = gcd(orig_sr as u64, new_sr as u64);
    let (o, n) = ((orig_sr as u64 / g) as usize, (new_sr as u64 / g) as usize);
    if o == n {
        return x.to_vec();
    }
    let lpw = 6.0f64;
    let base = (o.min(n) as f64) * 0.99;
    let width = (lpw * o as f64 / base).ceil() as usize;
    let k = 2 * width + o;
    let scale = base / o as f64;
    let mut kernels = vec![0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let idx = (j as f64 - width as f64) / o as f64;
            let mut t = (-(i as f64) / n as f64 + idx) * base;
            t = t.clamp(-lpw, lpw);
            let w = (t * PI / lpw / 2.0).cos().powi(2);
            let tp = t * PI;
            let s = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            kernels[i * k + j] = (s * w * scale) as f32;
        }
    }
    let len = x.len();
    let mut xp = vec![0f32; width];
    xp.extend_from_slice(x);
    xp.extend(std::iter::repeat_n(0.0, width + o));
    let nframes = (xp.len() - k) / o + 1;
    let mut out = vec![0f32; nframes * n];
    for f in 0..nframes {
        let seg = &xp[f * o..f * o + k];
        for i in 0..n {
            let ker = &kernels[i * k..(i + 1) * k];
            let mut acc = 0f32;
            for (a, b) in seg.iter().zip(ker) {
                acc += a * b;
            }
            out[f * n + i] = acc;
        }
    }
    out.truncate((n * len).div_ceil(o));
    out
}

// ---------------------------------------------------------------------------
// 最小の PyTorch ckpt(zip + pickle)リーダ。state_dict(OrderedDict[str, Tensor])のみ対応。
// `pth.rs` と統合する際に差し替える。
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Val {
    None,
    Int(i64),
    Str(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Global(String),
    Dict(Vec<(Val, Val)>),
    Storage { key: String, dtype: String },
    Tensor { key: String, dtype: String, offset: usize, size: Vec<usize> },
    Mark,
}

struct TensorInfo {
    key: String,
    dtype: String,
    offset: usize,
    size: Vec<usize>,
}

struct StateDict {
    /// zip 内のディレクトリ名(`enc_c` など)
    prefix: String,
    zip: std::cell::RefCell<zip::ZipArchive<std::io::BufReader<std::fs::File>>>,
    tensors: HashMap<String, TensorInfo>,
}

impl StateDict {
    fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))?;
        let pkl_name = (0..zip.len())
            .filter_map(|i| zip.by_index(i).ok().map(|f| f.name().to_string()))
            .find(|n| n.ends_with("data.pkl"))
            .ok_or_else(|| anyhow!("data.pkl not found in {}", path.display()))?;
        let prefix = pkl_name.trim_end_matches("data.pkl").trim_end_matches('/').to_string();
        let mut pkl = Vec::new();
        zip.by_name(&pkl_name)?.read_to_end(&mut pkl)?;
        let root = unpickle(&pkl)?;
        let Val::Dict(items) = root else { bail!("checkpoint root is not a dict") };
        let mut tensors = HashMap::new();
        for (k, v) in items {
            if let (Val::Str(k), Val::Tensor { key, dtype, offset, size }) = (k, v) {
                let k = k.strip_prefix("module.").unwrap_or(&k).to_string();
                tensors.insert(k, TensorInfo { key, dtype, offset, size });
            }
        }
        Ok(Self { prefix, zip: std::cell::RefCell::new(zip), tensors })
    }

    fn f32_vec(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let t = self.tensors.get(name).ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        ensure!(t.dtype == "FloatStorage", "{name}: unsupported storage {}", t.dtype);
        let numel: usize = t.size.iter().product();
        let mut z = self.zip.borrow_mut();
        let mut f = z.by_name(&format!("{}/data/{}", self.prefix, t.key))?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        let start = t.offset * 4;
        ensure!(bytes.len() >= start + numel * 4, "{name}: storage too small");
        let v = bytes[start..start + numel * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok((t.size.clone(), v))
    }

    fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>> {
        let (shape, data) = self.f32_vec(name)?;
        ensure!(shape.len() == D, "{name}: expected {D} dims, got {shape:?}");
        Ok(Tensor::<D>::from_data(TensorData::new(data, shape), device))
    }
}

fn unpickle(data: &[u8]) -> Result<Val> {
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let mut p = 0usize;
    macro_rules! take {
        ($n:expr) => {{
            let n = $n;
            ensure!(p + n <= data.len(), "pickle truncated");
            let s = &data[p..p + n];
            p += n;
            s
        }};
    }
    fn pop_mark(stack: &mut Vec<Val>) -> Result<Vec<Val>> {
        let i = stack.iter().rposition(|v| matches!(v, Val::Mark)).ok_or_else(|| anyhow!("no MARK"))?;
        let items = stack.split_off(i + 1);
        stack.pop();
        Ok(items)
    }
    loop {
        let op = *data.get(p).ok_or_else(|| anyhow!("pickle: unexpected end"))?;
        p += 1;
        match op {
            0x80 => {
                take!(1);
            } // PROTO
            b'.' => break,
            b'N' => stack.push(Val::None),
            0x88 | 0x89 => stack.push(Val::None), // NEWTRUE / NEWFALSE(値は使わない)
            b'(' => stack.push(Val::Mark),
            b')' => stack.push(Val::Tuple(vec![])),
            b']' => stack.push(Val::List(vec![])),
            b'}' => stack.push(Val::Dict(vec![])),
            b'K' => stack.push(Val::Int(take!(1)[0] as i64)),
            b'M' => stack.push(Val::Int(u16::from_le_bytes(take!(2).try_into().unwrap()) as i64)),
            b'J' => stack.push(Val::Int(i32::from_le_bytes(take!(4).try_into().unwrap()) as i64)),
            0x8a => {
                let n = take!(1)[0] as usize; // LONG1
                let b = take!(n);
                let mut v: i64 = 0;
                for (i, x) in b.iter().enumerate().take(8) {
                    v |= (*x as i64) << (8 * i);
                }
                stack.push(Val::Int(v));
            }
            b'X' => {
                let n = u32::from_le_bytes(take!(4).try_into().unwrap()) as usize;
                stack.push(Val::Str(String::from_utf8(take!(n).to_vec())?));
            }
            0x8c => {
                let n = take!(1)[0] as usize; // SHORT_BINUNICODE
                stack.push(Val::Str(String::from_utf8(take!(n).to_vec())?));
            }
            b'c' => {
                let rest = &data[p..];
                let nl1 = rest.iter().position(|&b| b == b'\n').ok_or_else(|| anyhow!("GLOBAL"))?;
                let nl2 = rest[nl1 + 1..].iter().position(|&b| b == b'\n').ok_or_else(|| anyhow!("GLOBAL"))?;
                let module = std::str::from_utf8(&rest[..nl1])?;
                let name = std::str::from_utf8(&rest[nl1 + 1..nl1 + 1 + nl2])?;
                stack.push(Val::Global(format!("{module} {name}")));
                p += nl1 + 1 + nl2 + 1;
            }
            b'q' => {
                let k = take!(1)[0] as u32;
                memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("BINPUT"))?);
            }
            b'r' => {
                let k = u32::from_le_bytes(take!(4).try_into().unwrap());
                memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("LONG_BINPUT"))?);
            }
            0x94 => {
                let k = memo.len() as u32; // MEMOIZE
                memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("MEMOIZE"))?);
            }
            b'h' => {
                let k = take!(1)[0] as u32;
                stack.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("BINGET {k}"))?);
            }
            b'j' => {
                let k = u32::from_le_bytes(take!(4).try_into().unwrap());
                stack.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("LONG_BINGET {k}"))?);
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Val::Tuple(items));
            }
            0x85 | 0x86 | 0x87 => {
                let n = (op - 0x84) as usize;
                let items = stack.split_off(stack.len() - n);
                stack.push(Val::Tuple(items));
            }
            b'a' => {
                let v = stack.pop().ok_or_else(|| anyhow!("APPEND"))?;
                if let Some(Val::List(l)) = stack.last_mut() {
                    l.push(v);
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                if let Some(Val::List(l)) = stack.last_mut() {
                    l.extend(items);
                }
            }
            b's' => {
                let v = stack.pop().ok_or_else(|| anyhow!("SETITEM"))?;
                let k = stack.pop().ok_or_else(|| anyhow!("SETITEM"))?;
                if let Some(Val::Dict(d)) = stack.last_mut() {
                    d.push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if let Some(Val::Dict(d)) = stack.last_mut() {
                    for kv in items.chunks_exact(2) {
                        d.push((kv[0].clone(), kv[1].clone()));
                    }
                }
            }
            b'b' => {
                stack.pop(); // BUILD: 状態(OrderedDict の _metadata など)は無視
            }
            b'Q' => {
                // BINPERSID: ('storage', <Global FloatStorage>, key, location, numel)
                let pid = stack.pop().ok_or_else(|| anyhow!("BINPERSID"))?;
                let Val::Tuple(t) = pid else { bail!("unexpected persistent id") };
                match (t.first(), t.get(1), t.get(2)) {
                    (Some(Val::Str(s)), Some(Val::Global(g)), Some(Val::Str(key))) if s == "storage" => {
                        let dtype = g.split(' ').nth(1).unwrap_or("").to_string();
                        stack.push(Val::Storage { key: key.clone(), dtype });
                    }
                    _ => bail!("unsupported persistent id {t:?}"),
                }
            }
            b'R' => {
                let args = stack.pop().ok_or_else(|| anyhow!("REDUCE"))?;
                let f = stack.pop().ok_or_else(|| anyhow!("REDUCE"))?;
                let (Val::Global(g), Val::Tuple(a)) = (f, args) else { bail!("unsupported REDUCE") };
                match g.as_str() {
                    "collections OrderedDict" => stack.push(Val::Dict(vec![])),
                    "torch._utils _rebuild_tensor_v2" => {
                        let (Some(Val::Storage { key, dtype }), Some(Val::Int(off)), Some(Val::Tuple(size))) =
                            (a.first(), a.get(1), a.get(2))
                        else {
                            bail!("bad _rebuild_tensor_v2 args");
                        };
                        let size = size
                            .iter()
                            .map(|v| if let Val::Int(i) = v { *i as usize } else { 0 })
                            .collect();
                        stack.push(Val::Tensor {
                            key: key.clone(),
                            dtype: dtype.clone(),
                            offset: *off as usize,
                            size,
                        });
                    }
                    other => bail!("unsupported pickle callable {other}"),
                }
            }
            other => bail!("unsupported pickle opcode 0x{other:02x}"),
        }
    }
    stack.pop().ok_or_else(|| anyhow!("empty pickle"))
}
