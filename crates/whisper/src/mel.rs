//! ログメルスペクトログラム(transformers の `WhisperFeatureExtractor` 互換)。
//!
//! 30 秒に零詰め/切り詰め → 反射パディング付き STFT(n_fft 400, hop 160, periodic Hann)→ パワー →
//! メルフィルタ(slaney, 128 本)→ log10(下限 1e-10)→ `max - 8` で下限クランプ → `(x + 4) / 4`。
//! transformers の numpy 実装と同じく倍精度で計算し、最後に f32 へ落とす。

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
/// 1 窓 = 30 秒
pub const N_SAMPLES: usize = 30 * SAMPLE_RATE;
/// 1 窓のメルフレーム数
pub const N_FRAMES: usize = N_SAMPLES / HOP;
const N_BINS: usize = N_FFT / 2 + 1;

pub struct MelExtractor {
    /// `[N_MELS][N_BINS]`
    filters: Vec<f64>,
    window: Vec<f64>,
    fft: Arc<dyn RealToComplex<f64>>,
}

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;

fn logstep() -> f64 {
    (6.4f64).ln() / 27.0
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz >= MIN_LOG_HZ { MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep() } else { hz / F_SP }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MIN_LOG_MEL { MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * logstep()).exp() } else { F_SP * mel }
}

/// `mel_filter_bank(.., norm="slaney", mel_scale="slaney")` 相当。`[N_MELS][N_BINS]`
fn mel_filters() -> Vec<f64> {
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
    let freqs: Vec<f64> =
        (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
    let fft_freqs: Vec<f64> =
        (0..N_BINS).map(|i| (SAMPLE_RATE / 2) as f64 * i as f64 / (N_BINS - 1) as f64).collect();
    let mut out = vec![0f64; N_MELS * N_BINS];
    for m in 0..N_MELS {
        let enorm = 2.0 / (freqs[m + 2] - freqs[m]);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let down = -(freqs[m] - f) / (freqs[m + 1] - freqs[m]);
            let up = (freqs[m + 2] - f) / (freqs[m + 2] - freqs[m + 1]);
            out[m * N_BINS + k] = down.min(up).max(0.0) * enorm;
        }
    }
    out
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MelExtractor {
    pub fn new() -> Self {
        let window = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
            .collect();
        Self { filters: mel_filters(), window, fft: RealFftPlanner::<f64>::new().plan_fft_forward(N_FFT) }
    }

    /// 16kHz モノ(最大 30 秒。長ければ切り詰め)→ `[N_MELS * N_FRAMES]`(メル軸が外側)
    pub fn log_mel(&self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len().min(N_SAMPLES);
        let mut x = vec![0f64; N_SAMPLES];
        for (d, &s) in x.iter_mut().zip(&audio[..n]) {
            *d = f64::from(s);
        }
        // center=True の反射パディング(両端 n_fft/2)
        let pad = N_FFT / 2;
        let mut p = Vec::with_capacity(N_SAMPLES + 2 * pad);
        p.extend((1..=pad).rev().map(|i| x[i]));
        p.extend_from_slice(&x);
        p.extend((0..pad).map(|i| x[N_SAMPLES - 2 - i]));

        let mut mel = vec![0f64; N_MELS * N_FRAMES];
        let mut buf = vec![0f64; N_FFT];
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0f64; N_BINS];
        for t in 0..N_FRAMES {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = p[t * HOP + i] * self.window[i];
            }
            self.fft.process_with_scratch(&mut buf, &mut spec, &mut scratch).expect("fft");
            for (pw, c) in power.iter_mut().zip(&spec) {
                *pw = c.re * c.re + c.im * c.im;
            }
            for m in 0..N_MELS {
                let f = &self.filters[m * N_BINS..(m + 1) * N_BINS];
                mel[m * N_FRAMES + t] = f.iter().zip(&power).map(|(a, b)| a * b).sum();
            }
        }
        let mut max = f64::NEG_INFINITY;
        for v in &mut mel {
            *v = v.max(1e-10).log10();
            max = max.max(*v);
        }
        mel.iter().map(|&v| ((v.max(max - 8.0) + 4.0) / 4.0) as f32).collect()
    }
}
