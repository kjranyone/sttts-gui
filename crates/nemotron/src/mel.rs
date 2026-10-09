//! 対数メルスペクトログラム(HF `NemotronAsrStreamingFeatureExtractor` の numpy 再現の移植)。
//!
//! プリエンファシス(0.97)は STFT の前に生波形へ適用し、STFT は n_fft=512 / hop=160 /
//! 窓 400 の Hann(対称、n_fft の中央に配置)、パワースペクトル → slaney メル 128 本 →
//! `ln(mel + 2^-24)`。正規化はしない。

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 512;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;
const PREEMPHASIS: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const N_BINS: usize = N_FFT / 2 + 1;

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;

fn hz_to_mel(f: f64) -> f64 {
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= MIN_LOG_HZ { min_log_mel + (f / MIN_LOG_HZ).ln() / logstep } else { f / F_SP }
}

fn mel_to_hz(m: f64) -> f64 {
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel { MIN_LOG_HZ * (logstep * (m - min_log_mel)).exp() } else { F_SP * m }
}

/// 疎な三角フィルタ(非ゼロ区間のみ保持)
struct Filter {
    start: usize,
    w: Vec<f32>,
}

pub struct LogMel {
    window: [f32; N_FFT],
    filters: Vec<Filter>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl LogMel {
    pub fn new() -> Self {
        // np.hanning(400) = 0.5 - 0.5 cos(2πn/(N-1)) を n_fft の中央に置く
        let mut window = [0f32; N_FFT];
        let pad = (N_FFT - WIN_LENGTH) / 2;
        for n in 0..WIN_LENGTH {
            let v = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN_LENGTH - 1) as f64).cos();
            window[pad + n] = v as f32;
        }
        // librosa.filters.mel(norm="slaney") 相当
        let sr = SAMPLE_RATE as f64;
        let fft_freqs: Vec<f64> = (0..N_BINS).map(|i| i as f64 * (sr / 2.0) / (N_BINS - 1) as f64).collect();
        let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
        let mel_f: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
        let filters = (0..N_MELS)
            .map(|i| {
                let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
                let (d0, d1) = (mel_f[i + 1] - mel_f[i], mel_f[i + 2] - mel_f[i + 1]);
                let w: Vec<f64> = fft_freqs
                    .iter()
                    .map(|&f| {
                        let lower = (f - mel_f[i]) / d0;
                        let upper = (mel_f[i + 2] - f) / d1;
                        lower.min(upper).max(0.0) * enorm
                    })
                    .collect();
                let start = w.iter().position(|&x| x > 0.0).unwrap_or(0);
                let end = w.iter().rposition(|&x| x > 0.0).map_or(start, |e| e + 1);
                Filter { start, w: w[start..end].iter().map(|&x| x as f32).collect() }
            })
            .collect();
        Self { window, filters, fft: RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT) }
    }

    /// `pcm`: 16 kHz モノラル。`(行優先の特徴量, フレーム数)` を返す(特徴量は `フレーム数 × N_MELS`)。
    ///
    /// `center=true`(先頭チャンク)は両端に `n_fft/2` のゼロを足す。
    /// `center=false`(後続チャンク)は足さない。
    pub fn compute(&self, pcm: &[f32], center: bool) -> (Vec<f32>, usize) {
        let mut x: Vec<f32> = Vec::with_capacity(pcm.len() + N_FFT);
        if center {
            x.resize(N_FFT / 2, 0.0);
        }
        if let Some(&first) = pcm.first() {
            x.push(first);
            x.extend(pcm.windows(2).map(|w| w[1] - PREEMPHASIS * w[0]));
        }
        if center {
            x.resize(x.len() + N_FFT / 2, 0.0);
        }
        if x.len() < N_FFT {
            return (Vec::new(), 0);
        }
        let frames = 1 + (x.len() - N_FFT) / HOP_LENGTH;
        let mut out = vec![0f32; frames * N_MELS];
        let mut inbuf = self.fft.make_input_vec();
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = [0f32; N_BINS];
        for f in 0..frames {
            let seg = &x[f * HOP_LENGTH..f * HOP_LENGTH + N_FFT];
            for ((d, &s), &w) in inbuf.iter_mut().zip(seg).zip(&self.window) {
                *d = s * w;
            }
            self.fft.process_with_scratch(&mut inbuf, &mut spec, &mut scratch).expect("fft");
            for (p, c) in power.iter_mut().zip(&spec) {
                *p = c.re * c.re + c.im * c.im;
            }
            let row = &mut out[f * N_MELS..(f + 1) * N_MELS];
            for (o, flt) in row.iter_mut().zip(&self.filters) {
                let m: f32 = flt.w.iter().zip(&power[flt.start..]).map(|(a, b)| a * b).sum();
                *o = (m + LOG_ZERO_GUARD).ln();
            }
        }
        (out, frames)
    }
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}
