//! テキスト(+キャプション+参照音声)から音声を作る一連の処理。
//! PyTorch 実装 `irodori_tts.inference_runtime.InferenceRuntime.synthesize`(MeanFlow)に対応する。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use rand::RngCore;

use crate::codec::{DEFAULT_NORMALIZE_DB, DacVae};
use crate::condition::{TextConditioner, caption_inputs};
use crate::config::ModelConfig;
use crate::dit::{Conditions, Dit};
use crate::duration::{DurationInputs, DurationPredictor, frames_from_log};
use crate::sampler::{find_flattening_point, sample_euler_meanflow, unpatchify_latent};
use crate::text::normalize_text;
use crate::tokenizer::Tokenizer;
use crate::watermark::{IRODORI_PAYLOAD, Watermarker};
use crate::weights::Weights;

/// 読み込むファイルの場所
#[derive(Debug, Clone)]
pub struct TtsPaths {
    /// `model.safetensors` と `tokenizer/tokenizer.json` があるディレクトリ
    pub model_dir: PathBuf,
    /// DACVAE の `weights.pth`
    pub codec_weights: PathBuf,
    /// SilentCipher の `44_1_khz/73999_iteration`(None なら透かしなし)
    pub watermark_dir: Option<PathBuf>,
}

impl TtsPaths {
    /// HuggingFace キャッシュ(`~/.cache/huggingface/hub`)から既定のモデルを探す
    pub fn from_hf_cache() -> Result<Self> {
        let model_dir = crate::testing::checkpoint_dir().context("Irodori-TTS v4.1 Small MF が HF キャッシュにありません")?;
        let codec_weights = crate::codec::default_weights_path().context("DACVAE の weights.pth が HF キャッシュにありません")?;
        let watermark_dir = crate::testing::hf_snapshot("models--sony--silentcipher")
            .map(|d| d.join("44_1_khz/73999_iteration"))
            .filter(|d| d.is_dir());
        Ok(Self { model_dir, codec_weights, watermark_dir })
    }
}

/// 1 回の合成の指定(`SamplingRequest` の MeanFlow で意味のある項目)
#[derive(Debug, Clone)]
pub struct SamplingRequest {
    pub text: String,
    pub caption: Option<String>,
    /// 参照音声(WAV / FLAC)。`no_ref` が false のとき必要
    pub ref_wav: Option<PathBuf>,
    pub no_ref: bool,
    /// None なら乱数で決め、`SynthResult::used_seed` に返す(PyTorch とは別の乱数列)
    pub seed: Option<u64>,
    pub num_steps: usize,
    pub duration_scale: f64,
    /// 長さを秒で指定(長さ予測を使わない)
    pub seconds: Option<f64>,
    pub min_seconds: f64,
    pub max_seconds: f64,
    pub max_ref_seconds: Option<f64>,
    pub ref_normalize_db: Option<f32>,
    pub ref_ensure_max: bool,
    pub trim_tail: bool,
    pub tail_window_size: usize,
    pub tail_std_threshold: f32,
    pub tail_mean_threshold: f32,
    /// SilentCipher の透かしを入れる(使えるときのみ)
    pub watermark: bool,
}

impl Default for SamplingRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            caption: None,
            ref_wav: None,
            no_ref: false,
            seed: None,
            num_steps: 4,
            duration_scale: 1.0,
            seconds: None,
            min_seconds: 0.5,
            max_seconds: 30.0,
            max_ref_seconds: None,
            ref_normalize_db: Some(DEFAULT_NORMALIZE_DB),
            ref_ensure_max: true,
            trim_tail: true,
            tail_window_size: 20,
            tail_std_threshold: 0.05,
            tail_mean_threshold: 0.1,
            watermark: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthResult {
    /// モノラル f32、`sample_rate` Hz
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub used_seed: u64,
    pub messages: Vec<String>,
    /// (段階名, 秒)
    pub timings: Vec<(String, f64)>,
}

/// 合成の途中成果(参照との段階比較用)
#[derive(Default)]
pub struct Trace {
    pub text_state: Option<Tensor<3>>,
    pub speaker_state: Option<Tensor<3>>,
    pub caption_state: Option<Tensor<3>>,
    pub duration_log_frames: Option<f32>,
    pub latent_steps: usize,
    pub latent: Option<Tensor<3>>,
    pub raw_audio: Vec<f32>,
}

pub struct Tts {
    pub cfg: ModelConfig,
    tokenizer: Tokenizer,
    cond: TextConditioner,
    dit: Dit,
    duration: Option<DurationPredictor>,
    codec: DacVae,
    watermarker: Option<Watermarker>,
    device: Device,
}

impl Tts {
    pub fn load(paths: &TtsPaths, device: &Device) -> Result<Self> {
        let w = Weights::open(paths.model_dir.join("model.safetensors"))?;
        let cond = TextConditioner::load(&w, device)?;
        let cfg = cond.cfg.clone();
        ensure!(cfg.is_meanflow(), "MeanFlow のチェックポイントのみ対応です(flow_parameterization={})", cfg.flow_parameterization);
        let tokenizer = Tokenizer::load(paths.model_dir.join("tokenizer"))?;
        let dit = Dit::load(&w, &cfg, device)?;
        let duration = if cfg.use_duration_predictor { Some(DurationPredictor::load(&w, &cfg, device)?) } else { None };
        let codec = DacVae::load(&paths.codec_weights, device)?;
        let watermarker = match &paths.watermark_dir {
            Some(d) => Some(Watermarker::load(d, device)?),
            None => None,
        };
        Ok(Self { cfg, tokenizer, cond, dit, duration, codec, watermarker, device: device.clone() })
    }

    pub fn sample_rate(&self) -> u32 {
        self.codec.sample_rate
    }

    pub fn has_watermark(&self) -> bool {
        self.watermarker.is_some()
    }

    pub fn synthesize(&self, req: &SamplingRequest) -> Result<SynthResult> {
        self.synthesize_traced(req, None, None)
    }

    /// `noise` を渡すと初期ノイズに使う(参照と比較するとき)。`trace` に途中成果を残す。
    pub fn synthesize_traced(
        &self,
        req: &SamplingRequest,
        noise: Option<Tensor<3>>,
        mut trace: Option<&mut Trace>,
    ) -> Result<SynthResult> {
        let mut messages = Vec::new();
        let mut timings: Vec<(String, f64)> = Vec::new();
        let mut lap = |name: &str, t0: &mut Instant| {
            timings.push((name.to_string(), t0.elapsed().as_secs_f64()));
            *t0 = Instant::now();
        };
        ensure!(req.num_steps > 0, "num_steps must be > 0");
        ensure!(req.duration_scale > 0.0, "duration_scale must be > 0");
        ensure!(req.min_seconds > 0.0 && req.max_seconds >= req.min_seconds, "bad min/max seconds");
        let mut t0 = Instant::now();

        // --- テキスト・キャプション
        let text = normalize_text(&req.text).trim().to_string();
        ensure!(!text.is_empty(), "text became empty after normalization.");
        let (ids, mask) = self.tokenizer.batch_encode(&[text.clone()], self.cfg.max_text_len, self.cfg.text_add_bos)?;
        let has_caption_text = self.cfg.use_caption_condition && req.caption.as_deref().is_some_and(|c| !c.trim().is_empty());
        let (cids, cmask) = caption_inputs(
            &self.tokenizer,
            req.caption.as_deref(),
            1,
            self.cond.max_caption_len(),
            self.cond.caption_add_bos(),
        )?;
        let text_state = self.cond.encode_text(&ids, &mask);
        let caption_state = self.cond.encode_caption(&cids, &cmask);
        let text_mask = mask_tensor(&mask, &self.device);
        let caption_mask = mask_tensor(&cmask, &self.device);
        lap("text_conditions", &mut t0);

        // --- 話者(参照音声)
        let (speaker_state, speaker_mask, has_speaker) = if self.dit.has_speaker_condition() {
            if req.no_ref {
                let (s, m) = self.dit.encode_speaker(None, 1)?;
                (Some(s), Some(m), false)
            } else {
                let path = req.ref_wav.as_deref().context("参照音声(ref_wav)を指定するか、no_ref を true にしてください")?;
                let latent = self.encode_reference(path, req, &mut messages)?;
                let steps = latent.dims()[1];
                ensure!(steps > 0, "Reference latent length became zero.");
                let mask = Tensor::<2>::ones([1, steps], &self.device);
                let (s, m) = self.dit.encode_speaker(Some((latent, mask)), 1)?;
                (Some(s), Some(m), true)
            }
        } else {
            (None, None, false)
        };
        lap("prepare_reference", &mut t0);

        // --- 長さ
        let hop = self.codec.hop;
        let sr = self.codec.sample_rate;
        let (latent_steps, target_samples) = if let Some(sec) = req.seconds {
            ensure!(sec > 0.0, "seconds must be > 0 when provided");
            let clamped = sec.clamp(req.min_seconds, req.max_seconds);
            if (clamped - sec).abs() > f64::EPSILON {
                messages.push(format!("warning: manual duration {sec:.3}s was clamped to {clamped:.3}s."));
            }
            let target = ((clamped * sr as f64) as usize).max(1);
            (target.div_ceil(hop), target)
        } else if let Some(dp) = &self.duration {
            let speaker = speaker_state.clone().unwrap_or_else(|| Tensor::<3>::zeros([1, 1, 768], &self.device));
            let caption = caption_state.clone().unwrap_or_else(|| Tensor::<3>::zeros([1, 1, self.cfg.caption_dim.unwrap_or(512)], &self.device));
            let inputs = DurationInputs {
                text_state: text_state.clone(),
                text_mask: text_mask.clone(),
                speaker_state: speaker,
                has_speaker: Tensor::<1>::from_data(TensorData::new(vec![if has_speaker { 1.0f32 } else { 0.0 }], vec![1]), &self.device),
                caption_state: caption,
                caption_mask: caption_mask.clone(),
                has_caption: Tensor::<1>::from_data(TensorData::new(vec![if has_caption_text { 1.0f32 } else { 0.0 }], vec![1]), &self.device),
            };
            let pred = dp.predict_log_frames(&inputs).into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0];
            if let Some(t) = trace.as_deref_mut() {
                t.duration_log_frames = Some(pred);
            }
            let frames = frames_from_log(pred, req.duration_scale, req.min_seconds, req.max_seconds, sr as usize, hop);
            messages.push(format!("info: predicted duration frames={:.1}, using_frames={frames} ({:.3}s).", pred.exp_m1(), (frames * hop) as f64 / sr as f64));
            (frames, frames * hop)
        } else {
            let target = 30 * sr as usize;
            (target.div_ceil(hop), target)
        };
        lap("predict_duration", &mut t0);

        // --- サンプリング
        let used_seed = req.seed.unwrap_or_else(|| rand::rng().next_u64() >> 1);
        let cond = Conditions {
            text_state: text_state.clone(),
            text_mask,
            speaker_state: speaker_state.clone(),
            speaker_mask,
            caption_state: caption_state.clone(),
            caption_mask: if caption_state.is_some() { Some(caption_mask) } else { None },
        };
        let patch = self.cfg.latent_patch_size;
        let patched_steps = latent_steps.div_ceil(patch);
        let z_patched = sample_euler_meanflow(&self.dit, &cond, patched_steps, req.num_steps, noise, used_seed)?;
        let z = unpatchify_latent(z_patched, patch, self.cfg.latent_dim);
        let z = z.narrow(1, 0, latent_steps);
        let z_host: Vec<f32> = z.clone().into_data().convert::<f32>().try_to_vec::<f32>().unwrap();
        lap("sample_meanflow", &mut t0);

        // --- デコード・末尾トリム
        let audio_t = self.codec.decode_latent(z.clone());
        let mut audio: Vec<f32> = audio_t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap();
        let mut max_samples = target_samples.min(audio.len());
        if req.trim_tail {
            let fp = find_flattening_point(
                &z_host,
                latent_steps,
                self.cfg.latent_dim,
                req.tail_window_size.max(1),
                req.tail_std_threshold,
                req.tail_mean_threshold,
            );
            let flat = fp * hop;
            if flat > 0 {
                max_samples = max_samples.min(flat);
            }
        }
        audio.truncate(max_samples);
        lap("decode_latent", &mut t0);

        if let Some(t) = trace.as_deref_mut() {
            t.text_state = Some(text_state);
            t.speaker_state = speaker_state;
            t.caption_state = caption_state;
            t.latent_steps = latent_steps;
            t.latent = Some(z);
            t.raw_audio = audio.clone();
        }

        // --- 透かし
        if req.watermark {
            if let Some(wm) = &self.watermarker {
                audio = wm.encode(&audio, sr, &IRODORI_PAYLOAD)?;
                lap("silentcipher_watermark", &mut t0);
            } else {
                messages.push("warning: SilentCipher watermark is unavailable; generated audio was not watermarked.".into());
            }
        }

        Ok(SynthResult { audio, sample_rate: sr, used_seed, messages, timings })
    }

    /// 参照音声 → DACVAE 潜在 `[1, T, latent_dim]`(最大長でトリム)
    fn encode_reference(&self, path: &Path, req: &SamplingRequest, messages: &mut Vec<String>) -> Result<Tensor<3>> {
        let max_ref_seconds = req.max_ref_seconds.or(self.cfg.ref_max_seconds).unwrap_or(30.0);
        let (channels, sr) = load_audio(path)?;
        let n = channels[0].len();
        ensure!(n > 0, "empty reference audio: {}", path.display());
        let mut keep = n;
        if max_ref_seconds > 0.0 {
            let max_samples = ((max_ref_seconds * sr as f64) as usize).max(1);
            if n > max_samples {
                messages.push(format!(
                    "warning: reference audio exceeds max_ref_seconds ({max_ref_seconds}s). Trimming from {:.2}s to {:.2}s.",
                    n as f64 / sr as f64,
                    max_samples as f64 / sr as f64
                ));
                keep = max_samples;
            }
        }
        let c = channels.len();
        let mut flat = Vec::with_capacity(c * keep);
        for ch in &channels {
            flat.extend_from_slice(&ch[..keep]);
        }
        let wav = Tensor::<3>::from_data(TensorData::new(flat, vec![1, c, keep]), &self.device);
        let mut latent = self.codec.encode_waveform(wav, sr, req.ref_normalize_db, req.ref_ensure_max)?;
        if max_ref_seconds > 0.0 {
            let max_steps = ((max_ref_seconds * self.codec.sample_rate as f64 / self.codec.hop as f64).ceil() as usize).max(1);
            let t = latent.dims()[1];
            if t > max_steps {
                messages.push(format!("warning: reference latent steps ({t}) exceed max_ref_seconds bound ({max_steps} steps). Trimming."));
                latent = latent.narrow(1, 0, max_steps);
            }
        }
        Ok(latent)
    }
}

fn mask_tensor(mask: &[Vec<bool>], device: &Device) -> Tensor<2> {
    let b = mask.len();
    let s = mask.first().map_or(0, Vec::len);
    let data: Vec<f32> = mask.iter().flat_map(|r| r.iter().map(|&x| if x { 1.0 } else { 0.0 })).collect();
    Tensor::<2>::from_data(TensorData::new(data, vec![b, s]), device)
}

/// WAV / FLAC を読み、チャンネルごとの f32 サンプルとサンプルレートを返す。
pub fn load_audio(path: &Path) -> Result<(Vec<Vec<f32>>, u32)> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "wav" => {
            let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
            let spec = r.spec();
            let ch = spec.channels as usize;
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
                hound::SampleFormat::Int => {
                    let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
                    r.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
                }
            };
            Ok((deinterleave(&samples, ch), spec.sample_rate))
        }
        "flac" => {
            let mut r = claxon::FlacReader::open(path).with_context(|| format!("open {}", path.display()))?;
            let info = r.streaminfo();
            let scale = 1.0 / (1u64 << (info.bits_per_sample - 1)) as f32;
            let samples: Vec<f32> = r.samples().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?;
            Ok((deinterleave(&samples, info.channels as usize), info.sample_rate))
        }
        other => bail!("unsupported reference audio format: .{other} (wav / flac)"),
    }
}

fn deinterleave(samples: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let channels = channels.max(1);
    let mut out = vec![Vec::with_capacity(samples.len() / channels); channels];
    for frame in samples.chunks_exact(channels) {
        for (c, v) in frame.iter().enumerate() {
            out[c].push(*v);
        }
    }
    out
}
