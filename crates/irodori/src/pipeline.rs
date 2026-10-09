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
use crate::sampler::{find_flattening_point, sample_euler_meanflow_padded, unpatchify_latent};
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

/// Irodori-TTS v4.1 Small MF のモデルと、組み合わせるコーデック・透かしの HF リポジトリ
pub const MODEL_REPO: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
pub const CODEC_REPO: &str = "Aratako/Semantic-DACVAE-Japanese-32dim";
pub const WATERMARK_REPO: &str = "sony/silentcipher";

impl TtsPaths {
    /// HF キャッシュに無いモデルをダウンロードして(あれば再利用して)パスを返す。
    pub fn ensure_downloaded(progress: &dyn Fn(&str)) -> Result<Self> {
        let model_dir = crate::hub::ensure_files(MODEL_REPO, &["model.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"], progress)?;
        let codec_dir = crate::hub::ensure_files(CODEC_REPO, &["weights.pth"], progress)?;
        let wm = crate::hub::ensure_files(
            WATERMARK_REPO,
            &["44_1_khz/73999_iteration/enc_c.ckpt", "44_1_khz/73999_iteration/dec_c.ckpt", "44_1_khz/73999_iteration/hparams.yaml"],
            progress,
        );
        // 透かしが取れなくても合成は動く(未取得の警告つき)。取れたときだけ使う
        let watermark_dir = wm.ok().map(|d| d.join("44_1_khz/73999_iteration"));
        Ok(Self { model_dir, codec_weights: codec_dir.join("weights.pth"), watermark_dir })
    }

    /// HuggingFace キャッシュ(`~/.cache/huggingface/hub`)から既定のモデルを探す
    pub fn from_hf_cache() -> Result<Self> {
        let model_dir = crate::hub::find_snapshot("models--Aratako--Irodori-TTS-v4.1-Small-MF", "model.safetensors").context("Irodori-TTS v4.1 Small MF が HF キャッシュにありません")?;
        let codec_weights = crate::codec::default_weights_path().context("DACVAE の weights.pth が HF キャッシュにありません")?;
        let watermark_dir = crate::hub::find_snapshot("models--sony--silentcipher", "44_1_khz/73999_iteration/enc_c.ckpt")
            .map(|d| d.join("44_1_khz/73999_iteration"));
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
    /// 符号化済みの参照潜在 `[1, T, latent_dim]`(`Tts::encode_reference` の結果)。指定すると `ref_wav` の読み込みと
    /// 符号化を省く(同じ声を繰り返し使うときのキャッシュ用)。
    pub ref_latent: Option<Tensor<3>>,
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
            ref_latent: None,
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

/// コーデックの 1 窓の長さ(潜在フレーム。25 フレーム = 1 秒)と、前後に付ける文脈(受容野の確保に 8 以上が必要)
const DECODE_WINDOW: usize = 25;
const DECODE_CONTEXT: usize = 8;

/// ウォームアップ用の文(トークン数・音声長・奇偶の違う長さを散らす)
const WARMUP_TEXTS: &[&str] = &[
    "あ。",
    "こんにちは。",
    "今日はいい天気ですね。",
    "えーと、明日の会議の資料を準備しておいてください。",
    "これは少し長めの文章で、途中に読点を含みながら最後まで読み上げられるかを確認します。",
];

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
        // 参照音声の潜在をパッチ化する処理は未実装(v4.1 Small MF は 1)
        ensure!(cfg.latent_patch_size == 1, "latent_patch_size != 1 は未対応です({})", cfg.latent_patch_size);
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

    /// GPU のカーネルを先にコンパイルしておく(長さの違うダミー発話を数回合成して捨てる)。
    /// GPU は形状の整列クラスごとにカーネルを作るため、起動直後の数発話は 1〜数秒余計にかかる。
    /// アプリ起動時にバックグラウンドで呼ぶと、利用者の最初の発話から定常の速度になる。
    pub fn warmup(&self) -> Result<()> {
        for text in WARMUP_TEXTS {
            let req = SamplingRequest { text: (*text).to_string(), no_ref: true, seed: Some(0), ..Default::default() };
            self.synthesize(&req)?;
        }
        Ok(())
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
        let trace_stages = std::env::var_os("IRODORI_TRACE").is_some();
        let mut lap = |name: &str, t0: &mut Instant| {
            let sec = t0.elapsed().as_secs_f64();
            if trace_stages {
                eprintln!("[irodori] {name}: {sec:.3}s");
            }
            timings.push((name.to_string(), sec));
            *t0 = Instant::now();
        };
        ensure!(req.num_steps > 0, "num_steps must be > 0");
        ensure!(req.duration_scale > 0.0, "duration_scale must be > 0");
        ensure!(req.min_seconds > 0.0 && req.max_seconds >= req.min_seconds, "bad min/max seconds");
        let mut t0 = Instant::now();

        // --- テキスト・キャプション
        let text = normalize_text(&req.text).trim().to_string();
        ensure!(!text.is_empty(), "text became empty after normalization.");
        let (ids, mask) = self.tokenizer.batch_encode(std::slice::from_ref(&text), self.cfg.max_text_len, self.cfg.text_add_bos)?;
        // パディングは右詰めで、無効トークンは注意から完全に除外される(マスク -1e9 → 確率 0)ので、
        // 有効な先頭部分だけで計算しても結果は同じ。256 トークン固定のまま回すより桁違いに速い。
        // ただし長さは段階(バケット)に揃えてパディングする。GPU のカーネルは形状ごとに作られ、初回は
        // 1 本数百 ms かかるので、系列長を数段階に絞ると発話ごとのコンパイルがほぼ無くなる。
        let n_text = mask[0].iter().filter(|&&b| b).count().max(1);
        let nb = bucket(n_text).min(ids[0].len());
        let ids = vec![ids[0][..nb].to_vec()];
        let mask = vec![mask[0][..nb].to_vec()];
        let has_caption_text = self.cfg.use_caption_condition && req.caption.as_deref().is_some_and(|c| !c.trim().is_empty());
        let text_state = self.cond.encode_text(&ids, &mask);
        let text_mask = mask_tensor(&mask, &self.device);
        // キャプションが空なら、原典では全無効マスクで 0 になる(参照の encode_conditions.out4 は 0)ので計算を省く
        let (caption_state, caption_mask) = if has_caption_text {
            let (cids, cmask) = caption_inputs(&self.tokenizer, req.caption.as_deref(), 1, self.cond.max_caption_len(), self.cond.caption_add_bos())?;
            let n = cmask[0].iter().filter(|&&b| b).count().max(1);
            let nb = bucket(n).min(cids[0].len());
            let cids = vec![cids[0][..nb].to_vec()];
            let cmask = vec![cmask[0][..nb].to_vec()];
            let state = self.cond.encode_caption(&cids, &cmask);
            let mask_t = mask_tensor(&cmask, &self.device);
            (state, Some(mask_t))
        } else {
            (None, None)
        };
        if trace_stages {
            // burn は遅延実行。段階ごとの時間を測るため、トレース時だけ読み戻して実行を完了させる
            let _ = text_state.clone().into_data();
            lap("text_encode", &mut t0);
            if let Some(c) = &caption_state {
                let _ = c.clone().into_data();
                lap("caption_encode", &mut t0);
            }
        }
        lap("text_conditions", &mut t0);

        // --- 話者(参照音声)
        let (speaker_state, speaker_mask, has_speaker) = if self.dit.has_speaker_condition() {
            if req.no_ref {
                // 全無効の話者トークンは注意から除外されるので、DiT には渡さない(結果は同じ)
                (None, None, false)
            } else {
                let latent = match &req.ref_latent {
                    Some(l) => l.clone(),
                    None => {
                        let path = req.ref_wav.as_deref().context("参照音声(ref_wav)を指定するか、no_ref を true にしてください")?;
                        self.encode_reference(path, req, &mut messages)?
                    }
                };
                let steps = latent.dims()[1];
                ensure!(steps > 0, "Reference latent length became zero.");
                let mask = Tensor::<2>::ones([1, steps], &self.device);
                let (s, m) = self.dit.encode_speaker(Some((latent, mask)), 1)?;
                (Some(s), Some(m), true)
            }
        } else {
            (None, None, false)
        };
        if let (true, Some(sp)) = (trace_stages, &speaker_state) {
            let _ = sp.clone().into_data();
        }
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
            let speaker = speaker_state
                .clone()
                .unwrap_or_else(|| Tensor::<3>::zeros([1, 1, self.cfg.speaker_dim.unwrap_or(768)], &self.device));
            let (caption, dur_caption_mask) = match (&caption_state, &caption_mask) {
                (Some(c), Some(m)) => (c.clone(), m.clone()),
                _ => (
                    Tensor::<3>::zeros([1, 1, self.cfg.caption_dim.unwrap_or(512)], &self.device),
                    Tensor::<2>::zeros([1, 1], &self.device),
                ),
            };
            let inputs = DurationInputs {
                text_state: text_state.clone(),
                text_mask: text_mask.clone(),
                speaker_state: speaker,
                has_speaker: Tensor::<1>::from_data(TensorData::new(vec![if has_speaker { 1.0f32 } else { 0.0 }], vec![1]), &self.device),
                caption_state: caption,
                caption_mask: dur_caption_mask,
                has_caption: Tensor::<1>::from_data(TensorData::new(vec![if has_caption_text { 1.0f32 } else { 0.0 }], vec![1]), &self.device),
            };
            if trace_stages {
                let _ = inputs.has_caption.clone().into_data();
                lap("  duration inputs upload", &mut t0);
            }
            let pred_t = dp.predict_log_frames(&inputs);
            if trace_stages {
                lap("  duration graph build", &mut t0);
            }
            let pred = to_host(pred_t)?[0];
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
            caption_mask,
        };
        let patch = self.cfg.latent_patch_size;
        let patched_steps = latent_steps.div_ceil(patch);
        let padded_steps = bucket(patched_steps);
        let z_patched = sample_euler_meanflow_padded(&self.dit, &cond, patched_steps, padded_steps, req.num_steps, noise, used_seed)?
            .narrow(1, 0, patched_steps);
        let z = unpatchify_latent(z_patched, patch, self.cfg.latent_dim);
        let z = z.narrow(1, 0, latent_steps);
        let z_host = to_host(z.clone())?;
        lap("sample_meanflow", &mut t0);

        // --- デコード・末尾トリム
        // 窓ごとにデコードする(ピークメモリが一定になり、同じ形状のカーネルを使い回せる。全体版との差は 1e-7 級)
        let audio_t = self.codec.decode_latent_windowed(z.clone(), DECODE_WINDOW, DECODE_CONTEXT);
        let mut audio = to_host(audio_t)?;
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

        if let Some(t) = trace {
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
    pub fn encode_reference(&self, path: &Path, req: &SamplingRequest, messages: &mut Vec<String>) -> Result<Tensor<3>> {
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

/// 系列長を数段階に丸める(1.33〜1.5 倍刻み)。計算量の増加は最大 1.5 倍で、形状の種類は対数で済む。
fn bucket(n: usize) -> usize {
    const STEPS: &[usize] = &[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
    STEPS.iter().copied().find(|&b| b >= n).unwrap_or_else(|| n.div_ceil(256) * 256)
}

/// GPU のテンソルをホストへ読み戻す(デバイスの異常はパニックにせず `Err` で返す)。
fn to_host<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>> {
    t.try_into_data()
        .map_err(|e| anyhow::anyhow!("GPU からの読み戻しに失敗しました: {e:?}"))?
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("GPU からの読み戻しに失敗しました: {e:?}"))
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
