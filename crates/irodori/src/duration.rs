//! 長さ予測(`irodori_tts/duration.py` の `build_duration_features`、`model.py` の
//! `DurationPredictor`(`token_sum_dual_adarn_zero_no_aux`)と `predict_duration_log_frames`、
//! `inference_runtime.py` のフレーム数への変換)。

use anyhow::{Result, bail};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Tensor};
use regex::Regex;

use crate::config::ModelConfig;
use crate::weights::Weights;

// ---------------------------------------------------------------- 特徴量

/// 注釈として許可された絵文字(`ALLOWED_ANNOTATION_EMOJIS`)
pub const ALLOWED_ANNOTATION_EMOJIS: &[&str] = &[
    "\u{23e9}", // ⏩
    "\u{23f1}\u{fe0f}", // ⏱️
    "\u{23f8}\u{fe0f}", // ⏸️
    "\u{1f32c}\u{fe0f}", // 🌬️
    "\u{1f36d}", // 🍭
    "\u{1f39b}\u{fe0f}", // 🎛️
    "\u{1f3ad}", // 🎭
    "\u{1f3b5}", // 🎵
    "\u{1f422}", // 🐢
    "\u{1f431}", // 🐱
    "\u{1f442}", // 👂
    "\u{1f443}", // 👃
    "\u{1f445}", // 👅
    "\u{1f44c}", // 👌
    "\u{1f44f}", // 👏
    "\u{1f48b}", // 💋
    "\u{1f4a5}", // 💥
    "\u{1f4a6}", // 💦
    "\u{1f4aa}", // 💪
    "\u{1f4c4}", // 📄
    "\u{1f4de}", // 📞
    "\u{1f4e2}", // 📢
    "\u{1f4e3}", // 📣
    "\u{1f606}", // 😆
    "\u{1f60a}", // 😊
    "\u{1f60c}", // 😌
    "\u{1f60e}", // 😎
    "\u{1f60f}", // 😏
    "\u{1f612}", // 😒
    "\u{1f616}", // 😖
    "\u{1f61f}", // 😟
    "\u{1f620}", // 😠
    "\u{1f62a}", // 😪
    "\u{1f62d}", // 😭
    "\u{1f62e}", // 😮
    "\u{1f62e}\u{200d}\u{1f4a8}", // 😮‍💨
    "\u{1f630}", // 😰
    "\u{1f631}", // 😱
    "\u{1f632}", // 😲
    "\u{1f634}", // 😴
    "\u{1f644}", // 🙄
    "\u{1f64f}", // 🙏
    "\u{1f910}", // 🤐
    "\u{1f914}", // 🤔
    "\u{1f922}", // 🤢
    "\u{1f927}", // 🤧
    "\u{1f92d}", // 🤭
    "\u{1f964}", // 🥤
    "\u{1f971}", // 🥱
    "\u{1f974}", // 🥴
    "\u{1f975}", // 🥵
    "\u{1f979}", // 🥹
    "\u{1f97a}", // 🥺
    "\u{1fae3}", // 🫣
    "\u{1faf6}", // 🫶
    "\u{1f4d6}", // 📖
];

/// 補助特徴量の次元数
pub const DURATION_FEATURE_DIM: usize = 14;

fn emoji_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // 長いものを先に(ZWJ 付きの 😮‍💨 が 😮 に食われないように)
        let mut v: Vec<&str> = ALLOWED_ANNOTATION_EMOJIS.to_vec();
        v.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        let pat = v.iter().map(|s| regex::escape(s)).collect::<Vec<_>>().join("|");
        Regex::new(&pat).expect("emoji regex")
    })
}

pub fn count_annotation_emojis(text: &str) -> usize {
    emoji_regex().find_iter(text).count()
}

fn log1p_cap(count: usize, cap: usize) -> f64 {
    (count.min(cap) as f64).ln_1p() / (cap as f64).ln_1p()
}

fn log1p_cap_float(value: f64, cap: f64) -> f64 {
    value.clamp(0.0, cap).ln_1p() / cap.ln_1p()
}

fn is_kana(c: char) -> bool {
    let c = c as u32;
    (0x3040..=0x309F).contains(&c) || (0x30A0..=0x30FF).contains(&c)
}

fn is_kanji(c: char) -> bool {
    let c = c as u32;
    (0x3400..=0x4DBF).contains(&c)
        || (0x4E00..=0x9FFF).contains(&c)
        || (0xF900..=0xFAFF).contains(&c)
        || (0x20000..=0x2FA1F).contains(&c)
}

/// 14 次元の補助特徴量(1 文ぶん)。`text` は正規化済みのテキスト、`token_count` はマスクの和。
pub fn build_duration_features(
    text: &str,
    token_count: usize,
    max_text_len: usize,
    has_speaker: bool,
) -> Result<[f32; DURATION_FEATURE_DIM]> {
    if max_text_len == 0 {
        bail!("max_text_len must be > 0");
    }
    let char_count = text.chars().count().max(1);
    let cc = char_count as f64;
    let count = |p: &dyn Fn(char) -> bool| text.chars().filter(|&c| p(c)).count();
    let kana = count(&is_kana);
    let kanji = count(&is_kanji);
    let alnum = count(&|c| c.is_ascii_alphanumeric());
    let emoji = count_annotation_emojis(text);
    let ch = |a: char, b: char| count(&|c| c == a || c == b);
    let period = ch('。', '.');
    let comma = ch('、', ',');
    let long_vowel = count(&|c| c == 'ー');
    let ellipsis = count(&|c| c == '…');
    let excl = ch('！', '!');
    let quest = ch('？', '?');
    let tc = token_count as f64;
    let row = [
        tc.clamp(0.0, max_text_len as f64) / max_text_len as f64,
        log1p_cap_float(cc, 512.0),
        tc / cc,
        log1p_cap(period, 8),
        log1p_cap(comma, 16),
        log1p_cap(long_vowel, 8),
        log1p_cap(ellipsis, 8),
        log1p_cap(excl, 8),
        log1p_cap(quest, 8),
        log1p_cap(emoji, 8),
        kana as f64 / cc,
        kanji as f64 / cc,
        alnum as f64 / cc,
        if has_speaker { 1.0 } else { 0.0 },
    ];
    Ok(row.map(|x| x as f32))
}

// ---------------------------------------------------------------- フレーム数への変換

/// 予測 log フレーム数 → 使うフレーム数(`inference_runtime.py`)。
/// `expm1` → `* duration_scale` → 四捨五入(Python の `round` と同じ偶数丸め)→
/// `[min_seconds, max_seconds]` 相当のフレーム数へ clamp。
pub fn frames_from_log(
    pred_log: f32,
    duration_scale: f64,
    min_seconds: f64,
    max_seconds: f64,
    sample_rate: usize,
    hop_length: usize,
) -> usize {
    let pred_frames = pred_log.exp_m1() as f64;
    let scaled = pred_frames * duration_scale;
    // Python: ceil(min_seconds * sample_rate / hop_length)
    let min_frames = ((min_seconds * sample_rate as f64 / hop_length as f64).ceil() as i64).max(1);
    let max_frames = ((max_seconds * sample_rate as f64 / hop_length as f64).floor() as i64).max(1);
    let steps = scaled.round_ties_even() as i64;
    steps.min(max_frames).max(min_frames) as usize
}

// ---------------------------------------------------------------- モデル

struct Linear {
    /// [in, out](転置済み)
    w: Tensor<3>,
    b: Option<Tensor<1>>,
}

impl Linear {
    fn load(w: &Weights, prefix: &str, bias: bool, dev: &Device) -> Result<Self> {
        let wt: Tensor<2> = w.tensor(&format!("{prefix}.weight"), dev)?;
        let b = if bias { Some(w.tensor::<1>(&format!("{prefix}.bias"), dev)?) } else { None };
        Ok(Self { w: crate::ops::store(wt.transpose().unsqueeze_dim::<3>(0)), b })
    }

    /// x: [B, S, in] → [B, S, out]
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = crate::ops::matmul(x, self.w.clone());
        match &self.b {
            Some(b) => y + b.clone().unsqueeze::<3>(),
            None => y,
        }
    }
}

struct RmsNorm {
    weight: Tensor<1>,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let ms = (x.clone() * x.clone()).mean_dim(2);
        let inv = (ms + self.eps).sqrt().recip();
        x * inv * self.weight.clone().unsqueeze::<3>()
    }
}

struct Block {
    norm: RmsNorm,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    modulation: Linear,
    caption_modulation: Linear,
}

impl Block {
    /// x: [B, S, H]、cond: [B, speaker_dim]、caption_cond: [B, caption_dim]
    fn forward(&self, x: Tensor<3>, cond: Tensor<2>, caption_cond: Tensor<2>) -> Tensor<3> {
        let h = self.norm.forward(x.clone());
        let hd = h.dims()[2];
        let m = self.modulation.forward(silu(cond).unsqueeze_dim::<3>(1));
        let cm = self.caption_modulation.forward(silu(caption_cond).unsqueeze_dim::<3>(1));
        let m = m + cm; // shift/scale/gate それぞれの和 = 連結テンソルの和
        let shift = m.clone().narrow(2, 0, hd);
        let scale = m.clone().narrow(2, hd, hd);
        let gate = m.narrow(2, 2 * hd, hd);
        let h = h * (scale + 1.0) + shift;
        let mlp = self.w2.forward(silu(self.w1.forward(h.clone())) * self.w3.forward(h));
        x + gate.tanh() * mlp
    }
}

/// `duration_predictor.*`(`token_sum_dual_adarn_zero_no_aux`)
pub struct DurationPredictor {
    null_speaker: Tensor<1>,
    null_caption: Tensor<1>,
    token_input_proj: Linear,
    blocks: Vec<Block>,
    out_norm: RmsNorm,
    out_proj: Linear,
}

/// 長さ予測の入力(PyTorch の `predict_duration_log_frames` の引数)。
/// マスクは f32(1.0 = 有効)、`has_*` は [B] の f32(1.0 = あり)。
pub struct DurationInputs {
    pub text_state: Tensor<3>,
    pub text_mask: Tensor<2>,
    pub speaker_state: Tensor<3>,
    pub has_speaker: Tensor<1>,
    pub caption_state: Tensor<3>,
    pub caption_mask: Tensor<2>,
    pub has_caption: Tensor<1>,
}

impl DurationPredictor {
    pub fn load(w: &Weights, cfg: &ModelConfig, dev: &Device) -> Result<Self> {
        let arch = cfg.duration_architecture.as_deref().unwrap_or("pooled");
        if arch != "token_sum_dual_adarn_zero_no_aux" {
            bail!("unsupported duration architecture: {arch}");
        }
        if !cfg.use_duration_predictor {
            bail!("checkpoint has no duration predictor");
        }
        if cfg.duration_speaker_fusion.as_deref() != Some("adarn_zero")
            || cfg.duration_caption_fusion.as_deref() != Some("adarn_zero")
        {
            bail!("unsupported duration fusion (need adarn_zero)");
        }
        if cfg.duration_caption_pooling.as_deref().unwrap_or("masked_mean") != "masked_mean" {
            bail!("unsupported duration caption pooling");
        }
        let layers = cfg.duration_layers.unwrap_or(3);
        let p = "duration_predictor";
        let norm = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm { weight: w.tensor::<1>(&format!("{p}.{name}.weight"), dev)?, eps: cfg.norm_eps })
        };
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let b = format!("{p}.token_blocks.{i}");
            blocks.push(Block {
                norm: RmsNorm { weight: w.tensor::<1>(&format!("{b}.norm.weight"), dev)?, eps: cfg.norm_eps },
                w1: Linear::load(w, &format!("{b}.mlp.w1"), false, dev)?,
                w2: Linear::load(w, &format!("{b}.mlp.w2"), false, dev)?,
                w3: Linear::load(w, &format!("{b}.mlp.w3"), false, dev)?,
                modulation: Linear::load(w, &format!("{b}.modulation"), true, dev)?,
                caption_modulation: Linear::load(w, &format!("{b}.caption_modulation"), true, dev)?,
            });
        }
        Ok(Self {
            null_speaker: w.tensor(&format!("{p}.null_speaker"), dev)?,
            null_caption: w.tensor(&format!("{p}.null_caption"), dev)?,
            token_input_proj: Linear::load(w, &format!("{p}.token_input_proj"), true, dev)?,
            blocks,
            out_norm: norm("token_out_norm")?,
            out_proj: Linear::load(w, &format!("{p}.token_out_proj"), true, dev)?,
        })
    }

    /// log フレーム数 [B]
    pub fn predict_log_frames(&self, inp: &DurationInputs) -> Tensor<1> {
        let [b, s, _] = inp.text_state.dims();

        // _safe_attention_mask: 有効トークンが 0 の行は状態を 0 にして先頭だけ有効にする
        let any = inp.text_mask.clone().max_dim(1); // [B,1]
        let none = any.clone().neg() + 1.0;
        let dev = inp.text_state.device();
        let mut first = vec![0f32; s];
        first[0] = 1.0;
        let first = Tensor::<1>::from_floats(first.as_slice(), &dev).unsqueeze_dim::<2>(0);
        let text_mask = inp.text_mask.clone() + none.clone() * first;
        let text_state = inp.text_state.clone() * any.unsqueeze_dim::<3>(2);

        // speaker_vec: 先頭トークン or null
        let hs = inp.has_speaker.clone().unsqueeze_dim::<2>(1); // [B,1]
        let spk0 = inp.speaker_state.clone().narrow(1, 0, 1).squeeze_dim::<2>(1);
        let null_s = self.null_speaker.clone().unsqueeze::<2>();
        let speaker_vec = spk0 * hs.clone() + null_s * (hs.neg() + 1.0);

        // caption_vec: マスク付き平均、有効トークンが無ければ null
        let cmask = inp.caption_mask.clone() * inp.has_caption.clone().unsqueeze_dim::<2>(1);
        let denom = cmask.clone().sum_dim(1).clamp_min(1.0); // [B,1]
        let sum = (inp.caption_state.clone() * cmask.clone().unsqueeze_dim::<3>(2)).sum_dim(1).squeeze_dim::<2>(1);
        let cap_mean = sum / denom;
        let cany = cmask.max_dim(1); // [B,1]
        let null_c = self.null_caption.clone().unsqueeze::<2>();
        let caption_vec = cap_mean * cany.clone() + null_c * (cany.neg() + 1.0);

        let mut h = self.token_input_proj.forward(text_state);
        for blk in &self.blocks {
            h = blk.forward(h, speaker_vec.clone(), caption_vec.clone());
        }
        let logits = self.out_proj.forward(self.out_norm.forward(h)).squeeze_dim::<2>(2); // [B,S]
        // softplus(x) = max(x,0) + log1p(exp(-|x|))
        let sp = logits.clone().clamp_min(0.0) + logits.abs().neg().exp().log1p();
        let total = (sp * text_mask).sum_dim(1).squeeze_dim::<1>(1);
        debug_assert_eq!(total.dims()[0], b);
        total.clamp_min(0.0).log1p()
    }
}
