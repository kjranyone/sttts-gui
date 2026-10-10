//! サンプラ: MeanFlow(`meanflow.py::sample_euler_meanflow`)と RF + CFG(`rf.py::sample_euler_rf_cfg`)、
//! 潜在の後処理(`unpatchify_latent`、`find_flattening_point`)。

use anyhow::{Result, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;

use crate::dit::{Conditions, Dit, Prepared};

/// 標準正規の初期ノイズ `[batch, seq_len, dim]`(自前 RNG。PyTorch の乱数列とは一致しない)
pub fn initial_noise(batch: usize, seq_len: usize, dim: usize, seed: u64, device: &Device) -> Tensor<3> {
    let mut rng = StdRng::seed_from_u64(seed);
    normal(&mut rng, [batch, seq_len, dim], device)
}

fn normal(rng: &mut StdRng, shape: [usize; 3], device: &Device) -> Tensor<3> {
    let n = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
    Tensor::<3>::from_data(TensorData::new(data, shape.to_vec()), device)
}

/// `torch.linspace(1, 0, steps + 1)`(fp32、torch と同じく前半は始点から、後半は終点から数える)
pub fn meanflow_schedule(steps: usize) -> Vec<f32> {
    let n = steps + 1;
    let (start, end) = (1.0f32, 0.0f32);
    let step = (end - start) / steps as f32;
    (0..n)
        .map(|i| if i < n / 2 { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 })
        .collect()
}

/// Euler サンプラ(時刻 1 → 0、`x += v * (next - t)`)。
///
/// `noise` が `Some` ならそれを初期値に使う(`[B, seq_len, dim]`)。`None` なら `seed` の RNG で生成する。
/// 戻り値はパッチ化された潜在 `[B, seq_len, latent_dim * latent_patch_size]`。
pub fn sample_euler_meanflow(
    dit: &Dit,
    cond: &Conditions,
    seq_len: usize,
    steps: usize,
    noise: Option<Tensor<3>>,
    seed: u64,
) -> Result<Tensor<3>> {
    sample_euler_meanflow_padded(dit, cond, seq_len, seq_len, steps, noise, seed)
}

/// [`sample_euler_meanflow`] を、系列を `padded_len` に零詰めして実行する版。有効なのは先頭 `seq_len` 位置で、
/// パディング位置は自己注意から除外されるので、有効位置の結果は零詰め無しと同じ。戻り値は `[B, padded_len, dim]`。
pub fn sample_euler_meanflow_padded(
    dit: &Dit,
    cond: &Conditions,
    seq_len: usize,
    padded_len: usize,
    steps: usize,
    noise: Option<Tensor<3>>,
    seed: u64,
) -> Result<Tensor<3>> {
    ensure!(steps > 0, "MeanFlow steps must be positive");
    ensure!(padded_len >= seq_len, "padded_len < seq_len");
    let batch = cond.batch();
    let dim = dit.cfg.patched_latent_dim();
    let dev = dit.device();
    let mut x = match noise {
        Some(n) => {
            ensure!(n.dims() == [batch, seq_len, dim], "noise shape {:?} != {:?}", n.dims(), [batch, seq_len, dim]);
            n
        }
        None => initial_noise(batch, seq_len, dim, seed, dev),
    };
    if padded_len > seq_len {
        x = Tensor::cat(vec![x, Tensor::<3>::zeros([batch, padded_len - seq_len, dim], dev)], 1);
    }
    let prepared = dit.prepare(cond, padded_len, seq_len)?;
    let sched = meanflow_schedule(steps);
    for i in 0..steps {
        let (t, next) = (sched[i], sched[i + 1]);
        let v = dit.forward_prepared(x.clone(), &vec![t; batch], Some(&vec![t - next; batch]), &prepared)?;
        x = x.add(v.mul_scalar(next - t));
    }
    Ok(x)
}

/// CFG のかけ方(原典の `cfg_guidance_mode`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgGuidanceMode {
    /// 条件ごとに外した版を同じバッチで計算して足し合わせる(既定。1 ステップの計算量は 1 + 有効な CFG の数)
    Independent,
    /// 全部外した 1 本との差(有効な倍率がすべて同じであること)
    Joint,
    /// ステップごとに 1 つの条件を順番に外す
    Alternating,
}

impl CfgGuidanceMode {
    pub const NAMES: [&str; 3] = ["independent", "joint", "alternating"];

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "independent" => Some(Self::Independent),
            "joint" => Some(Self::Joint),
            "alternating" => Some(Self::Alternating),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Joint => "joint",
            Self::Alternating => "alternating",
        }
    }
}

/// 話者条件を「外した」ときの中身(原典の `speaker_uncond_mode`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerUncondMode {
    /// 全無効マスク(既定)
    Mask,
    /// 話者状態と同じ標準偏差のノイズ(マスクは全有効)
    Noise,
}

impl SpeakerUncondMode {
    pub const NAMES: [&str; 2] = ["mask", "noise"];

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mask" => Some(Self::Mask),
            "noise" => Some(Self::Noise),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mask => "mask",
            Self::Noise => "noise",
        }
    }
}

/// RF の時刻の刻み方(原典の `t_schedule_mode` と `sway_coeff`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TSchedule {
    Linear,
    /// F5-TTS の Sway Sampling。係数が負だとノイズ側(序盤)を細かく刻む
    Sway(f32),
}

/// 話者 K/V の強調(原典の `speaker_kv_scale` / `speaker_kv_max_layers` / `speaker_kv_min_t`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerKvScale {
    pub scale: f32,
    /// 先頭から何層に掛けるか(None = 全層)
    pub max_layers: Option<usize>,
    /// 時刻がこれを下回ったら強調をやめる
    pub min_t: f32,
}

/// RF サンプラの指定(原典 `sample_euler_rf_cfg` の引数。倍率は `resolve_cfg_scales` で整理済みのもの)
#[derive(Debug, Clone, PartialEq)]
pub struct RfOptions {
    pub steps: usize,
    pub cfg_scale_text: f32,
    pub cfg_scale_caption: f32,
    pub cfg_scale_speaker: f32,
    pub mode: CfgGuidanceMode,
    /// CFG をかける時刻の範囲(両端を含む)
    pub cfg_min_t: f64,
    pub cfg_max_t: f64,
    pub truncation_factor: Option<f32>,
    /// Temporal score rescaling の (k, sigma)
    pub rescale: Option<(f32, f32)>,
    pub speaker_kv: Option<SpeakerKvScale>,
    pub speaker_uncond: SpeakerUncondMode,
    pub schedule: TSchedule,
}

impl Default for RfOptions {
    fn default() -> Self {
        Self {
            steps: 40,
            cfg_scale_text: 3.0,
            cfg_scale_caption: 3.0,
            cfg_scale_speaker: 5.0,
            mode: CfgGuidanceMode::Independent,
            cfg_min_t: 0.5,
            cfg_max_t: 1.0,
            truncation_factor: None,
            rescale: None,
            speaker_kv: None,
            speaker_uncond: SpeakerUncondMode::Mask,
            schedule: TSchedule::Linear,
        }
    }
}

/// `torch.linspace(start, end, steps + 1)`(fp32、torch と同じく前半は始点から、後半は終点から数える)
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    let n = steps + 1;
    let step = (end - start) / steps as f32;
    (0..n).map(|i| if i < n / 2 { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 }).collect()
}

/// RF の時刻列(1 → 0 に向かって狭義単調減少。`init_scale` = 0.999)
pub fn rf_schedule(steps: usize, schedule: TSchedule) -> Result<Vec<f32>> {
    ensure!(steps > 0, "num_steps must be > 0");
    let mut u = linspace(0.0, 1.0, steps);
    if let TSchedule::Sway(c) = schedule {
        ensure!(c.is_finite(), "sway_coeff must be finite, got {c}");
        for x in &mut u {
            *x = (*x + c * ((0.5 * std::f32::consts::PI * *x).cos() + *x - 1.0)).clamp(0.0, 1.0);
        }
    }
    let t: Vec<f32> = u.iter().map(|&x| (1.0 - x) * 0.999).collect();
    ensure!(t.windows(2).all(|w| w[0] > w[1]), "t_schedule must be strictly decreasing; adjust num_steps or sway_coeff.");
    Ok(t)
}

/// Temporal score rescaling(https://arxiv.org/pdf/2510.01184、原典 `temporal_score_rescale`)
fn temporal_score_rescale(v: Tensor<3>, x: &Tensor<3>, t: f32, k: f32, sigma: f32) -> Tensor<3> {
    let t = f64::from(t);
    if t >= 1.0 {
        return v;
    }
    let omt = 1.0 - t;
    let snr = (omt * omt) / (t * t);
    let s2 = f64::from(sigma) * f64::from(sigma);
    let ratio = (snr * s2 + 1.0) / (snr * s2 / f64::from(k) + 1.0);
    v.mul_scalar(omt as f32).add(x.clone()).mul_scalar(ratio as f32).sub(x.clone()).div_scalar(omt as f32)
}

/// 外す条件
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Guided {
    Text,
    Speaker,
    Caption,
}

/// 条件を外した版(text と caption は 0 + 全無効、話者は `speaker_uncond` に従う)
fn drop_condition(cond: &Conditions, which: &[Guided], speaker_uncond: &Option<(Tensor<3>, Tensor<2>)>) -> Conditions {
    let zeros3 = |t: &Tensor<3>| t.zeros_like();
    let zeros2 = |t: &Tensor<2>| t.zeros_like();
    let (mut ts, mut tm) = (cond.text_state.clone(), cond.text_mask.clone());
    let (mut ss, mut sm) = (cond.speaker_state.clone(), cond.speaker_mask.clone());
    let (mut cs, mut cm) = (cond.caption_state.clone(), cond.caption_mask.clone());
    for w in which {
        match w {
            Guided::Text => (ts, tm) = (zeros3(&ts), zeros2(&tm)),
            Guided::Speaker => {
                if let Some((s, m)) = speaker_uncond {
                    (ss, sm) = (Some(s.clone()), Some(m.clone()));
                }
            }
            Guided::Caption => {
                cs = cs.as_ref().map(zeros3);
                cm = cm.as_ref().map(zeros2);
            }
        }
    }
    Conditions { text_state: ts, text_mask: tm, speaker_state: ss, speaker_mask: sm, caption_state: cs, caption_mask: cm }
}

/// 話者状態の標本標準偏差(`torch.std`、不偏)
fn std_all(t: &Tensor<3>) -> Result<f32> {
    let v: Vec<f32> = t
        .clone()
        .try_into_data()
        .map_err(|e| anyhow::anyhow!("GPU からの読み戻しに失敗しました: {e:?}"))?
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let n = v.len() as f64;
    if n < 2.0 {
        return Ok(0.0);
    }
    let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = v.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok(var.sqrt() as f32)
}

/// RF + CFG のオイラーサンプラ(原典 `sample_euler_rf_cfg`)。系列を `padded_len` に零詰めして実行する
/// (パディング位置は自己注意から除外されるので、有効な先頭 `seq_len` 位置の結果は零詰め無しと同じ)。
///
/// `cond` は条件ありの状態。話者・キャプションが None ならその条件は使わず、CFG も掛けない。
/// `noise` が `Some` なら初期ノイズに使う(`truncation_factor` を掛ける前の標準正規)。戻り値は `[B, padded_len, dim]`。
#[allow(clippy::too_many_arguments)]
pub fn sample_euler_rf_cfg_padded(
    dit: &Dit,
    cond: &Conditions,
    seq_len: usize,
    padded_len: usize,
    opts: &RfOptions,
    noise: Option<Tensor<3>>,
    seed: u64,
) -> Result<Tensor<3>> {
    ensure!(!dit.is_meanflow(), "sample_euler_rf_cfg needs an RF checkpoint (this one is MeanFlow)");
    ensure!(padded_len >= seq_len, "padded_len < seq_len");
    let batch = cond.batch();
    let dim = dit.cfg.patched_latent_dim();
    let dev = dit.device().clone();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = match noise {
        Some(n) => {
            ensure!(n.dims() == [batch, seq_len, dim], "noise shape {:?} != {:?}", n.dims(), [batch, seq_len, dim]);
            // 乱数列は注入しないときと同じ位置から続ける(話者のノイズ用)
            let _ = normal(&mut rng, [batch, seq_len, dim], &dev);
            n
        }
        None => normal(&mut rng, [batch, seq_len, dim], &dev),
    };
    if let Some(f) = opts.truncation_factor {
        x = x.mul_scalar(f);
    }
    if padded_len > seq_len {
        x = Tensor::cat(vec![x, Tensor::<3>::zeros([batch, padded_len - seq_len, dim], &dev)], 1);
    }
    let sched = rf_schedule(opts.steps, opts.schedule)?;

    // 話者条件を外した版
    let speaker_uncond = match (&cond.speaker_state, &cond.speaker_mask) {
        (Some(s), Some(m)) => Some(match opts.speaker_uncond {
            SpeakerUncondMode::Mask => (s.zeros_like(), m.zeros_like()),
            SpeakerUncondMode::Noise => {
                let std = std_all(s)?.max(1e-6);
                (normal(&mut rng, s.dims(), &dev).mul_scalar(std), m.ones_like())
            }
        }),
        _ => None,
    };

    // 有効な CFG(原典と同じ順: text, speaker, caption)
    let mut guided: Vec<(Guided, f32)> = Vec::new();
    if opts.cfg_scale_text > 0.0 {
        guided.push((Guided::Text, opts.cfg_scale_text));
    }
    if opts.cfg_scale_speaker > 0.0 && speaker_uncond.is_some() {
        guided.push((Guided::Speaker, opts.cfg_scale_speaker));
    }
    if opts.cfg_scale_caption > 0.0 && cond.caption_state.is_some() {
        guided.push((Guided::Caption, opts.cfg_scale_caption));
    }
    if opts.mode == CfgGuidanceMode::Joint && guided.len() > 1 {
        let (lo, hi) = guided.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &(_, s)| (lo.min(s), hi.max(s)));
        if hi - lo > 1e-6 {
            bail!("cfg_guidance_mode='joint' expects equal enabled guidance scales; set matching text/speaker/caption scales or use cfg_scale.");
        }
    }

    let prep = |c: &Conditions| -> Result<Prepared> {
        let mut p = dit.prepare(c, padded_len, seq_len)?;
        if let Some(kv) = &opts.speaker_kv {
            p.kv.scale_speaker(kv.scale, kv.max_layers);
        }
        Ok(p)
    };
    let mut cond_p = prep(cond)?;
    // CFG 用(independent はまとめた 1 バッチ、joint は全部外した版、alternating は外す条件ごと)
    let mut guided_p: Vec<Prepared> = Vec::new();
    if !guided.is_empty() {
        match opts.mode {
            CfgGuidanceMode::Independent => {
                let mut parts = vec![drop_condition(cond, &[], &speaker_uncond)];
                parts.extend(guided.iter().map(|(g, _)| drop_condition(cond, &[*g], &speaker_uncond)));
                guided_p.push(prep(&Conditions::cat(&parts)?)?);
            }
            CfgGuidanceMode::Joint => {
                let all: Vec<Guided> = guided.iter().map(|(g, _)| *g).collect();
                guided_p.push(prep(&drop_condition(cond, &all, &speaker_uncond))?);
            }
            CfgGuidanceMode::Alternating => {
                for (g, _) in &guided {
                    guided_p.push(prep(&drop_condition(cond, &[*g], &speaker_uncond))?);
                }
            }
        }
    }
    let mut speaker_kv_active = opts.speaker_kv.is_some();

    for i in 0..opts.steps {
        let (t, next) = (sched[i], sched[i + 1]);
        let use_cfg = !guided.is_empty() && (opts.cfg_min_t..=opts.cfg_max_t).contains(&f64::from(t));
        let mut v = if !use_cfg {
            dit.forward_prepared(x.clone(), &vec![t; batch], None, &cond_p)?
        } else {
            match opts.mode {
                CfgGuidanceMode::Independent => {
                    let mult = 1 + guided.len();
                    let x_cfg = Tensor::cat(vec![x.clone(); mult], 0);
                    let out = dit.forward_prepared(x_cfg, &vec![t; batch * mult], None, &guided_p[0])?;
                    let chunks = out.chunk(mult, 0);
                    let mut v = chunks[0].clone();
                    for ((_, scale), chunk) in guided.iter().zip(&chunks[1..]) {
                        v = v.add(chunks[0].clone().sub(chunk.clone()).mul_scalar(*scale));
                    }
                    v
                }
                CfgGuidanceMode::Joint => {
                    let v_cond = dit.forward_prepared(x.clone(), &vec![t; batch], None, &cond_p)?;
                    let v_uncond = dit.forward_prepared(x.clone(), &vec![t; batch], None, &guided_p[0])?;
                    v_cond.clone().add(v_cond.sub(v_uncond).mul_scalar(guided[0].1))
                }
                CfgGuidanceMode::Alternating => {
                    let k = i % guided.len();
                    let v_cond = dit.forward_prepared(x.clone(), &vec![t; batch], None, &cond_p)?;
                    let v_uncond = dit.forward_prepared(x.clone(), &vec![t; batch], None, &guided_p[k])?;
                    v_cond.clone().add(v_cond.sub(v_uncond).mul_scalar(guided[k].1))
                }
            }
        };
        if let Some((k, sigma)) = opts.rescale {
            v = temporal_score_rescale(v, &x, t, k, sigma);
        }
        if let Some(kv) = opts.speaker_kv.filter(|_| speaker_kv_active)
            && next < kv.min_t
            && t >= kv.min_t
        {
            let inv = 1.0 / kv.scale;
            cond_p.kv.scale_speaker(inv, kv.max_layers);
            for p in &mut guided_p {
                p.kv.scale_speaker(inv, kv.max_layers);
            }
            speaker_kv_active = false;
        }
        x = x.add(v.mul_scalar(next - t));
    }
    Ok(x)
}

/// `[B, T_p, D*patch]` → `[B, T_p*patch, D]`
pub fn unpatchify_latent(patched: Tensor<3>, patch_size: usize, latent_dim: usize) -> Tensor<3> {
    if patch_size <= 1 {
        return patched;
    }
    let [b, t, _] = patched.dims();
    patched.reshape([b, t * patch_size, latent_dim])
}

/// 末尾が平坦(ほぼ無音)になる最初のフレーム。`latent` は `[T, D]` の行優先。
/// 戻り値は `[0, T]`(見つからなければ T)。既定値: window 20, std 0.05, mean 0.1。
pub fn find_flattening_point(
    latent: &[f32],
    frames: usize,
    dim: usize,
    window: usize,
    std_threshold: f32,
    mean_threshold: f32,
) -> usize {
    assert_eq!(latent.len(), frames * dim);
    if frames == 0 || window == 0 {
        return frames;
    }
    // 末尾に window 行のゼロを足したものとして扱う
    let at = |r: usize, c: usize| -> f64 { if r < frames { latent[r * dim + c] as f64 } else { 0.0 } };
    let n = (window * dim) as f64;
    for i in 0..frames {
        let mut sum = 0f64;
        for r in i..i + window {
            for c in 0..dim {
                sum += at(r, c);
            }
        }
        let mean = sum / n;
        let mut var = 0f64;
        for r in i..i + window {
            for c in 0..dim {
                let d = at(r, c) - mean;
                var += d * d;
            }
        }
        let std = (var / n).sqrt();
        if std < std_threshold as f64 && mean.abs() < mean_threshold as f64 {
            return i;
        }
    }
    frames
}

/// [`find_flattening_point`] の既定値版
pub fn find_flattening_point_default(latent: &[f32], frames: usize, dim: usize) -> usize {
    find_flattening_point(latent, frames, dim, 20, 0.05, 0.1)
}
