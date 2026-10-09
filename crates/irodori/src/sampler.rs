//! MeanFlow オイラーサンプラ(`meanflow.py::sample_euler_meanflow`)と潜在の後処理
//! (`unpatchify_latent`、`find_flattening_point`)。

use anyhow::{Result, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;

use crate::dit::{Conditions, Dit};

/// 標準正規の初期ノイズ `[batch, seq_len, dim]`(自前 RNG。PyTorch の乱数列とは一致しない)
pub fn initial_noise(batch: usize, seq_len: usize, dim: usize, seed: u64, device: &Device) -> Tensor<3> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..batch * seq_len * dim).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
    Tensor::<3>::from_data(TensorData::new(data, vec![batch, seq_len, dim]), device)
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
        let v = dit.forward_prepared(x.clone(), &vec![t; batch], &vec![t - next; batch], &prepared)?;
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
