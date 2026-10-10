//! テスト用の補助(参照出力・モデルの場所、テンソルの比較)。

use std::path::PathBuf;

use burn::tensor::{Device, Tensor};

use crate::weights::Weights;

/// 参照出力のディレクトリ(`IRODORI_REF_DIR` か `target/irodori-ref`)
pub fn ref_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("IRODORI_REF_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/irodori-ref")
}

/// 参照テンソル(キーは `<case>.<stage>.<n>.<name>`)。無ければ None。
/// 生成: `cd tools/reference && uv run python dump_irodori_ref.py`
pub fn refs() -> Option<Weights> {
    let p = ref_dir().join("refs.safetensors");
    p.is_file().then(|| Weights::open(p).ok()).flatten()
}

/// RF(v4.1 Small)の参照出力のディレクトリ(`IRODORI_RF_REF_DIR` か `target/irodori-ref-rf`)
pub fn rf_ref_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("IRODORI_RF_REF_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/irodori-ref-rf")
}

/// RF の参照テンソル。無ければ None。生成: `uv run python dump_irodori_ref.py --rf`
pub fn rf_refs() -> Option<Weights> {
    let p = rf_ref_dir().join("refs.safetensors");
    p.is_file().then(|| Weights::open(p).ok()).flatten()
}

/// RF の参照を作ったモデル
pub const RF_MODEL_REPO: &str = "Aratako/Irodori-TTS-v4.1-Small";

/// int8(v4.1 Small の int8 weight-only)の参照出力のディレクトリ(`IRODORI_INT8_REF_DIR` か `target/irodori-ref-int8`)
pub fn int8_ref_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("IRODORI_INT8_REF_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/irodori-ref-int8")
}

/// int8 の参照テンソル。無ければ None。生成: `uv run python dump_irodori_ref.py --int8`
pub fn int8_refs() -> Option<Weights> {
    let p = int8_ref_dir().join("refs.safetensors");
    p.is_file().then(|| Weights::open(p).ok()).flatten()
}

/// `target/<name>`(環境変数 `var` があればそれ)にある参照出力。無ければ None
pub fn named_refs(var: &str, name: &str) -> Option<(PathBuf, Weights)> {
    let dir = std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target").join(name));
    let p = dir.join("refs.safetensors");
    let w = p.is_file().then(|| Weights::open(&p).ok()).flatten()?;
    Some((dir, w))
}

/// v4 Large と、その int8 weight-only
pub const LARGE_MODEL_REPO: &str = "Aratako/Irodori-TTS-v4-Large";
pub const LARGE_INT8_MODEL_REPO: &str = "Aratako/Irodori-TTS-v4-Large-Quantized";

/// int8 の参照を作ったモデル(リポジトリと、その中の重み)
pub const INT8_MODEL_REPO: &str = "Aratako/Irodori-TTS-v4.1-Small-Quantized";
pub const INT8_MODEL_WEIGHTS: &str = "int8-weight-only/model.safetensors";

/// HF キャッシュ内のスナップショット(`org/name`。最新)
pub fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    crate::hub::find_snapshot(repo, &[])
}

/// Irodori-TTS v4.1 Small MF のスナップショットディレクトリ(`model.safetensors` と `tokenizer/`)
pub fn checkpoint_dir() -> Option<PathBuf> {
    crate::hub::find_snapshot(crate::pipeline::MODEL_REPO, &["model.safetensors"])
}

/// f32 の平坦なベクトルへ
pub fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

/// 最大絶対誤差と、基準 `b` の最大絶対値
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let mut d = 0f32;
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b) {
        d = d.max((x - y).abs());
        m = m.max(y.abs());
    }
    (d, m)
}

/// `got` が参照 `want` に(最大絶対値に対する)相対誤差 `rtol` 以内で一致することを検査する
pub fn assert_close(what: &str, got: &[f32], want: &[f32], rtol: f32) {
    let (d, m) = max_abs_diff(got, want);
    assert!(d <= rtol * m.max(1e-6) + 1e-6, "{what}: max|diff|={d:e} > rtol {rtol} * max|ref|={m:e}");
    eprintln!("{what}: ok (max|diff|={d:.3e}, max|ref|={m:.3e})");
}

/// テストで使うデバイス。既定は純 Rust の CPU(flex)。`IRODORI_DEVICE=gpu`(`gpu` feature 時)で
/// wgpu(Vulkan の独立 GPU)にして、同じ参照との一致を GPU でも確かめられる。
pub fn device() -> Device {
    #[cfg(feature = "_gpu")]
    if std::env::var("IRODORI_DEVICE").as_deref() == Ok("gpu") {
        return crate::device::gpu_device();
    }
    Device::flex()
}
