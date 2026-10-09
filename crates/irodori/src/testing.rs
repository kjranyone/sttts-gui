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
/// 生成: `cd backend && uv run --no-sync python scripts/dump_irodori_ref.py`
pub fn refs() -> Option<Weights> {
    let p = ref_dir().join("refs.safetensors");
    p.is_file().then(|| Weights::open(p).ok()).flatten()
}

/// HF キャッシュ内のスナップショット(最初に見つかったもの)
pub fn hf_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let base = PathBuf::from(home).join(".cache/huggingface/hub").join(repo_dir).join("snapshots");
    std::fs::read_dir(base).ok()?.flatten().map(|e| e.path()).next()
}

/// Irodori-TTS v4.1 Small MF のスナップショットディレクトリ(`model.safetensors` と `tokenizer/`)
pub fn checkpoint_dir() -> Option<PathBuf> {
    let d = hf_snapshot("models--Aratako--Irodori-TTS-v4.1-Small-MF")?;
    d.join("model.safetensors").is_file().then_some(d)
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

/// テスト用 CPU デバイス(純 Rust の flex バックエンド)
pub fn cpu() -> Device {
    Device::flex()
}
