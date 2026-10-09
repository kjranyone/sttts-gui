//! 行列積の精度切り替え。
//!
//! GPU(wgpu)のメモリは他のアプリと取り合いになり、足りなくなると演算が桁違いに遅くなる。重みを f16 で
//! 持てば約 1.7GB 減る。`set_half_matmul(true)` の後にロードした行列重みは f16 で保持され、
//! [`matmul`] / [`matmul_lw`] は活性を f16 に落として積を取り、結果を f32 に戻す(正規化・softmax・
//! 残差などは f32 のまま)。重みの dtype が f16 かどうかで分岐するので、f32 のまま保持した重みは
//! 従来どおり f32 で計算される。

use std::sync::atomic::{AtomicBool, Ordering};

use burn::tensor::{DType, FloatDType, Tensor};

static HALF: AtomicBool = AtomicBool::new(false);

/// 以降にロードする行列重みを f16 で保持する(`Tts::load_with` が設定する)。
pub fn set_half_matmul(on: bool) {
    HALF.store(on, Ordering::Relaxed);
}

pub fn half_matmul() -> bool {
    HALF.load(Ordering::Relaxed)
}

/// 行列重みを、現在の設定に従う dtype で保持する。
pub fn store<const D: usize>(w: Tensor<D>) -> Tensor<D> {
    if half_matmul() { w.cast(FloatDType::F16) } else { w }
}

/// `x @ w`(重みが右辺)。`w` が f16 なら f16 で積を取り、結果を f32 にして返す。
pub fn matmul<const D: usize>(x: Tensor<D>, w: Tensor<D>) -> Tensor<D> {
    if matches!(w.dtype(), DType::F16) {
        x.cast(FloatDType::F16).matmul(w).cast(FloatDType::F32)
    } else {
        x.matmul(w)
    }
}

/// `w @ x`(重みが左辺。畳み込みを im2col + 行列積にしたときの形)。
pub fn matmul_lw<const D: usize>(w: Tensor<D>, x: Tensor<D>) -> Tensor<D> {
    if matches!(w.dtype(), DType::F16) {
        w.matmul(x.cast(FloatDType::F16)).cast(FloatDType::F32)
    } else {
        w.matmul(x)
    }
}
