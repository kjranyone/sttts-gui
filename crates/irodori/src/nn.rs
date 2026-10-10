//! 全結合層(DiT・話者エンコーダ・テキストのバックボーンで共通)。
//!
//! 重みは f32(`[in, out]` に転置して保持)か、量子化チェックポイントの int8(`[out, in]` の burn 量子化テンソル)。
//! int8 は GPU に int8 のまま置き(VRAM は f32 の約 1/4)、掛け算の直前に f32 へ戻す(`qdata * 行の scale`。
//! torchao の int8 weight-only と同じ値)。戻した重みは一時テンソルで、層ごとに捨てる。

use anyhow::Result;
use burn::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{Device, Tensor, TensorData};

use crate::weights::Weights;

enum Weight {
    /// `[in, out]`(ロード時に転置済み)
    Dense(Tensor<2>),
    /// `[out, in]` の int8(行ごとの scale 付き)
    Int8(Tensor<2>),
}

pub struct Linear {
    w: Weight,
    b: Option<Tensor<1>>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    /// `<prefix>.weight`(と `bias` なら `<prefix>.bias`)を読む。量子化してある重みは int8 のまま置く
    pub fn load(w: &Weights, prefix: &str, bias: bool, dev: &Device) -> Result<Self> {
        let b = if bias { Some(w.tensor::<1>(&format!("{prefix}.bias"), dev)?) } else { None };
        Self::with_bias(w, &format!("{prefix}.weight"), b, dev)
    }

    /// 重みの名前を直接指定する(バイアスなし)
    pub fn weight(w: &Weights, name: &str, dev: &Device) -> Result<Self> {
        Self::with_bias(w, name, None, dev)
    }

    fn with_bias(w: &Weights, name: &str, b: Option<Tensor<1>>, dev: &Device) -> Result<Self> {
        if w.is_int8(name) {
            let q = w.int8(name)?;
            let [out_dim, in_dim] = q.shape;
            // burn のブロックは 1 次元あたり 255 まで。行ごとの scale を、行を割り切るブロックに繰り返して持たせる
            let block = (1..=in_dim.min(255)).rev().find(|d| in_dim.is_multiple_of(*d)).unwrap_or(1);
            let per_row = in_dim / block;
            let scales: Vec<f32> = q.row_scales.iter().flat_map(|&s| std::iter::repeat_n(s, per_row)).collect();
            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_store(QuantStore::PackedU32(0))
                .per_block([1u8, block as u8], ScaleDtype::F32);
            let data = TensorData::quantized(q.values, [out_dim, in_dim], scheme, &scales, None);
            return Ok(Self { w: Weight::Int8(Tensor::<2>::from_data(data, dev)), b, in_dim, out_dim });
        }
        let wt = w.tensor::<2>(name, dev)?.transpose();
        let [in_dim, out_dim] = wt.dims();
        Ok(Self { w: Weight::Dense(wt), b, in_dim, out_dim })
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn forward2(&self, x: Tensor<2>) -> Tensor<2> {
        let y = match &self.w {
            Weight::Dense(wt) => x.matmul(wt.clone()),
            Weight::Int8(q) => x.matmul(q.clone().dequantize().transpose()),
        };
        match &self.b {
            Some(b) => y.add(b.clone().reshape([1, self.out_dim])),
            None => y,
        }
    }

    pub fn forward3(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, s, i] = x.dims();
        self.forward2(x.reshape([b * s, i])).reshape([b, s, self.out_dim])
    }
}
