//! 全結合層(DiT・話者エンコーダ・テキストのバックボーンで共通)。
//!
//! 重みは f32(`[in, out]` に転置して保持)、16 ビット(GPU のみ。`[out, in]` の binary16)、または量子化チェックポイントの
//! int8(`[out, in]` の burn 量子化テンソル)。
//! 16 ビットは [`Precision::F16`] のときに読み込みで丸めて置き(VRAM は f32 の半分)、自前の行列積
//! ([`crate::gemm::matmul_half`])が共有メモリへ読むときに f32 へ戻す(一時テンソルは作らない)。
//! int8 は GPU に int8 のまま置き(VRAM は f32 の約 1/4)、自前の行列積([`crate::gemm::matmul_int8`])が共有メモリへ
//! 読むときに f32 へ戻し、結果に行ごとの scale を掛ける(torchao の int8 weight-only の `qdata * 行の scale` と同じ値。
//! 足す順だけが違う)。一時テンソルは作らず、同じ入力に掛ける層は 16 ビット・f32 と同じく 1 つにまとめる。
//! CPU(flex)の int8 は burn の量子化テンソルで、掛け算の直前に f32 へ戻す(参照との突き合わせ用)。

use anyhow::{Result, bail, ensure};
use burn::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData};

use crate::weights::{Precision, Weights};

enum Weight {
    /// `[in, out]`(ロード時に転置済み)
    Dense(Tensor<2>),
    /// `[out, in]` の binary16(GPU のみ)
    Half(Tensor<2>),
    /// `[out, in/4]`(int8 を 4 つずつ詰めた u32 のビット列。[`crate::gemm::matmul_int8`])と、行ごとの scale `[1, out]`(GPU のみ)
    Int8 { q: Tensor<2>, scale: Box<Tensor<2>> },
    /// `[out, in]` の burn の int8 量子化テンソル(行ごとの scale 付き。CPU)
    Quantized(Tensor<2>),
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
        Self::from_names(w, &[format!("{prefix}.weight")], b, dev)
    }

    /// 重みの名前を直接指定する(バイアスなし)
    pub fn weight(w: &Weights, name: &str, dev: &Device) -> Result<Self> {
        Self::from_names(w, &[name.to_string()], None, dev)
    }

    /// 同じ入力に掛ける複数の層(バイアスなし)を、出力方向に連結した 1 つの層として読む(1 回の行列積で済む)。
    /// 量子化した重みと通常の重みが混じるとき、CPU の量子化した重みは連結できないので None
    pub fn load_concat(w: &Weights, names: &[String], dev: &Device) -> Result<Option<Self>> {
        let int8 = names.iter().filter(|n| w.is_int8(n)).count();
        if int8 > 0 && (int8 < names.len() || !crate::gemm::gpu_kernels(dev)) {
            return Ok(None);
        }
        Self::from_names(w, names, None, dev).map(Some)
    }

    /// `names` の重み(各 `[out_i, in]`)を出力方向に連結して読む
    fn from_names(w: &Weights, names: &[String], b: Option<Tensor<1>>, dev: &Device) -> Result<Self> {
        if w.is_int8(&names[0]) {
            let qs = names.iter().map(|n| w.int8(n)).collect::<Result<Vec<_>>>()?;
            let in_dim = qs[0].shape[1];
            ensure!(qs.iter().all(|q| q.shape[1] == in_dim), "cannot concatenate int8 weights {names:?} with different input sizes");
            let out_dim: usize = qs.iter().map(|q| q.shape[0]).sum();
            // 自前の行列積は K(in)と N(out)が 4 の倍数の形だけ
            if crate::gemm::gpu_kernels(dev) && in_dim.is_multiple_of(4) && out_dim.is_multiple_of(4) {
                let packed: Vec<f32> = qs
                    .iter()
                    .flat_map(|q| q.values.as_chunks::<4>().0)
                    .map(|c| f32::from_bits(u32::from_le_bytes(c.map(|v| v as u8))))
                    .collect();
                let scales: Vec<f32> = qs.iter().flat_map(|q| q.row_scales.iter().copied()).collect();
                let q = Tensor::<2>::from_data(TensorData::new(packed, [out_dim, in_dim / 4]), dev);
                let scale = Tensor::<2>::from_data(TensorData::new(scales, [1, out_dim]), dev);
                return Ok(Self { w: Weight::Int8 { q, scale: Box::new(scale) }, b, in_dim, out_dim });
            }
            ensure!(qs.len() == 1, "int8 weights {names:?} can only be concatenated on the GPU");
            let q = qs.into_iter().next().unwrap();
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
            return Ok(Self { w: Weight::Quantized(Tensor::<2>::from_data(data, dev)), b, in_dim, out_dim });
        }
        let shapes = names.iter().map(|n| w.shape(n).map(<[usize]>::to_vec)).collect::<Result<Vec<_>>>()?;
        let in_dim = match shapes[0][..] {
            [_, i] => i,
            _ => bail!("{}: weight must be 2-D, got {:?}", names[0], shapes[0]),
        };
        ensure!(shapes.iter().all(|s| s.len() == 2 && s[1] == in_dim), "cannot concatenate weights {names:?} with shapes {shapes:?}");
        let out_dim: usize = shapes.iter().map(|s| s[0]).sum();
        // 自前の行列積は K(in)と N(out)が 4 の倍数の形だけ
        if w.precision() == Precision::F16 && crate::gemm::gpu_kernels(dev) && in_dim.is_multiple_of(4) && out_dim.is_multiple_of(4) {
            use half::slice::HalfFloatSliceExt;
            let mut half = vec![half::f16::ZERO; out_dim * in_dim];
            let mut at = 0;
            for n in names {
                let v = w.f32_vec(n)?.1;
                half[at..at + v.len()].convert_from_f32_slice(&v);
                at += v.len();
            }
            let opts = TensorCreationOptions::new(dev.clone()).with_dtype(DType::F16);
            return Ok(Self { w: Weight::Half(Tensor::<2>::from_data(TensorData::new(half, [out_dim, in_dim]), opts)), b, in_dim, out_dim });
        }
        let parts = names.iter().map(|n| Ok(w.tensor::<2>(n, dev)?.transpose())).collect::<Result<Vec<_>>>()?;
        let wt = if parts.len() == 1 { parts.into_iter().next().unwrap() } else { Tensor::cat(parts, 1) };
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
            Weight::Dense(wt) => crate::gemm::matmul(x, wt.clone()),
            Weight::Half(w) => crate::gemm::matmul_half(x, w),
            Weight::Int8 { q, scale } => crate::gemm::matmul_int8(x, q).mul((**scale).clone()),
            Weight::Quantized(q) => x.matmul(q.clone().dequantize().transpose()),
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
