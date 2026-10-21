//! f32 の行列積と 1 次元畳み込み(GPU は自前の WGSL カーネル)。
//!
//! burn 0.22 の既定の行列積(autotune なし)は Intel Arc で FP32 性能の 1 割も出ない形状が多い
//! (B580 で 2048 角が 0.8 TFLOPS)。DiT の全結合とコーデックの畳み込みはほぼ全時間が行列積なので、
//! 共有メモリでタイル化したカーネル(`gemm.wgsl`)を使う(同じ B580 で 2048 角が 9 TFLOPS 前後)。
//! コーデックの畳み込み(stride 1)は im2col を作らずに同じ行列積で計算し、前段の Snake・バイアス・
//! 残差の足し込みも同じカーネルで行う(`conv1d.wgsl`)。透かしのゲート付き 3x3 畳み込みも同じカーネルで、
//! 時間方向に切らずに全体を 1 回で計算する(一時メモリは入力と出力だけ)。
//!
//! 1 スレッドが受け持つ大きさ(タイル)は形状ごとに速いものが違うので、形状ごとに初回だけ候補を実測して選ぶ。
//! どのタイルでも各出力は K 方向に同じ順で足すので、選んだタイルで結果は変わらない(ビット単位で同じ)。
//! CPU(flex)と、行が 16 未満・K や N が 4 の倍数でない形状は burn の行列積のまま。

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Dispatch, Flex, backend_extension};
use burn::tensor::{Device, Tensor};

#[backend_extension(Cube: cfg(feature = "_gpu"), Flex)]
pub trait GemmBackend: Backend {
    /// `[M,K] x [K,N]`、またはバッチ付きの `[Bt,M,K] x [Bt,K,N]`。`tile` は 1 スレッドが受け持つ行数・列数(`tm * 16 + tn`)と
    /// K の分割数(`<< 8`)
    fn gemm(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, tile: u32) -> FloatTensor<Self>;
}

impl GemmBackend for Flex {
    fn gemm(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, _tile: u32) -> FloatTensor<Self> {
        <Self as burn::backend::ops::FloatTensorOps<Self>>::float_matmul(lhs, rhs)
    }
}

/// 候補のタイル(1 スレッドあたりの行 x 列)
const TILES: [(u32, u32); 3] = [(4, 4), (4, 8), (8, 8)];

fn pack(t: (u32, u32)) -> u32 {
    t.0 * 16 + t.1
}

fn gemm_tile(a: Tensor<2>, b: Tensor<2>, t: (u32, u32), splits: usize) -> Tensor<2> {
    let code = pack(t) | ((splits as u32) << 8);
    Tensor::from_dispatch(<Dispatch as GemmBackend>::gemm(a.into_dispatch(), b.into_dispatch(), code))
}

/// K の分割数。出力のタイル(64 x 64 換算)が少なく GPU が埋まらない形状だけ K を分ける(分割数は形状だけで決まる)。
/// 部分和を足す順は固定なので、同じ形状なら結果は毎回同じ(分割しない場合とは足す順が違う)。
fn k_splits(m: usize, n: usize, k: usize) -> usize {
    const TARGET_TILES: usize = 320;
    const MIN_CHUNK: usize = 512;
    let tiles = m.div_ceil(64) * n.div_ceil(64);
    TARGET_TILES.div_ceil(tiles).min(k / MIN_CHUNK).clamp(1, 8)
}

/// 種類と形状(M は 2 の冪に切り上げ、N、K)ごとに選んだタイル
type TileKey = (&'static str, usize, usize, usize);
static CHOSEN: Mutex<Option<HashMap<TileKey, (u32, u32)>>> = Mutex::new(None);

/// 形状ごとに選んだタイルで `run` を呼ぶ。未計測の形状なら各候補を 1 回(コンパイル)+ 3 回(計測)走らせて選ぶ
fn tuned<const D: usize>(key: TileKey, dev: &Device, run: impl Fn((u32, u32)) -> Tensor<D>) -> Tensor<D> {
    let key = (key.0, key.1.next_power_of_two(), key.2, key.3);
    if let Some(t) = CHOSEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().get(&key).copied() {
        return run(t);
    }
    let mut best: Option<(f64, (u32, u32), Tensor<D>)> = None;
    for t in TILES {
        let _ = run(t);
        let _ = dev.sync();
        let start = Instant::now();
        let mut out = None;
        for _ in 0..3 {
            out = Some(run(t));
        }
        let _ = dev.sync();
        let sec = start.elapsed().as_secs_f64();
        if best.as_ref().is_none_or(|(s, _, _)| sec < *s) {
            best = Some((sec, t, out.unwrap()));
        }
    }
    let (_, t, out) = best.unwrap();
    CHOSEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(key, t);
    out
}

/// `a [M,K] x b [K,N]`(f32)
pub fn matmul(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let [m, k] = a.dims();
    let n = b.dims()[1];
    let dev = a.device();
    // 行が少ない(adaLN の [1, dim] など)ときはタイルの大半が空になるので burn のまま
    if dev == Device::flex() || m < 16 || !k.is_multiple_of(4) || !n.is_multiple_of(4) {
        return a.matmul(b);
    }
    let splits = k_splits(m, n, k);
    tuned(("gemm", m, n, k), &dev, |t| gemm_tile(a.clone(), b.clone(), t, splits))
}

/// バッチ付きの行列積 `a [Bt,M,K] x b [Bt,K,N]`(注意のヘッドごとの QK^T と PV)
pub fn bmm(a: Tensor<3>, b: Tensor<3>) -> Tensor<3> {
    let [bt, m, k] = a.dims();
    let n = b.dims()[2];
    let dev = a.device();
    if dev == Device::flex() || b.dims() != [bt, k, n] || m < 16 || !k.is_multiple_of(4) || !n.is_multiple_of(4) {
        return a.matmul(b);
    }
    let splits = k_splits(m * bt, n, k);
    tuned(("bmm", m * bt, n, k), &dev, |t| {
        let code = pack(t) | ((splits as u32) << 8);
        Tensor::from_dispatch(<Dispatch as GemmBackend>::gemm(a.clone().into_dispatch(), b.clone().into_dispatch(), code))
    })
}

/// 指定したタイルで計算する(ベンチ用)
pub fn matmul_tile(a: Tensor<2>, b: Tensor<2>, tm: u32, tn: u32, splits: usize) -> Tensor<2> {
    gemm_tile(a, b, (tm, tn), splits)
}

/// 畳み込みの前段に掛ける Snake(`x + inv * sin(alpha * x)^2`、どちらも `[C]`)
pub struct SnakeParams<'a> {
    pub alpha: &'a Tensor<1>,
    pub inv: &'a Tensor<1>,
}

/// 1 次元畳み込み(stride 1、バッチ 1)を GPU の自前カーネルで計算する。`x`: `[1, C, L]`、`w2`: `[Cout, k*C]`
/// (`w[o, c, j]` を `j*C + c` に並べたもの)、`bias`: `[Cout]`、`res`: `[1, Cout, Lout]`(足し込む残差)。
/// GPU でない、または形が合わないときは None(呼び出し側が従来の計算をする)。
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &Tensor<3>,
    w2: &Tensor<2>,
    k: usize,
    dil: usize,
    pad: usize,
    bias: Option<&Tensor<1>>,
    snake: Option<SnakeParams<'_>>,
    res: Option<&Tensor<3>>,
) -> Option<Tensor<3>> {
    #[cfg(feature = "_gpu")]
    {
        let [batch, c, l] = x.dims();
        let [co, kc] = w2.dims();
        let dev = x.device();
        if dev == Device::flex() || batch != 1 || kc != k * c || !kc.is_multiple_of(4) || l + 2 * pad < dil * (k - 1) + 1 {
            return None;
        }
        let lout = l + 2 * pad - dil * (k - 1);
        if res.is_some_and(|r| r.dims() != [1, co, lout]) {
            return None;
        }
        // 使わない入力には x を渡す(カーネル側でダミーに差し替える)
        let flags = u32::from(bias.is_some()) | (u32::from(snake.is_some()) << 1) | (u32::from(res.is_some()) << 2);
        let dummy1 = || x.clone().reshape([c * l]);
        let bias = bias.cloned().unwrap_or_else(dummy1);
        let (alpha, inv) = match &snake {
            Some(s) => (s.alpha.clone(), s.inv.clone()),
            None => (dummy1(), dummy1()),
        };
        let res = res.cloned().unwrap_or_else(|| x.clone());
        let geom = conv_geom(co, c, l, lout, k, dil, -(pad as i32), k, 0, kc, None, flags);
        let out = tuned(("conv1d", co, lout, kc), &dev, |t| {
            Tensor::<2>::from_dispatch(<Dispatch as cube::ConvBackend>::conv1d(
                x.clone().into_dispatch(),
                w2.clone().into_dispatch(),
                bias.clone().into_dispatch(),
                alpha.clone().into_dispatch(),
                inv.clone().into_dispatch(),
                res.clone().into_dispatch(),
                geom.clone(),
                pack(t),
            ))
        });
        Some(out.reshape([1, co, lout]))
    }
    #[cfg(not(feature = "_gpu"))]
    {
        let _ = (x, w2, k, dil, pad, bias, snake, res);
        None
    }
}

/// 転置畳み込み(kernel = 2 * stride、padding = stride / 2、出力長 = L * stride)を GPU の自前カーネルで計算する。
/// `x`: `[1, Cin, L]`、`wt`: `[2*stride*Cout, Cin]`(`w[c, o, j]` を行 `j*Cout + o` に並べたもの)、`bias`: `[Cout]`。
/// 前段の Snake も同じ行列積に組み込む。GPU でない、または形が合わないときは None。
pub fn conv_transpose1d(x: &Tensor<3>, wt: &Tensor<2>, stride: usize, bias: &Tensor<1>, snake: Option<SnakeParams<'_>>) -> Option<Tensor<3>> {
    #[cfg(feature = "_gpu")]
    {
        let [batch, _, l] = x.dims();
        let co = bias.dims()[0];
        if x.device() == Device::flex() || batch != 1 || !stride.is_multiple_of(2) || wt.dims()[0] != 2 * stride * co {
            return None;
        }
        let z = conv1d(x, wt, 1, 1, 0, None, snake, None)?.reshape([2 * stride * co, l]);
        let geom = vec![co as u32, l as u32, (l * stride) as u32, stride as u32, (stride / 2) as u32];
        let out = Tensor::<2>::from_dispatch(<Dispatch as cube::ConvBackend>::overlap_add(
            z.into_dispatch(),
            bias.clone().into_dispatch(),
            geom,
        ));
        Some(out.reshape([1, co, l * stride]))
    }
    #[cfg(not(feature = "_gpu"))]
    {
        let _ = (x, wt, stride, bias, snake);
        None
    }
}

/// `conv1d.wgsl` の起動引数: [Cout, Cin, L, Lout, 近傍の数, dil, off0, kw, rs, K, (ncols, wp, w), flags]
#[cfg(feature = "_gpu")]
#[allow(clippy::too_many_arguments)]
fn conv_geom(
    co: usize,
    c: usize,
    l: usize,
    lout: usize,
    taps: usize,
    dil: usize,
    off0: i32,
    kw: usize,
    rs: usize,
    k_dim: usize,
    flat: Option<&Flat>,
    flags: u32,
) -> Vec<u32> {
    let (ncols, wp, w) = flat.map_or((0, 1, 0), |f| (f.h * f.wp(), f.wp(), f.w));
    [co, c, l, lout, taps, dil]
        .iter()
        .map(|&v| v as u32)
        .chain([off0 as u32, kw as u32, rs as u32, k_dim as u32, ncols as u32, wp as u32, w as u32, flags])
        .collect()
}

/// 透かしの縁取り付きの平らな配列(`h` 行 `w` 列。行の幅は `w + 2`、位置 `(i, j)` は `(i + 1) * (w + 2) + j + 1`)。
/// 層の出力は `n_out` 列(`h * (w + 2)` 以上)を計算し、長さ `n_out + 2 * (w + 2) + 2` の配列に置く
pub struct Flat {
    pub h: usize,
    pub w: usize,
    pub n_out: usize,
}

impl Flat {
    fn wp(&self) -> usize {
        self.w + 2
    }

    pub fn buf_len(&self) -> usize {
        self.n_out + 2 * self.wp() + 2
    }
}

/// 透かし(SilentCipher)の層: ゲート付き畳み込み(k = 1 か 3)+ 学習モードの BatchNorm(この入力の有効な位置で統計を取る)。
/// `x`: 平らな活性 `[Cin, buf_len]`(縁取りと無効な位置は 0)、`w2`: `[2*Cout, K]`(conv の行のあとに gate の行。
/// 列は (dy, dx, c) の順で、K は `k*k*Cin` を 4 の倍数に零で詰めたもの)、`b`: `[2*Cout]`、`bn_w` / `bn_b`: `[Cout]`。
/// 戻り値は同じ形式の `[Cout, buf_len]`。入力は畳み込みの後すぐ捨てる(同時に持つのは入力か出力と、ゲート後の値の 2 つ)。
/// GPU([`gpu_kernels`] が真)のときだけ呼ぶ。
#[allow(clippy::too_many_arguments)]
pub fn gated_conv_bn(
    x: Tensor<2>,
    w2: &Tensor<2>,
    b: &Tensor<1>,
    bn_w: &Tensor<1>,
    bn_b: &Tensor<1>,
    k: usize,
    flat: &Flat,
    eps: f32,
) -> Tensor<2> {
    #[cfg(feature = "_gpu")]
    {
        let [c, l] = x.dims();
        let [co2, kc] = w2.dims();
        let dev = x.device();
        assert!(dev != Device::flex(), "gated_conv_bn: GPU only");
        assert!(
            l == flat.buf_len() && kc >= k * k * c && kc.is_multiple_of(4) && co2.is_multiple_of(2),
            "gated_conv_bn: bad shapes x {:?} w2 {:?} k {k} buf_len {}",
            x.dims(),
            w2.dims(),
            flat.buf_len()
        );
        let co = co2 / 2;
        let wp = flat.wp();
        // 出力の位置 p は平らな配列の wp + 1 + p。近傍 (dy, dx) は p + dy * wp + dx(k = 1 なら p + wp + 1)
        let off0 = ((1 - k / 2) * (wp + 1)) as i32;
        let geom = conv_geom(co, c, l, flat.n_out, k * k, 1, off0, k, wp, kc, Some(flat), 8 | 1);
        // 使わない入力(Snake・残差)にはバイアスを渡す(カーネル側でダミーに差し替える)
        let y = tuned(("gconv", co2, 0, kc), &dev, |t| {
            Tensor::<2>::from_dispatch(<Dispatch as cube::ConvBackend>::conv1d(
                x.clone().into_dispatch(),
                w2.clone().into_dispatch(),
                b.clone().into_dispatch(),
                b.clone().into_dispatch(),
                b.clone().into_dispatch(),
                b.clone().into_dispatch(),
                geom.clone(),
                pack(t),
            ))
        });
        drop(x);
        let ncols = (flat.h * wp) as u32;
        let n_valid = (flat.h * flat.w) as f32;
        let sgeom = vec![flat.n_out as u32, wp as u32, flat.w as u32, ncols, n_valid.to_bits(), eps.to_bits()];
        let ab = <Dispatch as cube::ConvBackend>::bn_stats(y.clone().into_dispatch(), bn_w.clone().into_dispatch(), bn_b.clone().into_dispatch(), sgeom);
        let ageom = vec![flat.n_out as u32, wp as u32, flat.w as u32, ncols, flat.buf_len() as u32];
        Tensor::from_dispatch(<Dispatch as cube::ConvBackend>::bn_apply(y.into_dispatch(), ab, ageom))
    }
    #[cfg(not(feature = "_gpu"))]
    {
        let _ = (x, w2, b, bn_w, bn_b, k, flat, eps);
        unreachable!("gated_conv_bn: GPU only")
    }
}

/// `dev` で自前の GPU カーネルが使えるか(GPU ビルドで、CPU(flex)でない)
pub fn gpu_kernels(dev: &Device) -> bool {
    cfg!(feature = "_gpu") && *dev != Device::flex()
}

/// 注意の q / k のヘッドごとの RMSNorm(重み `w`: `[heads*hd]`)と RoPE(前半 `rot_heads` 個のヘッド。表 `cos` / `sin`: `[S*hd/2]`)。
/// `y`: `[B*S, stride]` の `offset` 列目から `heads*hd` 列。戻り値は `[B*S, heads*hd]`。GPU でなければ None。
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    y: &Tensor<2>,
    offset: usize,
    seq: usize,
    heads: usize,
    rot_heads: usize,
    w: &Tensor<1>,
    cos: &Tensor<1>,
    sin: &Tensor<1>,
    eps: f32,
) -> Option<Tensor<2>> {
    #[cfg(feature = "_gpu")]
    {
        let [rows, stride] = y.dims();
        let hd = w.dims()[0] / heads;
        if y.device() == Device::flex() || offset + heads * hd > stride || !rows.is_multiple_of(seq) || cos.dims()[0] != seq * hd / 2 {
            return None;
        }
        let geom = vec![rows as u32, seq as u32, heads as u32, hd as u32, rot_heads as u32, stride as u32, offset as u32, eps.to_bits()];
        Some(Tensor::from_dispatch(<Dispatch as cube::ConvBackend>::qk_norm_rope(
            y.clone().into_dispatch(),
            w.clone().into_dispatch(),
            cos.clone().into_dispatch(),
            sin.clone().into_dispatch(),
            geom,
        )))
    }
    #[cfg(not(feature = "_gpu"))]
    {
        let _ = (y, offset, seq, heads, rot_heads, w, cos, sin, eps);
        None
    }
}

/// adaLN の正規化と変調: `x * rsqrt(mean(x^2) + eps) * (scale + 1) + shift`。`x`: `[B, S, D]`、`shift` / `scale`: `[1 か B, 1, D]`。
/// GPU でなければ None。
pub fn norm_modulate(x: &Tensor<3>, shift: &Tensor<3>, scale: &Tensor<3>, eps: f32) -> Option<Tensor<3>> {
    #[cfg(feature = "_gpu")]
    {
        let [b, s, d] = x.dims();
        let mb = shift.dims()[0];
        if x.device() == Device::flex() || shift.dims() != [mb, 1, d] || scale.dims() != [mb, 1, d] || (mb != 1 && mb != b) {
            return None;
        }
        let geom = vec![(b * s) as u32, d as u32, s as u32, mb as u32, eps.to_bits()];
        let out = Tensor::<2>::from_dispatch(<Dispatch as cube::ConvBackend>::norm_modulate(
            x.clone().into_dispatch(),
            shift.clone().into_dispatch(),
            scale.clone().into_dispatch(),
            geom,
        ));
        Some(out.reshape([b, s, d]))
    }
    #[cfg(not(feature = "_gpu"))]
    {
        let _ = (x, shift, scale, eps);
        None
    }
}

#[cfg(feature = "_gpu")]
mod cube {
    use burn::backend::tensor::FloatTensor;
    use burn::backend::{Backend, Cube, backend_extension};
    use burn_cubecl::cubecl::prelude::KernelId;
    use burn_cubecl::cubecl::server::{Handle, KernelArguments};
    use burn_cubecl::cubecl::{CubeCount, CubeDim};
    use burn_cubecl::kernel::into_contiguous;
    use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::template::{KernelSource, SourceKernel, SourceTemplate};

    use super::GemmBackend;

    /// K 方向のタイル幅
    const BK: usize = 16;

    #[backend_extension(Cube: cfg(feature = "_gpu"))]
    pub trait ConvBackend: Backend {
        /// [`super::conv1d`] と [`super::gated_conv_bn`] の畳み込み。`geom` は `conv_geom` の並び
        /// (flags = bias | snake<<1 | res<<2 | gate<<3)
        #[allow(clippy::too_many_arguments)]
        fn conv1d(
            x: FloatTensor<Self>,
            w2: FloatTensor<Self>,
            bias: FloatTensor<Self>,
            alpha: FloatTensor<Self>,
            inv: FloatTensor<Self>,
            res: FloatTensor<Self>,
            geom: Vec<u32>,
            tile: u32,
        ) -> FloatTensor<Self>;

        /// [`super::conv_transpose1d`] の重なり加算。`geom` = [Cout, L, Lout, stride, pad]
        fn overlap_add(z: FloatTensor<Self>, bias: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self>;

        /// [`super::qk_norm_rope`]。`geom` = [rows, seq, heads, hd, rot_heads, stride, offset, eps]
        fn qk_norm_rope(
            y: FloatTensor<Self>,
            w: FloatTensor<Self>,
            cos: FloatTensor<Self>,
            sin: FloatTensor<Self>,
            geom: Vec<u32>,
        ) -> FloatTensor<Self>;

        /// [`super::norm_modulate`]。`geom` = [rows, d, seq, mod_batches, eps]
        fn norm_modulate(x: FloatTensor<Self>, shift: FloatTensor<Self>, scale: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self>;

        /// [`super::gated_conv_bn`] の BatchNorm の統計 `[2*C]`(係数と足す値)。`geom` = [n, wp, w, ncols, n_valid, eps]
        fn bn_stats(y: FloatTensor<Self>, bn_w: FloatTensor<Self>, bn_b: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self>;

        /// [`super::gated_conv_bn`] の BatchNorm の適用と平らな配列への配置。`geom` = [n, wp, w, ncols, buf_len]
        fn bn_apply(y: FloatTensor<Self>, ab: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self>;
    }

    /// 行列積の内側(累積の宣言・K 方向 1 段・書き出し)を展開したコード。`store(i, j, row, colv)` が書き出し 1 つ分を返す
    fn unrolled(tm: usize, tnv: usize, store: impl Fn(usize, usize, String, String) -> String) -> (String, String, String) {
        let mut decl = String::new();
        let mut body = String::new();
        let mut out = String::new();
        for i in 0..tm {
            for j in 0..tnv {
                decl += &format!("    var c{i}_{j} = vec4<f32>(0.0);\n");
            }
        }
        for i in 0..tm {
            body += &format!("            let a{i} = vec4<f32>(As[ao + {}u]);\n", 16 * i);
        }
        for j in 0..tnv {
            body += &format!("            let b{j} = Bs[bo + {}u];\n", 16 * j);
        }
        for i in 0..tm {
            for j in 0..tnv {
                body += &format!("            c{i}_{j} = fma(a{i}, b{j}, c{i}_{j});\n");
            }
        }
        for i in 0..tm {
            for j in 0..tnv {
                out += &store(i, j, format!("(row0 + ty + {}u)", 16 * i), format!("(colv0 + tx + {}u)", 16 * j));
            }
        }
        (decl, body, out)
    }

    /// テンプレートに埋め込む(SourceTemplate の置換は複数行の値を壊すので、自前で置き換える)
    fn fill(src: &str, tm: usize, tnv: usize, extra: &[(&str, String)], code: (String, String, String)) -> SourceTemplate {
        let (bm, bnv) = (16 * tm, 16 * tnv);
        let mut s = src
            .replace("{{ bm }}", &bm.to_string())
            .replace("{{ bnv }}", &bnv.to_string())
            .replace("{{ bk }}", &BK.to_string())
            .replace("{{ as_len }}", &(BK * bm).to_string())
            .replace("{{ bs_len }}", &(BK * bnv).to_string())
            .replace("{{ decl }}", &code.0)
            .replace("{{ body }}", &code.1)
            .replace("{{ store }}", &code.2);
        for (k, v) in extra {
            s = s.replace(&format!("{{{{ {k} }}}}"), v);
        }
        SourceTemplate::new(s)
    }

    #[derive(Debug)]
    struct Gemm {
        tm: u32,
        tn: u32,
    }

    impl KernelSource for Gemm {
        fn source(&self) -> SourceTemplate {
            let (tm, tnv) = (self.tm as usize, self.tn as usize / 4);
            let code = unrolled(tm, tnv, |i, j, row, colv| {
                format!("    if ({row} < m_dim && {colv} < nv) {{ c[out_base + {row} * nv + {colv}] = c{i}_{j}; }}\n")
            });
            fill(include_str!("gemm.wgsl"), tm, tnv, &[], code)
        }

        fn id(&self) -> KernelId {
            KernelId::new::<Self>().info((self.tm, self.tn))
        }

        fn lang(&self) -> &'static str {
            "wgsl"
        }
    }

    #[derive(Debug)]
    struct Conv {
        tm: u32,
        tn: u32,
        flags: u32,
    }

    impl KernelSource for Conv {
        fn source(&self) -> SourceTemplate {
            let (tm, tnv) = (self.tm as usize, self.tn as usize / 4);
            let gate = self.flags & 8 != 0;
            let code = unrolled(tm, tnv, |i, j, row, colv| {
                if !gate {
                    format!("    store({row}, {colv} * 4u, c{i}_{j}, m_dim, n_dim);\n")
                } else if i < tm / 2 {
                    // 前半の行が conv、後半(i + tm / 2)が同じチャネルの gate
                    format!("    store_gate({row}, {colv} * 4u, c{i}_{j}, c{}_{j}, m_dim, n_dim);\n", i + tm / 2)
                } else {
                    String::new()
                }
            });
            let flag = |b: u32| (self.flags & b != 0).to_string();
            let extra = [("has_bias", flag(1)), ("has_snake", flag(2)), ("has_res", flag(4)), ("has_gate", flag(8))];
            fill(include_str!("conv1d.wgsl"), tm, tnv, &extra, code)
        }

        fn id(&self) -> KernelId {
            KernelId::new::<Self>().info((self.tm, self.tn, self.flags))
        }

        fn lang(&self) -> &'static str {
            "wgsl"
        }
    }

    /// 置換の要らない WGSL をそのまま使うカーネル
    macro_rules! plain_kernel {
        ($name:ident, $file:literal) => {
            #[derive(Debug)]
            struct $name;

            impl KernelSource for $name {
                fn source(&self) -> SourceTemplate {
                    SourceTemplate::new(include_str!($file))
                }

                fn id(&self) -> KernelId {
                    KernelId::new::<Self>()
                }

                fn lang(&self) -> &'static str {
                    "wgsl"
                }
            }
        };
    }

    plain_kernel!(OverlapAdd, "ola.wgsl");
    plain_kernel!(Reduce, "reduce.wgsl");
    plain_kernel!(QkNormRope, "qk_norm_rope.wgsl");
    plain_kernel!(NormModulate, "norm_modulate.wgsl");
    plain_kernel!(BnStats, "bn_stats.wgsl");
    plain_kernel!(BnApply, "bn_apply.wgsl");

    fn upload_u32(t: &CubeTensor, v: &[u32]) -> Handle {
        let bytes: Vec<u8> = v.iter().flat_map(|d| d.to_le_bytes()).collect();
        t.client.create_from_slice(&bytes)
    }

    fn unpack(tile: u32) -> (u32, u32) {
        (tile / 16, tile % 16)
    }

    /// 1 次元のワークグループ数を、x の上限(65535)を超えないよう y に折り返す
    fn grid_1d(groups: usize) -> CubeCount {
        CubeCount::new_2d(groups.min(65535) as u32, groups.div_ceil(65535) as u32)
    }

    impl GemmBackend for Cube {
        fn gemm(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, tile: u32) -> FloatTensor<Self> {
            let (tm, tn) = unpack(tile & 0xff);
            let splits = (tile >> 8).max(1) as usize;
            let ls = lhs.meta.shape().clone();
            let rs = rhs.meta.shape().clone();
            let nd = ls.num_dims();
            assert!(nd == rs.num_dims() && (nd == 2 || nd == 3), "gemm: shapes {ls:?} x {rs:?}");
            let bt = if nd == 3 { ls[0] } else { 1 };
            let (m, k, n) = (ls[nd - 2], ls[nd - 1], rs[nd - 1]);
            assert!(rs[nd - 2] == k && (nd == 2 || rs[0] == bt), "gemm: inner dims {ls:?} x {rs:?}");
            assert!(k % 4 == 0 && n % 4 == 0, "gemm: K and N must be multiples of 4 ({ls:?} x {rs:?})");
            let lhs = into_contiguous(lhs);
            let rhs = into_contiguous(rhs);
            let client = lhs.client.clone();
            // K を splits 個に分ける(区間は BK の倍数)。部分和は [splits, M, N] に書いて、後で足す
            let kchunk = k.div_ceil(splits).div_ceil(BK) * BK;
            let splits = k.div_ceil(kchunk);
            let out_shape = if nd == 3 { vec![bt, m, n] } else { vec![m, n] };
            let mut shape = out_shape.clone();
            if splits > 1 {
                shape.insert(0, splits);
            }
            let part = empty_device_contiguous_dtype(client.clone(), lhs.device.clone(), shape.into(), lhs.dtype);
            let dims = upload_u32(&lhs, &[m as u32, n as u32, k as u32, kchunk as u32, splits as u32, (m * k / 4) as u32, (k * n / 4) as u32]);
            let (bm, bn) = (16 * tm as usize, 16 * tn as usize);
            client.launch(
                Box::new(SourceKernel::new(Gemm { tm, tn }, CubeDim::new_2d(16, 16))),
                CubeCount::new_3d(n.div_ceil(bn) as u32, m.div_ceil(bm) as u32, (splits * bt) as u32),
                KernelArguments::new()
                    .with_buffer(lhs.handle.clone().binding())
                    .with_buffer(rhs.handle.clone().binding())
                    .with_buffer(part.handle.clone().binding())
                    .with_buffer(dims.binding()),
            );
            if splits == 1 {
                return part;
            }
            let out = empty_device_contiguous_dtype(client.clone(), lhs.device.clone(), out_shape.into(), lhs.dtype);
            let len = bt * m * n / 4;
            let rdims = upload_u32(&lhs, &[len as u32, splits as u32]);
            client.launch(
                Box::new(SourceKernel::new(Reduce, CubeDim::new_1d(256))),
                grid_1d(len.div_ceil(256)),
                KernelArguments::new()
                    .with_buffer(part.handle.clone().binding())
                    .with_buffer(out.handle.clone().binding())
                    .with_buffer(rdims.binding()),
            );
            out
        }
    }

    impl ConvBackend for Cube {
        fn conv1d(
            x: FloatTensor<Self>,
            w2: FloatTensor<Self>,
            bias: FloatTensor<Self>,
            alpha: FloatTensor<Self>,
            inv: FloatTensor<Self>,
            res: FloatTensor<Self>,
            geom: Vec<u32>,
            tile: u32,
        ) -> FloatTensor<Self> {
            let (tm, tn) = unpack(tile);
            let (co, lout, flags) = (geom[0] as usize, geom[3] as usize, geom[13]);
            let x = into_contiguous(x);
            let w2 = into_contiguous(w2);
            let client = x.client.clone();
            let out = empty_device_contiguous_dtype(client.clone(), x.device.clone(), [co, lout].into(), x.dtype);
            let dims = upload_u32(&x, &geom[..13]);
            // 使わない入力はダミー(同じバッファを 2 か所に書き込み可能で束ねないため)
            let dummy = || client.create_from_slice(&[0u8; 16]).binding();
            let opt = |t: FloatTensor<Self>, bit: u32| if flags & bit != 0 { into_contiguous(t).handle.binding() } else { dummy() };
            // ゲート付きは 1 ブロックの行の半分ずつが conv と gate なので、1 ブロックが受け持つチャネルは半分
            let bm = if flags & 8 != 0 { 8 * tm as usize } else { 16 * tm as usize };
            let bn = 16 * tn as usize;
            client.launch(
                Box::new(SourceKernel::new(Conv { tm, tn, flags }, CubeDim::new_2d(16, 16))),
                CubeCount::new_2d(lout.div_ceil(bn) as u32, co.div_ceil(bm) as u32),
                KernelArguments::new()
                    .with_buffer(w2.handle.clone().binding())
                    .with_buffer(x.handle.clone().binding())
                    .with_buffer(opt(bias, 1))
                    .with_buffer(opt(alpha, 2))
                    .with_buffer(opt(inv, 2))
                    .with_buffer(opt(res, 4))
                    .with_buffer(out.handle.clone().binding())
                    .with_buffer(dims.binding()),
            );
            out
        }

        fn qk_norm_rope(
            y: FloatTensor<Self>,
            w: FloatTensor<Self>,
            cos: FloatTensor<Self>,
            sin: FloatTensor<Self>,
            geom: Vec<u32>,
        ) -> FloatTensor<Self> {
            let (rows, heads, hd) = (geom[0] as usize, geom[2] as usize, geom[3] as usize);
            let y = into_contiguous(y);
            let client = y.client.clone();
            let out = empty_device_contiguous_dtype(client.clone(), y.device.clone(), [rows, heads * hd].into(), y.dtype);
            let dims = upload_u32(&y, &geom);
            client.launch(
                Box::new(SourceKernel::new(QkNormRope, CubeDim::new_1d(64))),
                CubeCount::new_2d(((rows * heads).div_ceil(64)).min(65535) as u32, (rows * heads).div_ceil(64).div_ceil(65535) as u32),
                KernelArguments::new()
                    .with_buffer(y.handle.clone().binding())
                    .with_buffer(into_contiguous(w).handle.binding())
                    .with_buffer(into_contiguous(cos).handle.binding())
                    .with_buffer(into_contiguous(sin).handle.binding())
                    .with_buffer(out.handle.clone().binding())
                    .with_buffer(dims.binding()),
            );
            out
        }

        fn norm_modulate(x: FloatTensor<Self>, shift: FloatTensor<Self>, scale: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self> {
            let (rows, d) = (geom[0] as usize, geom[1] as usize);
            let x = into_contiguous(x);
            let client = x.client.clone();
            let out = empty_device_contiguous_dtype(client.clone(), x.device.clone(), [rows, d].into(), x.dtype);
            let dims = upload_u32(&x, &geom);
            client.launch(
                Box::new(SourceKernel::new(NormModulate, CubeDim::new_1d(256))),
                grid_1d(rows),
                KernelArguments::new()
                    .with_buffer(x.handle.clone().binding())
                    .with_buffer(into_contiguous(shift).handle.binding())
                    .with_buffer(into_contiguous(scale).handle.binding())
                    .with_buffer(out.handle.clone().binding())
                    .with_buffer(dims.binding()),
            );
            out
        }

        fn bn_stats(y: FloatTensor<Self>, bn_w: FloatTensor<Self>, bn_b: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self> {
            let y = into_contiguous(y);
            let c = y.meta.shape()[0];
            let client = y.client.clone();
            let ab = empty_device_contiguous_dtype(client.clone(), y.device.clone(), [2 * c].into(), y.dtype);
            let dims = upload_u32(&y, &geom);
            client.launch(
                Box::new(SourceKernel::new(BnStats, CubeDim::new_1d(256))),
                CubeCount::new_1d(c as u32),
                KernelArguments::new()
                    .with_buffer(y.handle.clone().binding())
                    .with_buffer(into_contiguous(bn_w).handle.binding())
                    .with_buffer(into_contiguous(bn_b).handle.binding())
                    .with_buffer(ab.handle.clone().binding())
                    .with_buffer(dims.binding()),
            );
            ab
        }

        fn bn_apply(y: FloatTensor<Self>, ab: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self> {
            let y = into_contiguous(y);
            let c = y.meta.shape()[0];
            let buf_len = geom[4];
            let client = y.client.clone();
            let out = empty_device_contiguous_dtype(client.clone(), y.device.clone(), [c, buf_len as usize].into(), y.dtype);
            let dims = upload_u32(&y, &geom);
            client.launch(
                Box::new(SourceKernel::new(BnApply, CubeDim::new_1d(256))),
                CubeCount::new_2d(buf_len.div_ceil(256), c as u32),
                KernelArguments::new()
                    .with_buffer(y.handle.clone().binding())
                    .with_buffer(into_contiguous(ab).handle.binding())
                    .with_buffer(out.handle.clone().binding())
                    .with_buffer(dims.binding()),
            );
            out
        }

        fn overlap_add(z: FloatTensor<Self>, bias: FloatTensor<Self>, geom: Vec<u32>) -> FloatTensor<Self> {
            let [co, _, lout, _, _] = geom[..] else { panic!("overlap_add: bad geometry {geom:?}") };
            let z = into_contiguous(z);
            let bias = into_contiguous(bias);
            let client = z.client.clone();
            let out = empty_device_contiguous_dtype(client.clone(), z.device.clone(), [co as usize, lout as usize].into(), z.dtype);
            let dims = upload_u32(&z, &geom);
            client.launch(
                Box::new(SourceKernel::new(OverlapAdd, CubeDim::new_1d(256))),
                CubeCount::new_3d(lout.div_ceil(256), co.min(65535), co.div_ceil(65535)),
                KernelArguments::new()
                    .with_buffer(z.handle.clone().binding())
                    .with_buffer(bias.handle.clone().binding())
                    .with_buffer(out.handle.clone().binding())
                    .with_buffer(dims.binding()),
            );
            out
        }
    }
}
