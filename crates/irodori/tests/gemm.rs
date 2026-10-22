//! 16 ビットの重みの行列積([`irodori::gemm::matmul_half`])を、同じ値の f32 の行列積(burn)と突き合わせる。
//! GPU のみ(`IRODORI_DEVICE=gpu`)。行が少ない形(adaLN の `[1, dim]`)と K の分割も含める。

use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData};
use irodori::testing::{assert_close, device, to_vec};

fn values(n: usize, seed: u32) -> Vec<f32> {
    // 決定的な疑似乱数(-1..1)
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn half_weights_matmul_matches_f32() {
    let dev = device();
    if dev == Device::flex() {
        eprintln!("skip: 16 ビットの重みは GPU のみ(IRODORI_DEVICE=gpu)");
        return;
    }
    // (M, K, N)。K が大きく出力が小さい形は K を分割する
    for (m, k, n) in [(1, 1024, 3072), (5, 640, 1024), (16, 1024, 1024), (37, 2816, 1024), (200, 1024, 4096), (3, 4096, 64)] {
        let a = values(m * k, 1);
        let w: Vec<half::f16> = values(n * k, 2).into_iter().map(half::f16::from_f32).collect();
        let wf: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
        let at = Tensor::<2>::from_data(TensorData::new(a, [m, k]), &dev);
        let wh = Tensor::<2>::from_data(TensorData::new(w, [n, k]), TensorCreationOptions::new(dev.clone()).with_dtype(DType::F16));
        let got = to_vec(irodori::gemm::matmul_half(at.clone(), &wh));
        let want = to_vec(at.matmul(Tensor::<2>::from_data(TensorData::new(wf, [n, k]), &dev).transpose()));
        assert_close(&format!("[{m},{k}] x [{n},{k}]^T"), &got, &want, 1e-5);
    }
}
