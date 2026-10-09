//! モデルが使う演算を CPU(flex)と GPU(wgpu)で同じ入力に対して実行し、結果を比べる。
//! 実行: cargo run -p irodori --release --features gpu --example gpu_ops_check
//! どの演算が GPU で食い違うか(burn-wgpu の不具合・レイアウトの扱い)を特定するための診断。

use burn::tensor::activation;
use burn::tensor::{Device, DeviceKind, Int, Tensor, TensorData};

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    // 決定的な疑似乱数 [-1, 1)
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn t<const D: usize>(shape: [usize; D], seed: u64, dev: &Device) -> Tensor<D> {
    let n: usize = shape.iter().product();
    Tensor::<D>::from_data(TensorData::new(rnd(n, seed), shape.to_vec()), dev)
}

fn out<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

type Op = (&'static str, fn(&Device) -> Vec<f32>);

fn main() {
    let cpu = Device::flex();
    use burn::tensor::wgpu::WgpuBackend;
    let api = match std::env::var("GPU_API").as_deref() {
        Ok("vulkan") => WgpuBackend::Vulkan,
        Ok("webgpu") => WgpuBackend::WebGpu,
        _ => WgpuBackend::Auto,
    };
    let (gpu, setup) = Device::wgpu_options()
        .device_kind(DeviceKind::DiscreteGpu(0))
        .graphics_api(api)
        .init_with_setup()
        .expect("wgpu");
    let info = setup.adapter.get_info();
    println!("adapter: {} backend={:?} (api request: {api:?})", info.name, info.backend);

    let ops: Vec<Op> = vec![
        ("matmul 2D", |d| out(t([37, 53], 1, d).matmul(t([53, 29], 2, d)))),
        ("matmul with transposed weight [out,in]^T", |d| out(t([37, 53], 1, d).matmul(t([29, 53], 2, d).transpose()))),
        ("matmul 3D x 3D batched", |d| out(t([3, 17, 23], 1, d).matmul(t([3, 23, 19], 2, d)))),
        ("matmul 4D batched after swap_dims", |d| {
            let q = t([2, 11, 4, 8], 1, d).swap_dims(1, 2); // [B,H,S,D]
            let k = t([2, 13, 4, 8], 2, d).swap_dims(1, 2);
            out(q.matmul(k.swap_dims(2, 3)))
        }),
        ("reshape after swap_dims", |d| out(t([2, 5, 4, 8], 1, d).swap_dims(1, 2).reshape([2, 4, 40]))),
        ("narrow dim1", |d| out(t([2, 10, 6], 1, d).narrow(1, 3, 4))),
        ("narrow dim2", |d| out(t([2, 10, 6], 1, d).narrow(2, 1, 4))),
        ("narrow then reshape", |d| out(t([2, 10, 6], 1, d).narrow(2, 1, 4).reshape([20, 4]))),
        ("cat dim1", |d| out(Tensor::cat(vec![t([2, 3, 4], 1, d), t([2, 5, 4], 2, d)], 1))),
        ("cat dim2", |d| out(Tensor::cat(vec![t([2, 3, 4], 1, d), t([2, 3, 5], 2, d)], 2))),
        ("select (embedding)", |d| {
            let w = t([100, 16], 1, d);
            let ids = Tensor::<1, Int>::from_data(TensorData::new(vec![3i64, 99, 0, 42, 42, 7], vec![6]), d);
            out(w.select(0, ids))
        }),
        ("mean_dim", |d| out(t([4, 9, 33], 1, d).mean_dim(2))),
        ("sum_dim", |d| out(t([4, 9, 33], 1, d).sum_dim(1))),
        ("max_dim", |d| out(t([4, 9, 33], 1, d).max_dim(2))),
        ("min_dim", |d| out(t([4, 9, 33], 1, d).min_dim(1))),
        ("powf 2.0", |d| out(t([4, 9, 33], 1, d).powf_scalar(2.0))),
        ("sqrt(abs)+recip", |d| out((t([4, 9, 33], 1, d).abs() + 0.5).sqrt().recip())),
        ("sin/cos", |d| out(t([4, 9, 33], 1, d).mul_scalar(5.0).sin() + t([4, 9, 33], 1, d).mul_scalar(5.0).cos())),
        ("exp/tanh", |d| out(t([4, 9, 33], 1, d).exp() + t([4, 9, 33], 2, d).tanh())),
        ("silu", |d| out(activation::silu(t([4, 9, 33], 1, d)))),
        ("gelu", |d| out(activation::gelu(t([4, 9, 33], 1, d)))),
        ("softplus-ish log1p exp", |d| out((t([4, 9, 33], 1, d).exp() + 1.0).log())),
        ("clamp_min", |d| out(t([4, 9, 33], 1, d).clamp_min(0.1))),
        ("expand", |d| out(t([1, 1, 6], 1, d).expand([3, 5, 6]))),
        ("broadcast mul [B,S,D]*[1,1,D]", |d| out(t([3, 5, 6], 1, d) * t([1, 1, 6], 2, d))),
        ("broadcast add [B,S,D]+[B,1,D]", |d| out(t([3, 5, 6], 1, d) + t([3, 1, 6], 2, d))),
        ("attention (no bias)", |d| {
            let q = t([2, 4, 11, 8], 1, d);
            let k = t([2, 4, 13, 8], 2, d);
            let v = t([2, 4, 13, 8], 3, d);
            out(burn::tensor::module::attention(q, k, v, None, None, Default::default()))
        }),
        ("attention (additive bias [B,1,1,Sk])", |d| {
            let q = t([2, 4, 11, 8], 1, d);
            let k = t([2, 4, 13, 8], 2, d);
            let v = t([2, 4, 13, 8], 3, d);
            let mut bias = vec![0f32; 2 * 13];
            for i in 9..13 {
                bias[i] = -1e9;
                bias[13 + i - 2] = -1e9;
            }
            let b = Tensor::<4>::from_data(TensorData::new(bias, vec![2, 1, 1, 13]), d).expand([2, 4, 11, 13]);
            out(burn::tensor::module::attention(q, k, v, None, Some(b), Default::default()))
        }),
        ("softmax", |d| out(activation::softmax(t([2, 4, 11, 13], 1, d), 3))),
        ("conv1d", |d| {
            out(burn::tensor::module::conv1d(
                t([1, 8, 50], 1, d),
                t([6, 8, 7], 2, d),
                None,
                burn::tensor::ops::ConvOptions::new([1], [3], [1], 1),
            ))
        }),
        ("conv2d", |d| {
            out(burn::tensor::module::conv2d(
                t([1, 4, 20, 30], 1, d),
                t([6, 4, 3, 3], 2, d),
                None,
                burn::tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
            ))
        }),
        ("conv_transpose1d", |d| {
            out(burn::tensor::module::conv_transpose1d(
                t([1, 8, 20], 1, d),
                t([8, 6, 10], 2, d),
                None,
                burn::tensor::ops::ConvTransposeOptions::new([5], [3], [0], [1], 1),
            ))
        }),
    ];

    let mut bad = 0;
    for (name, f) in &ops {
        let a = f(&cpu);
        let b = f(&gpu);
        let (mut d, mut m) = (0f32, 0f32);
        for (x, y) in a.iter().zip(&b) {
            d = d.max((x - y).abs());
            m = m.max(x.abs());
        }
        let ok = a.len() == b.len() && d <= 1e-4 * m.max(1.0);
        if !ok {
            bad += 1;
        }
        println!("{:<4} {name:<46} len {}/{}  max|diff|={d:.3e}  max|cpu|={m:.3e}", if ok { "ok" } else { "BAD" }, a.len(), b.len());
    }
    println!("{bad} op(s) differ");
}
