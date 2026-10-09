//! GPU(wgpu / Vulkan)の基本性能の確認: アダプタ、matmul、conv1d、conv_transpose1d。
//! 実行: cargo run -p irodori --release --features gpu --example gpu_bench
//! マイクや他のエンジンは使わない。短時間で終わる。

use std::time::Instant;

use burn::tensor::{Device, Tensor, module};

fn sync<const D: usize>(t: Tensor<D>) -> f32 {
    // 結果を CPU へ読み戻して GPU の完了を待つ
    t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0]
}

fn bench<F: FnMut() -> f32>(name: &str, flops: f64, iters: usize, mut f: F) {
    let _ = f(); // ウォームアップ(カーネルのコンパイル・autotune を含む)
    let _ = f();
    let t0 = Instant::now();
    let mut acc = 0f32;
    for _ in 0..iters {
        acc += f();
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!("{name:<44} {:>8.2} ms  {:>7.1} GFLOPS  (acc={acc:.1})", dt * 1e3, flops / dt / 1e9);
}

fn main() {
    use burn::tensor::DeviceKind;
    let kind = match std::env::var("GPU_KIND").as_deref() {
        Ok("discrete") => DeviceKind::DiscreteGpu(0),
        Ok("integrated") => DeviceKind::IntegratedGpu(0),
        _ => DeviceKind::DefaultDevice,
    };
    let (device, setup) = Device::wgpu_options().device_kind(kind).init_with_setup().expect("wgpu init");
    let info = setup.adapter.get_info();
    println!("device: {device:?}");
    println!("adapter: {} ({:?}, {:?}) driver={} {}", info.name, info.device_type, info.backend, info.driver, info.driver_info);
    println!("features: {:?}", setup.adapter.features());

    // DiT の MLP 相当を 8 回つないで 1 回だけ読み戻す(読み戻しの固定費を薄める)
    use burn::tensor::FloatDType;
    for (m, k, n) in [(100usize, 1280usize, 1280usize), (256, 1280, 1280), (512, 1280, 1280), (1024, 1024, 1024)] {
        for (label, dt) in [("f32", FloatDType::F32), ("f16", FloatDType::F16), ("bf16", FloatDType::BF16)] {
            let a = Tensor::<2>::random([m, k], burn::tensor::Distribution::Default, &device).cast(dt);
            let b = Tensor::<2>::random([k, n], burn::tensor::Distribution::Default, &device).cast(dt) * 0.02;
            bench(&format!("matmul x8 {label} [{m},{k}]x[{k},{n}]"), 8.0 * 2.0 * (m * k * n) as f64, 10, || {
                let mut x = a.clone();
                for _ in 0..8 {
                    x = x.matmul(b.clone());
                    if k != n {
                        x = x.narrow(1, 0, k.min(n));
                    }
                }
                sync(x.cast(FloatDType::F32))
            });
        }
    }

    // DACVAE デコーダ相当の畳み込み
    let x = Tensor::<3>::random([1, 512, 200], burn::tensor::Distribution::Default, &device);
    let w = Tensor::<3>::random([512, 512, 7], burn::tensor::Distribution::Default, &device);
    bench("conv1d [1,512,200] k7 -> 512", 2.0 * (512 * 512 * 7 * 200) as f64, 10, || {
        let o = module::conv1d(x.clone(), w.clone(), None, burn::tensor::ops::ConvOptions::new([1], [3], [1], 1));
        sync(o)
    });
    let wt = Tensor::<3>::random([512, 256, 10], burn::tensor::Distribution::Default, &device);
    bench("conv_transpose1d [1,512,200] k10 s5", 2.0 * (512 * 256 * 10 * 200) as f64, 10, || {
        let o = module::conv_transpose1d(
            x.clone(),
            wt.clone(),
            None,
            burn::tensor::ops::ConvTransposeOptions::new([5], [3], [0], [1], 1),
        );
        sync(o)
    });
}
