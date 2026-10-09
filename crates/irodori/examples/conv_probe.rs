//! conv1d の GPU 性能調査: burn の conv1d と、im2col + matmul の自前実装を比べる。
use std::time::Instant;

use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, DeviceKind, Distribution, Tensor};

fn sync<const D: usize>(t: Tensor<D>) -> f32 {
    t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0]
}

/// 自前: ゼロ詰め → k 本のずらしスライスを積んで matmul([Cout, C*k] x [C*k, L])
fn conv1d_mm(x: Tensor<3>, w: Tensor<3>, pad: usize, dil: usize) -> Tensor<3> {
    let [b, c, l] = x.dims();
    let [co, _ci, k] = w.dims();
    assert_eq!(b, 1);
    let lout = l + 2 * pad - dil * (k - 1);
    let dev = x.device();
    let xp = Tensor::cat(vec![Tensor::<3>::zeros([1, c, pad], &dev), x, Tensor::<3>::zeros([1, c, pad], &dev)], 2);
    let cols: Vec<Tensor<3>> = (0..k).map(|j| xp.clone().narrow(2, j * dil, lout)).collect();
    // [1, k*c, lout] の並びは (j, c) -> 重みも [co, k*c] に並べ替える(w: [co, c, k] -> [co, k, c])
    let cols = Tensor::cat(cols, 1).reshape([k * c, lout]);
    let w2 = w.swap_dims(1, 2).reshape([co, k * c]);
    w2.matmul(cols).reshape([1, co, lout])
}

fn main() {
    let device = Device::wgpu_options().device_kind(DeviceKind::DiscreteGpu(0)).init().expect("wgpu");
    for (c, co, l, k, dil) in [(384usize, 384usize, 3000usize, 7usize, 3usize), (768, 768, 300, 7, 1), (96, 96, 138240, 7, 1), (384, 384, 3000, 1, 1)] {
        let pad = (k - 1) * dil / 2;
        let x = Tensor::<3>::random([1, c, l], Distribution::Default, &device);
        let w = Tensor::<3>::random([co, c, k], Distribution::Default, &device) * 0.02;
        let flops = 2.0 * (co * c * k * l) as f64;
        println!("--- C={c} Cout={co} L={l} k={k} dil={dil}  ({:.2} GFLOP)", flops / 1e9);
        for round in 0..3 {
            let t = Instant::now();
            let a = sync(conv1d(x.clone(), w.clone(), None, ConvOptions::new([1], [pad], [dil], 1)));
            let dt_burn = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let b = sync(conv1d_mm(x.clone(), w.clone(), pad, dil));
            let dt_mm = t.elapsed().as_secs_f64();
            println!("round {round}: burn conv1d {:8.3}s ({:7.1} GFLOPS) | im2col+matmul {:8.3}s ({:7.1} GFLOPS)  first-elem {a:.4}/{b:.4}", dt_burn, flops / dt_burn / 1e9, dt_mm, flops / dt_mm / 1e9);
        }
    }
}
