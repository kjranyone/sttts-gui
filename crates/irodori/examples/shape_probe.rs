//! 4D テンソルの零パディングの方法ごとの GPU 時間(device.sync で完了を待つ)。性能調査用。
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};

fn bench<const D: usize>(name: &str, dev: &burn::tensor::Device, mut f: impl FnMut() -> Tensor<D>) {
    let _ = f();
    let _ = dev.sync();
    let reps = 3;
    let t0 = Instant::now();
    let mut keep = Vec::new();
    for _ in 0..reps {
        keep.push(f());
    }
    let _ = dev.sync();
    println!("{name:<64} {:8.2} ms", t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
}

fn main() {
    let dev = irodori::gpu_device();
    let (c, h, w) = (96usize, 2049usize, 310usize);
    let x = Tensor::<4>::random([1, c, h, w], Distribution::Default, &dev);
    bench("cat rows(dim2) then cols(dim3)  [現行]", &dev, || {
        let zr = Tensor::<4>::zeros([1, c, 1, w], &dev);
        let y = Tensor::cat(vec![zr.clone(), x.clone(), zr], 2);
        let zc = Tensor::<4>::zeros([1, c, h + 2, 1], &dev);
        Tensor::cat(vec![zc.clone(), y, zc], 3)
    });
    bench("burn pad((1,1,1,1))", &dev, || x.clone().pad((1, 1, 1, 1), 0.0));
    bench("zeros + slice_assign", &dev, || {
        Tensor::<4>::zeros([1, c, h + 2, w + 2], &dev).slice_assign([0..1, 0..c, 1..h + 1, 1..w + 1], x.clone())
    });
    bench("reshape [c, h*w] only", &dev, || x.clone().reshape([c, h * w]) + 0.0);
    let flat = x.clone().reshape([c, h * w]);
    bench("pad flat [c, h*w] by (left=wp+1, right=wp+1+tail) via cat", &dev, || {
        let z = Tensor::<2>::zeros([c, 400], &dev);
        Tensor::cat(vec![z.clone(), flat.clone(), z], 1)
    });
    let big = Tensor::<2>::random([c, 2051 * 312 + 1000], Distribution::Default, &dev);
    bench("narrow 9 taps of big + cat(dim0) one chunk 65536", &dev, || {
        let parts: Vec<Tensor<2>> = (0..9).map(|i| big.clone().narrow(1, 1000 + i * 313, 65536)).collect();
        Tensor::cat(parts, 0)
    });
}
