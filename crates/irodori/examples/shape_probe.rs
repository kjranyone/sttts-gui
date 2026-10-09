//! 形状が変わるたびに GPU のカーネルが作り直されるか(= 初回コストが形状ごとに発生するか)を測る。
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};

fn main() {
    let dev = irodori::gpu_device();
    let w = Tensor::<2>::random([96, 672], Distribution::Default, &dev);
    let sync = |t: Tensor<2>| t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0];
    println!("--- matmul [96,672] x [672,n]: n を毎回変える");
    for n in [8192usize, 8192, 8000, 7000, 6000, 5000, 4097, 4096, 3000, 2999, 1000, 999, 8192] {
        let x = Tensor::<2>::random([672, n], Distribution::Default, &dev);
        let t = Instant::now();
        let _ = sync(w.clone().matmul(x));
        println!("n={n:5}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    println!("--- 要素ごとの演算 [1,384,n]: sin(x*a)^2 + x");
    for n in [3000usize, 3000, 2999, 2500, 2000, 1999, 1500, 3000] {
        let x = Tensor::<3>::random([1, 384, n], Distribution::Default, &dev);
        let t = Instant::now();
        let s = (x.clone() * 2.0).sin();
        let y = x + s.clone() * s;
        let _ = y.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0];
        println!("n={n:5}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    println!("--- attention [1,20,S,64] x ctx 30: S を変える");
    for s in [72usize, 72, 73, 100, 150, 188, 188, 72] {
        let q = Tensor::<4>::random([1, 20, s, 64], Distribution::Default, &dev);
        let k = Tensor::<4>::random([1, 20, s + 30, 64], Distribution::Default, &dev);
        let v = Tensor::<4>::random([1, 20, s + 30, 64], Distribution::Default, &dev);
        let t = Instant::now();
        let o = burn::tensor::module::attention(q, k, v, None, None, Default::default());
        let _ = o.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0];
        println!("S={s:4}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}
