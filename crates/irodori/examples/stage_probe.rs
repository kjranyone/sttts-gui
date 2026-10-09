//! 各段階の GPU 時間の内訳(アップロード / 実行 / 読み戻し)を測る。性能調査用。
use std::time::Instant;

use burn::tensor::{Tensor, TensorData};
use irodori::condition::TextConditioner;
use irodori::config::ModelConfig;
use irodori::duration::{DurationInputs, DurationPredictor};
use irodori::testing;
use irodori::weights::Weights;

fn sync<const D: usize>(t: &Tensor<D>) -> f32 {
    t.clone().into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0]
}

fn main() -> anyhow::Result<()> {
    let dev = testing::gpu_device();
    let dir = testing::checkpoint_dir().expect("model");
    let w = Weights::open(dir.join("model.safetensors"))?;
    let cfg = ModelConfig::from_weights(&w)?;
    let dp = DurationPredictor::load(&w, &cfg, &dev)?;
    let _cond = TextConditioner::load(&w, &dev)?; // 常駐させて実運用に近いメモリ状況にする

    let mk3 = |b: usize, s: usize, d: usize| Tensor::<3>::from_data(TensorData::new(vec![0.1f32; b * s * d], vec![b, s, d]), &dev);
    let mk2 = |b: usize, s: usize| Tensor::<2>::from_data(TensorData::new(vec![1.0f32; b * s], vec![b, s]), &dev);
    let mk1 = |b: usize| Tensor::<1>::from_data(TensorData::new(vec![1.0f32; b], vec![b]), &dev);
    for round in 0..6 {
        let t = Instant::now();
        let inputs = DurationInputs {
            text_state: mk3(1, 256, 512),
            text_mask: mk2(1, 256),
            speaker_state: mk3(1, 2, 768),
            has_speaker: mk1(1),
            caption_state: mk3(1, 512, 512),
            caption_mask: mk2(1, 512),
            has_caption: mk1(1),
        };
        let _ = sync(&inputs.has_caption);
        let t_up = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let out = dp.predict_log_frames(&inputs);
        let t_build = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let v = sync(&out);
        let t_run = t.elapsed().as_secs_f64();
        // 比較用: 空の同期だけの往復
        let t = Instant::now();
        let _ = sync(&mk1(1));
        let t_rt = t.elapsed().as_secs_f64();
        println!("round {round}: upload {t_up:.4}s | graph build {t_build:.4}s | run+readback {t_run:.4}s | empty roundtrip {t_rt:.4}s (out {v:.3})");
    }
    Ok(())
}
