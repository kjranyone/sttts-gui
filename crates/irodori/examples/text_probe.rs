//! ModernBERT(テキスト条件)の GPU 初回コンパイル時間と定常速度の測定(カーネル構成の比較用)。
use std::time::Instant;

use irodori::Device;
use irodori::condition::TextConditioner;
use irodori::testing;
use irodori::tokenizer::Tokenizer;
use irodori::weights::Weights;

fn main() -> anyhow::Result<()> {
    let device = if std::env::var("DEV").as_deref() == Ok("cpu") { Device::flex() } else { testing::gpu_device() };
    let dir = testing::checkpoint_dir().expect("model");
    let w = Weights::open(dir.join("model.safetensors"))?;
    let cond = TextConditioner::load(&w, &device)?;
    let tok = Tokenizer::load(dir.join("tokenizer"))?;
    let (ids, mask) = tok.batch_encode(&["こんにちは、よろしくお願いします。".to_string()], 256, true)?;
    for i in 0..4 {
        let t = Instant::now();
        let s = cond.encode_text(&ids, &mask);
        let v = s.into_data().convert::<f32>().try_to_vec::<f32>().unwrap();
        println!("run {i}: {:.3}s (sum {:.3})", t.elapsed().as_secs_f64(), v.iter().sum::<f32>());
    }
    Ok(())
}
