//! 透かし単体の GPU 時間(他のモデルの重みを載せない状態)。メモリ圧迫の影響を切り分ける性能調査用。
use std::time::Instant;

use irodori::testing;
use irodori::watermark::{IRODORI_PAYLOAD, Watermarker};

fn main() -> anyhow::Result<()> {
    let dev = irodori::gpu_device();
    let dir = testing::hf_snapshot("models--sony--silentcipher").expect("silentcipher").join("44_1_khz/73999_iteration");
    let wm = Watermarker::load(&dir, &dev)?;
    let secs: f32 = std::env::var("SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(14.0);
    let n = (48000.0 * secs) as usize;
    let audio: Vec<f32> = (0..n).map(|i| 0.1 * ((i as f32) * 0.01).sin() + 0.05 * ((i as f32) * 0.173).sin()).collect();
    for r in 0..3 {
        let t = Instant::now();
        let out = wm.encode(&audio, 48000, &IRODORI_PAYLOAD)?;
        println!("run {r}: {:.3}s for {secs}s of audio (len {})", t.elapsed().as_secs_f64(), out.len());
    }
    Ok(())
}
