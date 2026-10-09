//! Irodori-TTS の CLI(純 Rust)。段階ごとの所要時間を表示し、WAV を書き出す。
//!
//!   cargo run -p irodori --release --features gpu --example tts -- \
//!       --text "こんにちは" [--caption "落ち着いた声で"] [--ref ref.wav] [--seed 1] \
//!       [--device gpu|cpu] [--repeat 3] [--out out.wav] [--no-watermark]
//!
//! `--repeat` は同じ設定で繰り返し、2 回目以降(カーネルのコンパイル後)の速度を見るためのもの。

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use irodori::Device;
use irodori::pipeline::{SamplingRequest, Tts, TtsPaths};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut req = SamplingRequest::default();
    let mut device_kind = "gpu".to_string();
    let mut repeat = 1usize;
    let mut warmup = false;
    let mut texts: Vec<String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    req.no_ref = true;
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().with_context(|| format!("{name} needs a value"));
        match a.as_str() {
            "--text" => texts.push(val("--text")?),
            "--caption" => req.caption = Some(val("--caption")?),
            "--ref" => {
                req.ref_wav = Some(PathBuf::from(val("--ref")?));
                req.no_ref = false;
            }
            "--seed" => req.seed = Some(val("--seed")?.parse()?),
            "--steps" => req.num_steps = val("--steps")?.parse()?,
            "--seconds" => req.seconds = Some(val("--seconds")?.parse()?),
            "--device" => device_kind = val("--device")?,
            "--repeat" => repeat = val("--repeat")?.parse()?,
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "--no-watermark" => req.watermark = false,
            "--warmup" => warmup = true,
            other => bail!("unknown argument {other}"),
        }
    }
    if texts.is_empty() {
        bail!("--text is required (repeatable; the texts are cycled over --repeat)");
    }

    let device = match device_kind.as_str() {
        "cpu" => Device::flex(),
        #[cfg(feature = "_gpu")]
        "gpu" => irodori::gpu_device(),
        other => bail!("unsupported --device {other} (cpu{})", if cfg!(feature = "_gpu") { " / gpu" } else { "; build with --features gpu for gpu" }),
    };
    eprintln!("device: {device:?}");

    let t0 = std::time::Instant::now();
    let paths = TtsPaths::from_hf_cache()?;
    let tts = Tts::load(&paths, &device)?;
    eprintln!("load: {:.1}s (watermark: {})", t0.elapsed().as_secs_f64(), tts.has_watermark());

    if warmup {
        let t = std::time::Instant::now();
        tts.warmup()?;
        eprintln!("warmup: {:.1}s", t.elapsed().as_secs_f64());
    }
    for i in 0..repeat {
        req.text = texts[i % texts.len()].clone();
        let t = std::time::Instant::now();
        let res = tts.synthesize(&req)?;
        let wall = t.elapsed().as_secs_f64();
        let dur = res.audio.len() as f64 / res.sample_rate as f64;
        let stages: Vec<String> = res.timings.iter().map(|(n, s)| format!("{n} {s:.2}s")).collect();
        eprintln!("#{i} [{} chars]: {wall:.2}s for {dur:.2}s of audio (RTF {:.2}) seed={} | {}", req.text.chars().count(), wall / dur, res.used_seed, stages.join(", "));
        for m in &res.messages {
            eprintln!("   {m}");
        }
        if i + 1 == repeat {
            if let Some(path) = &out {
                let spec = hound::WavSpec { channels: 1, sample_rate: res.sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
                let mut w = hound::WavWriter::create(path, spec)?;
                for s in &res.audio {
                    w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                }
                w.finalize()?;
                eprintln!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}
