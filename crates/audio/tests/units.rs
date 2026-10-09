//! リサンプラ・ブロック分割・WavSource・MicStream(実デバイスは開かない)・デバイス列挙の単体テスト。

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sttts_audio::{
    BlockPipeline, MicStream, StreamResampler, WavSource, WavSourceOptions, list_input_devices,
    list_output_devices, load_wav_16k, resample_all,
};

fn sine(rate: u32, hz: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
        .collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// ---------- リサンプラ ----------

/// 正弦波を最小二乗で当てはめ、振幅を返す
fn fit_sine(y: &[f32], out_rate: u32, hz: f32) -> f64 {
    let w = 2.0 * std::f64::consts::PI * hz as f64 / out_rate as f64;
    let (mut s, mut c) = (0.0, 0.0);
    for (i, v) in y.iter().enumerate() {
        s += *v as f64 * (w * i as f64).sin();
        c += *v as f64 * (w * i as f64).cos();
    }
    2.0 * (s * s + c * c).sqrt() / y.len() as f64
}

/// 48k -> 16k の正弦波が、振幅・位相とも理想値に近いこと(遅延補償込み)
#[test]
fn sine_48k_to_16k_matches_ideal() {
    for hz in [200.0, 1000.0, 3000.0] {
        let y = resample_all(&sine(48000, hz, 48000), 48000, 16000).unwrap();
        assert_eq!(y.len(), 16000);
        let amp = fit_sine(&y[400..15000], 16000, hz);
        assert!((amp - 0.5).abs() < 1e-3, "{hz}Hz 振幅 {amp}");
    }
    // 位相: 理想波との最大誤差(1/3 サンプル程度のずれは許容)
    let y = resample_all(&sine(48000, 1000.0, 48000), 48000, 16000).unwrap();
    let err = max_err(&y[400..15600], &sine(16000, 1000.0, 16000)[400..15600]);
    assert!(err < 0.1, "最大誤差 {err}");
}

#[test]
fn ratio_44100() {
    let y = resample_all(&sine(44100, 440.0, 44100), 44100, 16000).unwrap();
    assert_eq!(y.len(), 16000);
    let amp = fit_sine(&y[400..15000], 16000, 440.0);
    assert!((amp - 0.5).abs() < 1e-3, "振幅 {amp}");
    let err = max_err(&y[400..15600], &sine(16000, 440.0, 16000)[400..15600]);
    assert!(err < 0.1, "最大誤差 {err}");
}

/// 入力の切り方に依存せず同じ出力になる(ストリーミング性)
#[test]
fn chunking_invariant() {
    let x = sine(48000, 700.0, 48000);
    let mut a = StreamResampler::new(48000, 16000, 1440).unwrap();
    let mut ya = Vec::new();
    a.process(&x, &mut ya).unwrap();
    let mut b = StreamResampler::new(48000, 16000, 1440).unwrap();
    let mut yb = Vec::new();
    for c in x.chunks(317) {
        b.process(c, &mut yb).unwrap();
    }
    assert_eq!(ya, yb);
}

#[test]
fn same_rate_passthrough() {
    let x = sine(16000, 300.0, 1000);
    assert_eq!(resample_all(&x, 16000, 16000).unwrap(), x);
    let mut r = StreamResampler::new(16000, 16000, 480).unwrap();
    let mut y = Vec::new();
    r.process(&x[..7], &mut y).unwrap();
    assert_eq!(y, &x[..7]);
}

// ---------- BlockPipeline(cpal コールバックの代わりに直接叩く) ----------

/// ステレオ 48k を不揃いなコールバック長で流しても、480 サンプル(30ms@16k)のブロックになる
#[test]
fn pipeline_blocks_and_downmix() {
    let got: Arc<Mutex<Vec<Vec<f32>>>> = Arc::default();
    let g = got.clone();
    let mut p = BlockPipeline::new(48000, 2, move |b| g.lock().unwrap().push(b)).unwrap();
    // L=0.5, R=0.3 の直流 → 平均 0.4
    let data: Vec<f32> = (0..48000).flat_map(|_| [0.5f32, 0.3]).collect();
    for c in data.chunks(2 * 441) {
        p.push_interleaved(c);
    }
    let blocks = got.lock().unwrap();
    // 1 秒 = 33 チャンク(480 出力) - 先頭の遅延補償 120 サンプル → 32 ブロック
    assert_eq!(blocks.len(), 32);
    assert!(blocks.iter().all(|b| b.len() == 480));
    let last = blocks.last().unwrap();
    assert!(
        last.iter().all(|v| (v - 0.4).abs() < 0.01),
        "{:?}",
        &last[..4]
    );
}

#[test]
fn pipeline_mono_44100() {
    let got: Arc<Mutex<usize>> = Arc::default();
    let g = got.clone();
    let mut p = BlockPipeline::new(44100, 1, move |b| {
        assert_eq!(b.len(), 480);
        *g.lock().unwrap() += 1;
    })
    .unwrap();
    p.push_interleaved(&vec![0.1; 44100]);
    let n = *got.lock().unwrap();
    assert!((31..=33).contains(&n), "{n} ブロック");
}

// ---------- MicStream(実デバイスは開かない) ----------

/// 未開始の stop は冪等、不正な index は open 前に失敗し、状態は「停止中」のまま
#[test]
fn mic_stop_idempotent_and_bad_index() {
    let m = MicStream::new();
    m.stop();
    m.stop();
    assert!(!m.is_running());
    assert!(m.start(Some(-1), |_| {}).is_err());
    assert!(m.start(Some(99_999), |_| {}).is_err());
    assert!(!m.is_running());
    m.stop();
}

/// 列挙のみ。環境依存なので件数は問わず整合だけ見る。
#[test]
fn enumerate_devices() {
    for list in [list_input_devices(), list_output_devices()] {
        eprintln!("{list:#?}");
        assert!(list.iter().filter(|d| d.is_default).count() <= 1);
        assert!(list.windows(2).all(|w| w[0].index < w[1].index));
    }
}

// ---------- WavSource ----------

fn write_wav(name: &str, rate: u32, channels: u16, frames: usize) -> String {
    let p = std::env::temp_dir().join(format!("sttts_audio_{}_{name}.wav", std::process::id()));
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&p, spec).unwrap();
    for i in 0..frames {
        let v = (0.5 * (i as f32 * 0.05).sin() * 32767.0) as i16;
        for _ in 0..channels {
            w.write_sample(v).unwrap();
        }
    }
    w.finalize().unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn wav_load_converts_to_16k_mono() {
    let p = write_wav("load", 48000, 2, 48000);
    let a = load_wav_16k(&p).unwrap();
    assert_eq!(a.len(), 16000);
    std::fs::remove_file(p).unwrap();
}

#[test]
fn wav_streams_all_blocks_then_eof() {
    let p = write_wav("eof", 16000, 1, 16000);
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (etx, erx) = mpsc::channel::<()>();
    let src = WavSource::new(
        vec![p.clone()],
        WavSourceOptions {
            gap_s: 0.5,
            lead_s: 0.25,
            realtime: false,
        },
        move |b| tx.send(b).unwrap(),
        Some(Box::new(move || etx.send(()).unwrap())),
    );
    assert_eq!(src.start().unwrap(), 16000);
    erx.recv_timeout(Duration::from_secs(5)).unwrap();
    src.stop();
    let blocks: Vec<Vec<f32>> = rx.try_iter().collect();
    let total: usize = blocks.iter().map(Vec::len).sum();
    assert_eq!(total, 4000 + 16000 + 8000);
    assert!(blocks[..blocks.len() - 1].iter().all(|b| b.len() == 480));
    // lead は無音、クリップ部分は非無音
    assert!(blocks[0].iter().all(|v| *v == 0.0));
    assert!(blocks.iter().any(|b| b.iter().any(|v| v.abs() > 0.1)));
    std::fs::remove_file(p).unwrap();
}

#[test]
fn wav_realtime_pacing() {
    let p = write_wav("rt", 16000, 1, 8000);
    let (etx, erx) = mpsc::channel::<()>();
    let src = WavSource::new(
        vec![p.clone()],
        WavSourceOptions {
            gap_s: 0.0,
            lead_s: 0.0,
            realtime: true,
        },
        |_| {},
        Some(Box::new(move || etx.send(()).unwrap())),
    );
    let t = Instant::now();
    src.start().unwrap();
    erx.recv_timeout(Duration::from_secs(5)).unwrap();
    let dt = t.elapsed().as_secs_f64();
    assert!((0.45..0.9).contains(&dt), "0.5 秒の音声が {dt} 秒で流れた");
    src.stop();
    std::fs::remove_file(p).unwrap();
}

#[test]
fn wav_stop_midway_skips_eof_and_joins() {
    let p = write_wav("stop", 16000, 1, 16000 * 5);
    let (etx, erx) = mpsc::channel::<()>();
    let src = WavSource::new(
        vec![p.clone()],
        WavSourceOptions::default(),
        |_| {},
        Some(Box::new(move || etx.send(()).unwrap())),
    );
    src.start().unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let t = Instant::now();
    src.stop();
    src.stop();
    assert!(t.elapsed() < Duration::from_secs(1));
    assert!(erx.try_recv().is_err());
    std::fs::remove_file(p).unwrap();
}
