//! WAV ファイルをマイクの代わりに流す音声ソース(テスト・ベンチ用)。
//!
//! MicStream と同じ形(start / stop、on_block に 16kHz mono f32 を 30ms ずつ渡す)。
//! 先頭に lead_s、各ファイルの後ろに gap_s 秒の無音を足し、VAD の発話終了を確実に発生させる。

use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::resample::resample_all;
use crate::{BLOCK_SECONDS, TARGET_RATE};

/// WAV を読み、モノラル平均 → 16kHz にして返す。
pub fn load_wav_16k(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let mut r = hound::WavReader::open(path)
        .with_context(|| format!("WAV を開けません: {}", path.display()))?;
    let spec = r.spec();
    let ch = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = if ch == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    resample_all(&mono, spec.sample_rate, TARGET_RATE)
}

#[derive(Debug, Clone)]
pub struct WavSourceOptions {
    pub gap_s: f64,
    pub lead_s: f64,
    /// true: 実時間ペース(ブロック末尾の時刻に届ける)。false: 可能な限り高速。
    pub realtime: bool,
}

impl Default for WavSourceOptions {
    fn default() -> Self {
        Self {
            gap_s: 2.0,
            lead_s: 0.5,
            realtime: true,
        }
    }
}

type BlockCb = Box<dyn FnMut(Vec<f32>) + Send>;
type EofCb = Box<dyn FnOnce() + Send>;

pub struct WavSource {
    paths: Vec<String>,
    opts: WavSourceOptions,
    on_block: Mutex<Option<BlockCb>>,
    on_eof: Mutex<Option<EofCb>>,
    stop: Arc<AtomicBool>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl WavSource {
    pub fn new(
        paths: Vec<String>,
        opts: WavSourceOptions,
        on_block: impl FnMut(Vec<f32>) + Send + 'static,
        on_eof: Option<EofCb>,
    ) -> Self {
        Self {
            paths,
            opts,
            on_block: Mutex::new(Some(Box::new(on_block))),
            on_eof: Mutex::new(on_eof),
            stop: Arc::new(AtomicBool::new(false)),
            handle: Mutex::new(None),
        }
    }

    /// 全ファイルを読み込んでから送出スレッドを起動する。戻り値は 16000。
    pub fn start(&self) -> Result<u32> {
        let mut clips = Vec::new();
        for p in &self.paths {
            clips.push(load_wav_16k(p)?);
        }
        let mut cb = self
            .on_block
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .context("WavSource は既に開始されています")?;
        let on_eof = self.on_eof.lock().unwrap_or_else(|p| p.into_inner()).take();
        let opts = self.opts.clone();
        let stop = self.stop.clone();
        let h = std::thread::Builder::new()
            .name("wav-source".into())
            .spawn(move || {
                let block = (TARGET_RATE as f64 * BLOCK_SECONDS) as usize;
                let mut stream = vec![0.0f32; (TARGET_RATE as f64 * opts.lead_s) as usize];
                for clip in clips {
                    stream.extend_from_slice(&clip);
                    let n = stream.len() + (TARGET_RATE as f64 * opts.gap_s) as usize;
                    stream.resize(n, 0.0);
                }
                let t0 = Instant::now();
                let mut start = 0;
                while start < stream.len() {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    let end = (start + block).min(stream.len());
                    if opts.realtime {
                        // ブロック末尾の時刻まで待つ(マイクと同じく「録り終わった瞬間」に届く)
                        let due = t0
                            + Duration::from_secs_f64((start + block) as f64 / TARGET_RATE as f64);
                        let now = Instant::now();
                        if due > now {
                            std::thread::sleep(due - now);
                        }
                    }
                    cb(stream[start..end].to_vec());
                    start = end;
                }
                if !stop.load(Ordering::SeqCst)
                    && let Some(f) = on_eof
                {
                    f();
                }
            })?;
        *self.handle.lock().unwrap_or_else(|p| p.into_inner()) = Some(h);
        Ok(TARGET_RATE)
    }

    /// 停止して送出スレッドを join する。冪等。
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = h.join();
        }
    }
}

impl Drop for WavSource {
    fn drop(&mut self) {
        self.stop();
    }
}
