//! マイク入力(cpal / WASAPI 共有モード、または ASIO)。
//!
//! デバイス既定レートで開き、30ms ブロックに区切って 16kHz モノラルへリサンプルする。
//! 排他モードや 16k 直開きは Windows で失敗しやすいため避ける。ASIO は選んだチャンネルを
//! ドライバ設定のレート・バッファで開く(`OpenedDevice::stream_config`)。
//!
//! cpal の `Stream` は専用スレッドが所有する。`stop()` はそのスレッドを join するので、
//! 戻り時点で `Stream` は drop 済み(= デバイス解放済み)。デバイスの短時間反復 open/close は
//! BugCheck 0xD1 の実績があるため、`start`/`stop` は 1 つのロックで直列化し、重複 start は拒否する。
//!
//! cpal に触れる部分は薄く保ち、変換ロジックは `BlockPipeline`(コールバックを直接叩ける)に置く。

use std::sync::Mutex;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{FromSample, I24, SampleFormat, SizedSample};

use crate::devices::{OpenedDevice, open_input_device};
use crate::resample::StreamResampler;
use crate::{BLOCK_SECONDS, TARGET_RATE};

/// デバイスレートの生サンプル(インターリーブ)→ モノラル → 16kHz → 30ms ブロック → `on_block`。
pub struct BlockPipeline {
    channels: usize,
    /// モノラル化に使うチャンネル(フレーム内の位置)
    pick: Vec<usize>,
    mono: Vec<f32>,
    resampler: StreamResampler,
    out: Vec<f32>,
    out_block: usize,
    on_block: Box<dyn FnMut(Vec<f32>) + Send>,
}

impl BlockPipeline {
    /// 全チャンネルを平均してモノラルにする。
    pub fn new(
        device_rate: u32,
        channels: u16,
        on_block: impl FnMut(Vec<f32>) + Send + 'static,
    ) -> Result<Self> {
        let all = (0..channels.max(1) as usize).collect();
        Self::with_pick(device_rate, channels, all, on_block)
    }

    /// `pick` のチャンネル(フレーム内の 0 始まりの位置)だけを平均してモノラルにする。
    /// ASIO で「入力 3 だけ」のように使うチャンネルを選ぶときに使う。
    pub fn with_pick(
        device_rate: u32,
        channels: u16,
        pick: Vec<usize>,
        on_block: impl FnMut(Vec<f32>) + Send + 'static,
    ) -> Result<Self> {
        let channels = channels.max(1) as usize;
        if pick.is_empty() || pick.iter().any(|&c| c >= channels) {
            return Err(anyhow!("チャンネル指定が不正です: {pick:?}({channels} チャンネル)"));
        }
        // 入力側のチャンクは 30ms 相当(rubato が比に合わせて丸める)
        let chunk = (device_rate as f64 * BLOCK_SECONDS) as usize;
        Ok(Self {
            channels,
            pick,
            mono: Vec::new(),
            resampler: StreamResampler::new(device_rate, TARGET_RATE, chunk)?,
            out: Vec::new(),
            out_block: (TARGET_RATE as f64 * BLOCK_SECONDS) as usize,
            on_block: Box::new(on_block),
        })
    }

    /// cpal のデータコールバックから呼ぶ。`data` は f32 のインターリーブ。
    /// 選んだチャンネルを平均してモノラルにする(既定は全チャンネル = WASAPI のモノラル化と同等)。
    /// 出力は常に 30ms(480 サンプル)ちょうどのブロックで `on_block` に渡る。
    pub fn push_interleaved(&mut self, data: &[f32]) {
        self.mono.clear();
        if self.channels == 1 {
            self.mono.extend_from_slice(data);
        } else {
            let pick = &self.pick;
            let inv = 1.0 / pick.len() as f32;
            self.mono.extend(
                data.chunks_exact(self.channels)
                    .map(|f| pick.iter().map(|&c| f[c]).sum::<f32>() * inv),
            );
        }
        if let Err(e) = self.resampler.process(&self.mono, &mut self.out) {
            eprintln!("[mic] resample failed: {e}");
            return;
        }
        while self.out.len() >= self.out_block {
            let blk: Vec<f32> = self.out.drain(..self.out_block).collect();
            (self.on_block)(blk);
        }
    }
}

struct Running {
    stop_tx: mpsc::Sender<()>,
    handle: JoinHandle<()>,
}

/// マイクストリーム。`start`/`stop` は `&self` で呼べ、いずれも直列化される(stop は冪等)。
#[derive(Default)]
pub struct MicStream {
    inner: Mutex<Option<Running>>,
}

impl MicStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// `device_id` は `list_input_devices` の `id`。None で既定入力。
    /// `channels` は使うチャンネル(0 始まり、`DeviceInfo::channels` の番号)。空なら既定
    /// (WASAPI は全チャンネルの平均、ASIO は 1ch 目)。
    /// 戻り値はデバイスの実サンプルレート。`on_block` は cpal のオーディオスレッドから呼ばれる。
    pub fn start(
        &self,
        device_id: Option<String>,
        channels: Vec<u16>,
        on_block: impl FnMut(Vec<f32>) + Send + 'static,
    ) -> Result<u32> {
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_some() {
            return Err(anyhow!("マイクは既に開始されています"));
        }
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<u32>>();
        let handle = std::thread::Builder::new()
            .name("mic-stream".into())
            .spawn(move || {
                // Stream はこのスレッドで生成し、このスレッドで drop する
                match open_stream(device_id.as_deref(), &channels, on_block) {
                    Ok((stream, rate)) => {
                        let _ = ready_tx.send(Ok(rate));
                        // stop 指示、または MicStream 側の Sender が落ちるまで待つ
                        let _ = stop_rx.recv();
                        drop(stream);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }
            })
            .context("マイクスレッドを起動できません")?;
        match ready_rx.recv() {
            Ok(Ok(rate)) => {
                *guard = Some(Running { stop_tx, handle });
                Ok(rate)
            }
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            Err(_) => {
                let _ = handle.join();
                Err(anyhow!("マイクスレッドが異常終了しました"))
            }
        }
    }

    /// 停止。戻り時点で Stream は drop 済み(デバイス解放済み)。冪等。
    pub fn stop(&self) {
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(r) = guard.take() {
            let _ = r.stop_tx.send(());
            if r.handle.join().is_err() {
                eprintln!("[mic] mic thread panicked");
            }
        }
    }

    pub fn is_running(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some()
    }
}

impl Drop for MicStream {
    fn drop(&mut self) {
        self.stop();
    }
}

/// 開いたストリームとデバイス。drop はフィールド順(ストリーム → デバイスの返却)。
struct Opened {
    _stream: cpal::Stream,
    _device: OpenedDevice,
}

fn open_stream(
    device_id: Option<&str>,
    channels: &[u16],
    on_block: impl FnMut(Vec<f32>) + Send + 'static,
) -> Result<(Opened, u32)> {
    let opened = open_input_device(device_id)?;
    let device = &opened.device;
    let name = opened.name();
    let supported = device
        .default_input_config()
        .context("入力設定を取得できません")?;
    let (config, pick) = opened.stream_config(&supported, channels, true)?;
    let rate = config.sample_rate;
    let channels = config.channels;
    let used: Vec<String> = pick.iter().map(|c| (c + 1).to_string()).collect();
    let pipe = BlockPipeline::with_pick(rate, channels, pick, on_block)?;

    let stream = match supported.sample_format() {
        SampleFormat::F32 => build::<f32>(device, &config, pipe),
        SampleFormat::I16 => build::<i16>(device, &config, pipe),
        SampleFormat::I24 => build::<I24>(device, &config, pipe),
        SampleFormat::I32 => build::<i32>(device, &config, pipe),
        SampleFormat::U16 => build::<u16>(device, &config, pipe),
        other => Err(anyhow!("未対応のサンプル形式です: {other:?}")),
    }?;
    stream.play().context("マイクを開始できません")?;
    eprintln!("[mic] started: {name} @ {rate}Hz x{channels} (use ch {})", used.join("+"));
    Ok((Opened { _stream: stream, _device: opened }, rate))
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut pipe: BlockPipeline,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let mut tmp: Vec<f32> = Vec::new();
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                tmp.clear();
                tmp.extend(
                    data.iter()
                        .map(|&s| <f32 as FromSample<T>>::from_sample_(s)),
                );
                pipe.push_interleaved(&tmp);
            },
            |e| eprintln!("[mic] stream error: {e}"),
            None,
        )
        .context("マイクストリームを開けません")
}
