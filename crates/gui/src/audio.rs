//! ストリーミング音声再生。バックエンドから届いたチャンクWAVを順次キューに入れて再生する。

use std::io::Cursor;

use anyhow::{anyhow, Result};
use base64::Engine as _;
use rodio::cpal::traits::HostTrait as _;
use rodio::{Decoder, DeviceTrait, OutputStreamBuilder, Sink};

/// 再生可能な出力デバイス名の一覧(cpal の既定ホスト = WASAPI)。
pub fn list_output_devices() -> Vec<String> {
    let host = rodio::cpal::default_host();
    let mut names = Vec::new();
    if let Ok(devices) = host.devices() {
        for device in devices {
            if device.default_output_config().is_err() {
                continue;
            }
            if let Ok(name) = device.name() {
                names.push(name);
            }
        }
    }
    names
}

/// Sink を空にして再生可能な状態に戻す。
pub fn clear_and_resume(sink: &Sink) {
    sink.clear();
    sink.play();
}

pub struct AudioOut {
    // OutputStream は再生デバイスを保持し続けるためにフィールドに置いておく必要がある
    _stream: rodio::OutputStream,
    sink: Sink,
}

impl AudioOut {
    /// 出力デバイスを開く。`preferred` はデバイス名(None ならシステム既定)。
    pub fn open(preferred: Option<&str>) -> Result<Self> {
        let builder = match preferred {
            Some(name) => {
                let host = rodio::cpal::default_host();
                let found = host
                    .devices()?
                    .filter(|d| d.default_output_config().is_ok())
                    .find(|d| d.name().map(|n| n.eq_ignore_ascii_case(name)).unwrap_or(false));
                match found {
                    Some(device) => OutputStreamBuilder::from_device(device)?,
                    None => {
                        return Err(anyhow!("出力デバイス「{name}」が見つかりません"));
                    }
                }
            }
            None => OutputStreamBuilder::from_default_device()?,
        };
        let mut stream = builder.open_stream()?;
        // 終了時の "Dropping OutputStream, ..." という eprintln を抑制する(情報メッセージのため)
        stream.log_on_drop(false);
        let sink = Sink::connect_new(stream.mixer());
        Ok(Self {
            _stream: stream,
            sink,
        })
    }

    /// base64 エンコードされた WAV をデコードしてキューに積む(完了したチャンクから順に再生)。
    pub fn enqueue_wav_base64(&self, wav_base64: &str) -> Result<()> {
        let wav = base64::engine::general_purpose::STANDARD.decode(wav_base64)?;
        self.enqueue_wav_bytes(wav)
    }

    /// WAV バイト列をキューに積む。
    pub fn enqueue_wav_bytes(&self, wav: Vec<u8>) -> Result<()> {
        let source = Decoder::new(Cursor::new(wav))?;
        self.sink.append(source);
        Ok(())
    }

    /// 再生中・未再生のチャンクをすべて破棄する(キャンセル)。
    ///
    /// rodio 0.21 の `Sink::clear()` はキューを空にした上で Sink を一時停止状態にするため、
    /// そのままだと以降に積んだチャンクが一切鳴らなくなる。clear 後に必ず play() で再開する。
    pub fn clear(&self) {
        clear_and_resume(&self.sink);
    }

    /// 未再生チャンク数(現在再生中のものを含む)。
    pub fn pending_chunks(&self) -> usize {
        self.sink.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::source::{SineWave, Source};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// 出力デバイス無しで Sink を駆動する(オーディオスレッドの代わりにサンプルを引き続ける)。
    fn drive(mut out: rodio::queue::SourcesQueueOutput, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                for _ in 0..512 {
                    let _ = out.next();
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        })
    }

    #[test]
    fn clear_does_not_leave_sink_paused() {
        let (sink, out) = Sink::new();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = drive(out, stop.clone());

        sink.append(SineWave::new(440.0).take_duration(Duration::from_secs(5)));
        assert_eq!(sink.len(), 1);
        clear_and_resume(&sink);
        assert!(!sink.is_paused(), "clear 後に Sink が一時停止のまま");
        assert_eq!(sink.len(), 0);

        // キャンセル後に届いたチャンクが再生される(=消費されて len が減る)こと
        sink.append(SineWave::new(440.0).take_duration(Duration::from_millis(20)));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while sink.len() > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(sink.len(), 0, "キャンセル後のチャンクが再生されない");

        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
    }

    #[test]
    fn plain_clear_pauses_sink_in_rodio() {
        // 回帰検知用: rodio の clear() が pause する仕様であることを確認(仕様変更に気付けるように)
        let (sink, out) = Sink::new();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = drive(out, stop.clone());
        sink.append(SineWave::new(440.0).take_duration(Duration::from_secs(5)));
        sink.clear();
        assert!(sink.is_paused());
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
    }
}
