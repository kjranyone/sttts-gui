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

    /// 未再生分を破棄する(キャンセル)。再生中の1チャンクは中断できない前提。
    pub fn clear(&self) {
        self.sink.clear();
    }

    /// 未再生チャンク数(現在再生中のものを含む)。
    pub fn pending_chunks(&self) -> usize {
        self.sink.len()
    }
}
