//! ストリーミング音声再生。バックエンドから届いたチャンクWAVを順次キューに入れて再生する。
//!
//! 出力デバイスは sttts-audio 経由で開く(WASAPI / ASIO)。ASIO はマイクと同じドライバを
//! 使うことがあり、その場合は同じデバイスインスタンスを共有しないと壊れるため(sttts-audio の
//! `devices` 参照)、rodio に直接デバイスを探させない。

use std::io::Cursor;
use std::num::NonZero;

use anyhow::{Result, anyhow};
use base64::Engine as _;
use rodio::cpal::traits::{DeviceTrait as _, StreamTrait as _};
use rodio::cpal::{FromSample, I24, SampleFormat, SizedSample};
use rodio::{Decoder, DeviceSinkBuilder, Player};
use sttts_audio::{DeviceInfo, OpenedDevice};

/// 再生可能な出力デバイスの一覧(既定ホスト = WASAPI と ASIO)。
pub fn list_output_devices() -> Vec<DeviceInfo> {
    sttts_audio::list_output_devices()
}

/// Player を空にして再生可能な状態に戻す。
pub fn clear_and_resume(player: &Player) {
    player.clear();
    player.play();
}

#[expect(dead_code, reason = "drop でストリームを閉じるために保持するだけ")]
enum OutStream {
    /// rodio が開いたストリーム(WASAPI)
    Rodio(rodio::MixerDeviceSink),
    /// 指定チャンネルへ書き込む自前のストリーム(ASIO)
    Routed(rodio::cpal::Stream),
}

pub struct AudioOut {
    // drop はフィールド順: Player → ストリーム → デバイスの返却(ASIO ドライバの解放)
    player: Player,
    // 再生デバイスを保持し続けるためにフィールドに置いておく必要がある
    _stream: OutStream,
    _device: OpenedDevice,
}

impl AudioOut {
    /// 出力デバイスを開く。`id` は `DeviceInfo::id`(None ならシステム既定)。
    /// `channels` は ASIO の出力チャンネル(0 始まり。1 つならモノラル、2 つならステレオ)。
    /// 空なら既定(ASIO は 1/2ch)。WASAPI では使わない。
    pub fn open(id: Option<&str>, channels: &[u16]) -> Result<Self> {
        let device = sttts_audio::open_output_device(id)?;
        if !device.is_asio() {
            let mut stream = DeviceSinkBuilder::from_device(device.device.clone())?.open_stream()?;
            // 終了時の "Dropping OutputStream, ..." という eprintln を抑制する(情報メッセージのため)
            stream.log_on_drop(false);
            let player = Player::connect_new(stream.mixer());
            return Ok(Self { player, _stream: OutStream::Rodio(stream), _device: device });
        }
        // ASIO: rodio はストリームの全チャンネルへ同じ音を出すので、ミキサーの出力を
        // 選んだチャンネルにだけ書き込むストリームを自前で張る
        let supported = device.device.default_output_config()?;
        let (config, pick) = device.stream_config(&supported, channels, false)?;
        let mix_channels = NonZero::new(pick.len() as u16).ok_or_else(|| anyhow!("出力チャンネルが空です"))?;
        let rate = NonZero::new(config.sample_rate).ok_or_else(|| anyhow!("サンプルレートが 0 です"))?;
        let (mixer, source) = rodio::mixer::mixer(mix_channels, rate);
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build_routed::<f32>(&device, &config, pick, source),
            SampleFormat::I16 => build_routed::<i16>(&device, &config, pick, source),
            SampleFormat::I24 => build_routed::<I24>(&device, &config, pick, source),
            SampleFormat::I32 => build_routed::<i32>(&device, &config, pick, source),
            other => Err(anyhow!("未対応のサンプル形式です: {other:?}")),
        }?;
        stream.play()?;
        let player = Player::connect_new(&mixer);
        Ok(Self { player, _stream: OutStream::Routed(stream), _device: device })
    }

    /// base64 エンコードされた WAV をデコードしてキューに積む(完了したチャンクから順に再生)。
    pub fn enqueue_wav_base64(&self, wav_base64: &str) -> Result<()> {
        let wav = base64::engine::general_purpose::STANDARD.decode(wav_base64)?;
        self.enqueue_wav_bytes(wav)
    }

    /// WAV バイト列をキューに積む。
    pub fn enqueue_wav_bytes(&self, wav: Vec<u8>) -> Result<()> {
        let source = Decoder::new(Cursor::new(wav))?;
        self.player.append(source);
        Ok(())
    }

    /// 再生中・未再生のチャンクをすべて破棄する(キャンセル)。
    ///
    /// rodio 0.22 の `Player::clear()` はキューを空にした上で一時停止状態にするため、
    /// そのままだと以降に積んだチャンクが一切鳴らなくなる。clear 後に必ず play() で再開する。
    pub fn clear(&self) {
        clear_and_resume(&self.player);
    }

    /// 未再生チャンク数(現在再生中のものを含む)。
    pub fn pending_chunks(&self) -> usize {
        self.player.len()
    }
}

/// ミキサーの出力(`pick.len()` チャンネルのインターリーブ)を、デバイスの `pick` 位置へ書く。
/// 他のチャンネルは無音。
fn build_routed<T>(
    device: &OpenedDevice,
    config: &rodio::cpal::StreamConfig,
    pick: Vec<usize>,
    mut source: rodio::mixer::MixerSource,
) -> Result<rodio::cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let n = config.channels as usize;
    let stream = device.device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            for frame in data.chunks_exact_mut(n) {
                frame.fill(T::EQUILIBRIUM);
                for &c in &pick {
                    frame[c] = T::from_sample_(source.next().unwrap_or(0.0));
                }
            }
        },
        |e| eprintln!("[audio] output stream error: {e}"),
        None,
    )?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::source::{SineWave, Source};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// 出力デバイス無しで Player を駆動する(オーディオスレッドの代わりにサンプルを引き続ける)。
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
    fn clear_does_not_leave_player_paused() {
        let (player, out) = Player::new();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = drive(out, stop.clone());

        player.append(SineWave::new(440.0).take_duration(Duration::from_secs(5)));
        assert_eq!(player.len(), 1);
        clear_and_resume(&player);
        assert!(!player.is_paused(), "clear 後に Player が一時停止のまま");
        assert_eq!(player.len(), 0);

        // キャンセル後に届いたチャンクが再生される(=消費されて len が減る)こと
        player.append(SineWave::new(440.0).take_duration(Duration::from_millis(20)));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while player.len() > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(player.len(), 0, "キャンセル後のチャンクが再生されない");

        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
    }

    #[test]
    fn plain_clear_pauses_player_in_rodio() {
        // 回帰検知用: rodio の clear() が pause する仕様であることを確認(仕様変更に気付けるように)
        let (player, out) = Player::new();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = drive(out, stop.clone());
        player.append(SineWave::new(440.0).take_duration(Duration::from_secs(5)));
        player.clear();
        assert!(player.is_paused());
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
    }
}
