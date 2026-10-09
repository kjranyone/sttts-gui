//! 本番の `Platform`: Irodori-TTS(GPU)、各 ASR、cpal のマイク、Silero VAD、デバイス列挙。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;
use sttts_protocol::AudioDeviceInfo;

use crate::app::{Platform, TtsProgress};
use crate::asr::{AsrEngine, Progress, create_asr};
use crate::config::{get, get_f64, get_i64};
use crate::session::{AudioSource, OnBlock, Vad, VadEvent};
use crate::tts::{IrodoriTts, TtsEngine};

pub struct RealPlatform;

impl Platform for RealPlatform {
    fn create_tts(&self, cfg: &Value, progress: TtsProgress) -> Result<Arc<dyn TtsEngine>> {
        let model = get(cfg, "tts", "model").as_str().unwrap_or("v4.1-small-mf");
        let steps = get(cfg, "tts", "num_steps").as_u64().map(|n| n as usize);
        Ok(Arc::new(IrodoriTts::load(model, steps, progress)?))
    }

    fn create_asr(&self, cfg: &Value, progress: Progress) -> Result<Arc<dyn AsrEngine>> {
        create_asr(cfg, progress)
    }

    fn open_source(&self, cfg: &Value, input_wavs: &[PathBuf], on_block: OnBlock, on_eof: Option<Box<dyn FnOnce() + Send>>) -> Result<Box<dyn AudioSource>> {
        if !input_wavs.is_empty() {
            let paths = input_wavs.iter().map(|p| p.to_string_lossy().into_owned()).collect();
            let src = sttts_audio::WavSource::new(paths, sttts_audio::WavSourceOptions::default(), move |b| on_block(b), on_eof);
            return Ok(Box::new(WavSrc(src)));
        }
        let device = get(cfg, "audio", "input_device_index").as_i64();
        Ok(Box::new(MicSrc { mic: sttts_audio::MicStream::new(), device, on_block }))
    }

    fn create_vad(&self, cfg: &Value) -> Result<Box<dyn Vad>> {
        let threshold = get_f64(cfg, "asr", "vad_threshold", 0.5) as f32;
        let min_silence = get_i64(cfg, "asr", "vad_min_silence_ms", 280).max(0) as u32;
        Ok(Box::new(SileroAdapter(sttts_audio::SileroVad::new(threshold, min_silence)?)))
    }

    fn list_devices(&self) -> (Vec<AudioDeviceInfo>, Vec<AudioDeviceInfo>) {
        let conv = |v: Vec<sttts_audio::DeviceInfo>| {
            v.into_iter().map(|d| AudioDeviceInfo { index: d.index, name: d.name, default_rate: Some(d.default_rate), is_default: d.is_default }).collect()
        };
        (conv(sttts_audio::list_input_devices()), conv(sttts_audio::list_output_devices()))
    }
}

struct MicSrc {
    mic: sttts_audio::MicStream,
    device: Option<i64>,
    on_block: OnBlock,
}

impl AudioSource for MicSrc {
    fn start(&mut self) -> Result<()> {
        let cb = Arc::clone(&self.on_block);
        self.mic.start(self.device, move |b| cb(b))?;
        Ok(())
    }
    fn stop(&mut self) {
        self.mic.stop();
    }
}

struct WavSrc(sttts_audio::WavSource);

impl AudioSource for WavSrc {
    fn start(&mut self) -> Result<()> {
        self.0.start()?;
        Ok(())
    }
    fn stop(&mut self) {
        self.0.stop();
    }
}

struct SileroAdapter(sttts_audio::SileroVad);

impl Vad for SileroAdapter {
    fn process(&mut self, frame: &[f32]) -> Option<VadEvent> {
        self.0.process(frame).map(|e| match e {
            sttts_audio::VadEvent::Start(n) => VadEvent::Start(n),
            sttts_audio::VadEvent::End(n) => VadEvent::End(n),
        })
    }
    fn reset(&mut self) {
        self.0.reset();
    }
}
