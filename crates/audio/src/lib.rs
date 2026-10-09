//! sttts-audio: マイク入力・デバイス列挙・WAV ソース・Silero VAD。
//!
//! Python backend の `engines/mic.py` / `wav_source.py` / `vad_silero.py` の Rust 版。
//! すべて 16kHz モノラル f32 を共通の音声表現とする。

pub mod devices;
pub mod mic;
pub mod resample;
pub mod vad;
pub mod wav_source;

pub use devices::{DeviceInfo, list_input_devices, list_output_devices};
pub use mic::{BlockPipeline, MicStream};
pub use resample::{StreamResampler, resample_all};
pub use vad::{SileroVad, VadEvent};
pub use wav_source::{WavSource, WavSourceOptions, load_wav_16k};

/// VAD の 1 フレームのサンプル数(16kHz で 32ms)。
pub const FRAME: usize = 512;
/// 全体で使う目標サンプルレート。
pub const TARGET_RATE: u32 = 16000;
/// マイク / WAV ソースが on_block に渡すブロック長(秒)。
pub const BLOCK_SECONDS: f64 = 0.03;
