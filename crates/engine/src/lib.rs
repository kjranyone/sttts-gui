//! sttts-gui のバックエンド(Rust 版): マイク → VAD → ASR → チャンク分割 → Irodori-TTS を GUI と同じプロセスで動かす。
//!
//! 以前の Python バックエンド(stdio NDJSON)の置き換え。メッセージの型は `sttts-protocol` のまま、
//! 伝送路だけがプロセス内のチャネルになる。

pub mod app;
pub mod asr;
pub mod chunker;
pub mod config;
#[cfg(feature = "live")]
pub mod engines;
pub mod mock;
pub mod performance;
pub mod presets;
#[cfg(feature = "live")]
pub mod real;
pub mod root;
pub mod say;
pub mod session;
pub mod sink;
pub mod tts;
pub mod util;

pub use app::{Backend, BackendOptions, Platform};
#[cfg(feature = "live")]
pub use real::RealPlatform;
pub use sink::Sink;
