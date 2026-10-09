//! sttts-gui のバックエンド(Rust 版): マイク → VAD → ASR → チャンク分割 → Irodori-TTS を GUI と同じプロセスで動かす。
//!
//! 以前の Python バックエンド(stdio NDJSON)の置き換え。メッセージの型は `sttts-protocol` のまま、
//! 伝送路だけがプロセス内のチャネルになる。

pub mod app;
pub mod asr;
pub mod chunker;
pub mod config;
pub mod engines;
pub mod mock;
pub mod performance;
pub mod real;
pub mod session;
pub mod sink;
pub mod tts;
pub mod util;

pub use app::{Backend, BackendOptions, Platform};
pub use real::RealPlatform;
pub use sink::Sink;
