//! Irodori-TTS の純 Rust 推論。burn の上に実装し、CPU(flex)は参照・テスト用、
//! 実運用は wgpu(Vulkan)で GPU を使う。PyTorch / Python には依存しない。

pub use burn::tensor::{Device, Tensor};
