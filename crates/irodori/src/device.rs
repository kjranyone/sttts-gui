//! 実行デバイスの作成。
//!
//! GPU は wgpu(Vulkan)。メモリ管理は `ExclusivePages`(1 確保 = 1 ページ)にする。既定の適応型
//! (大きい確保に合わせてページを作り直し、中身を移し替える方式)は、長い発話を 1 回処理した後に
//! 以降のすべての演算が約 10 倍遅くなる状態に入ることがあった(Arc B570 で再現。メモリ使用量は
//! 変わらないまま遅くなる)。`ExclusivePages` ではこの劣化が起きない。

use burn::tensor::Device;

/// 純 Rust の CPU バックエンド(参照・テスト用)
pub fn cpu_device() -> Device {
    Device::flex()
}

/// 独立 GPU(無ければ最初の GPU)を、プロセスで 1 回だけ初期化して返す(2 回目以降は同じデバイスの複製)。
#[cfg(feature = "_gpu")]
pub fn gpu_device() -> Device {
    use std::sync::OnceLock;

    use burn::tensor::DeviceKind;
    use burn::tensor::wgpu::MemoryConfiguration;

    static DEV: OnceLock<Device> = OnceLock::new();
    DEV.get_or_init(|| {
        let init = |kind: DeviceKind| {
            Device::wgpu_options().device_kind(kind).memory_config(MemoryConfiguration::ExclusivePages).init()
        };
        init(DeviceKind::DiscreteGpu(0))
            .or_else(|_| init(DeviceKind::DefaultDevice))
            .expect("wgpu: GPU を初期化できません")
    })
    .clone()
}
