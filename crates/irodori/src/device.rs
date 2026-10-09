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
/// GPU が使えないときは `Err`(初期化に失敗した場合は次の呼び出しで再試行する)。
#[cfg(feature = "_gpu")]
pub fn try_gpu_device() -> anyhow::Result<Device> {
    use std::sync::Mutex;

    use burn::tensor::DeviceKind;
    use burn::tensor::wgpu::MemoryConfiguration;

    static DEV: Mutex<Option<Device>> = Mutex::new(None);
    let mut slot = DEV.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(d) = slot.as_ref() {
        return Ok(d.clone());
    }
    let init = |kind: DeviceKind| {
        // wgpu の初期化失敗は panic で返ることがあるため、ここで受けて Err にする
        std::panic::catch_unwind(|| Device::wgpu_options().device_kind(kind).memory_config(MemoryConfiguration::ExclusivePages).init())
            .map_err(|_| anyhow::anyhow!("wgpu の初期化中に panic しました"))?
            .map_err(|e| anyhow::anyhow!("{e:?}"))
    };
    let dev = init(DeviceKind::DiscreteGpu(0))
        .or_else(|_| init(DeviceKind::DefaultDevice))
        .map_err(|e| anyhow::anyhow!("GPU を初期化できません: {e:#}"))?;
    *slot = Some(dev.clone());
    Ok(dev)
}

/// `try_gpu_device` の panic 版(テスト・example 用)。
#[cfg(feature = "_gpu")]
pub fn gpu_device() -> Device {
    try_gpu_device().expect("wgpu: GPU を初期化できません")
}
