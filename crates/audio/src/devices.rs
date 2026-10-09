//! 入出力デバイスの列挙(デバイスは open しない)。
//!
//! `index` は cpal の `host.input_devices()` / `output_devices()` の列挙順位置。
//! `MicStream::start` の `device_index` も同じ体系。

use cpal::traits::{DeviceTrait, HostTrait};

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    pub index: i64,
    pub name: String,
    pub default_rate: u32,
    pub is_default: bool,
}

/// 入力デバイス一覧。列挙に失敗した場合は空リスト(致命傷にしない)。
pub fn list_input_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default_id = host.default_input_device().and_then(|d| d.id().ok());
    let Ok(devs) = host.input_devices() else {
        return Vec::new();
    };
    collect(devs, default_id, true)
}

/// 出力デバイス一覧。
pub fn list_output_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default_id = host.default_output_device().and_then(|d| d.id().ok());
    let Ok(devs) = host.output_devices() else {
        return Vec::new();
    };
    collect(devs, default_id, false)
}

fn collect(
    devs: impl Iterator<Item = cpal::Device>,
    default_id: Option<cpal::DeviceId>,
    input: bool,
) -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    // index は列挙位置を保つため、情報が取れないデバイスも番号だけ消費する
    for (i, d) in devs.enumerate() {
        let Ok(desc) = d.description() else { continue };
        let cfg = if input {
            d.default_input_config()
        } else {
            d.default_output_config()
        };
        let Ok(cfg) = cfg else { continue };
        let is_default = match (&default_id, d.id()) {
            (Some(a), Ok(b)) => *a == b,
            _ => false,
        };
        out.push(DeviceInfo {
            index: i as i64,
            name: desc.name().to_string(),
            default_rate: cfg.sample_rate(),
            is_default,
        });
    }
    out
}
