//! 入出力デバイスの列挙と、ID によるデバイス取得。
//!
//! 対象ホストは既定ホスト(Windows では WASAPI)と、Windows では ASIO。
//! デバイスは cpal の `DeviceId` の文字列表現(`wasapi:{...}` / `asio:{ドライバ名}`)で指す。
//! 列挙位置は ASIO ドライバの使用状況で変わるため、位置(index)では指さない。
//!
//! ## ASIO の制約
//! - ASIO はプロセス全体で同時に 1 ドライバしかロードできない。
//! - 同じドライバの入力と出力は、同じ `cpal::Device` インスタンスから作らないと
//!   バッファ(ASIOCreateBuffers)を互いに作り直して壊し合う。
//!
//! - cpal の ASIO ストリームは常に先頭から N チャンネルを確保する。任意のチャンネル(例: 入力 3)を
//!   使うには N = 最大チャンネル + 1 で開き、コールバック側で該当チャンネルだけを取り出す
//!   (`OpenedDevice::stream_config` が返す `pick`)。
//!
//! そのため ASIO デバイスは本モジュールの共有レジストリから貸し出す(`OpenedDevice`)。
//! 入力(マイク)と出力(GUI の再生)が同じドライバを使う場合は同じインスタンスを共有し、
//! 最後の貸し出しが返った時点でドライバを解放する。

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait};

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    /// `open_*_device` に渡す ID(cpal の `DeviceId` の文字列表現)。
    pub id: String,
    /// ホスト(ドライバ方式)名。`WASAPI` / `ASIO`。
    pub host: String,
    /// 表示名(ASIO はドライバ名)。
    pub name: String,
    pub default_rate: u32,
    pub is_default: bool,
    /// ASIO のチャンネル名(この方向の全チャンネル、0 始まりの番号順)。WASAPI は空
    /// (チャンネルを選ばず全チャンネルを使う)。
    pub channels: Vec<String>,
}

/// 入力デバイス一覧(既定ホスト → ASIO の順)。列挙に失敗したホストは飛ばす(致命傷にしない)。
///
/// ASIO の列挙は各ドライバを順にロードして問い合わせる。別の ASIO ドライバを使用中の間は、
/// 使用中のドライバ以外は列挙されない(ASIO は同時に 1 ドライバのみ)。
pub fn list_input_devices() -> Vec<DeviceInfo> {
    list(Direction::Input)
}

/// 出力デバイス一覧。
pub fn list_output_devices() -> Vec<DeviceInfo> {
    list(Direction::Output)
}

#[derive(Clone, Copy, PartialEq)]
enum Direction {
    Input,
    Output,
}

fn hosts() -> Vec<cpal::Host> {
    #[allow(unused_mut)]
    let mut hosts = vec![cpal::default_host()];
    #[cfg(windows)]
    match cpal::host_from_id(cpal::HostId::Asio) {
        Ok(h) => hosts.push(h),
        Err(e) => eprintln!("[audio] ASIO host unavailable: {e}"),
    }
    hosts
}

fn list(dir: Direction) -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    for host in hosts() {
        let default_id = match dir {
            Direction::Input => host.default_input_device(),
            Direction::Output => host.default_output_device(),
        }
        .and_then(|d| d.id().ok());
        // ASIO には「既定」の概念が無い(cpal は先頭を返す)ので既定ホストのものだけ採る
        let default_id = default_id.filter(|_| !is_asio_host(host.id()));
        let devs = match dir {
            Direction::Input => host.input_devices(),
            Direction::Output => host.output_devices(),
        };
        let Ok(devs) = devs else { continue };
        for d in devs {
            let Ok(id) = d.id() else { continue };
            let Ok(desc) = d.description() else { continue };
            let cfg = match dir {
                Direction::Input => d.default_input_config(),
                Direction::Output => d.default_output_config(),
            };
            let Ok(cfg) = cfg else { continue };
            // ASIO のチャンネル情報は、このデバイス(= ドライバのロード)が生きている間に問い合わせる
            let channels = if is_asio_host(id.0) {
                asio_channel_names(dir == Direction::Input, cfg.channels())
            } else {
                Vec::new()
            };
            out.push(DeviceInfo {
                host: host_label(id.0),
                name: desc.name().to_string(),
                is_default: default_id.as_ref() == Some(&id),
                id: id.to_string(),
                default_rate: cfg.sample_rate(),
                channels,
            });
        }
    }
    out
}

fn host_label(host: cpal::HostId) -> String {
    host.name().to_uppercase()
}

/// 現在ロード中の ASIO ドライバのチャンネル名。名前が取れないチャンネルは `Ch N` とする。
#[cfg(windows)]
fn asio_channel_names(input: bool, count: u16) -> Vec<String> {
    use asio_sys::bindings::asio_import as ai;
    (0..count)
        .map(|ch| {
            let mut info = ai::ASIOChannelInfo {
                channel: ch.into(),
                isInput: input.into(),
                isActive: 0,
                channelGroup: 0,
                type_: 0,
                name: [0; 32],
            };
            // SAFETY: ドライバは呼び出し元のデバイスが保持している間ロード済み。info は有効な領域
            let ok = unsafe { ai::ASIOGetChannelInfo(&mut info) } == 0;
            let name = if ok { decode_ansi(&info.name) } else { String::new() };
            if name.trim().is_empty() { format!("Ch {}", ch + 1) } else { name.trim().to_string() }
        })
        .collect()
}

#[cfg(not(windows))]
fn asio_channel_names(_input: bool, count: u16) -> Vec<String> {
    (0..count).map(|ch| format!("Ch {}", ch + 1)).collect()
}

/// ASIO SDK の文字列(NUL 終端の ANSI = システムのコードページ)を UTF-8 へ。
/// UTF-8 として読むと日本語環境のチャンネル名(CP932)が化ける。
#[cfg(windows)]
fn decode_ansi(raw: &[std::os::raw::c_char]) -> String {
    use windows::Win32::Globalization::{CP_ACP, MULTI_BYTE_TO_WIDE_CHAR_FLAGS, MultiByteToWideChar};
    let bytes: Vec<u8> = raw.iter().map(|&c| c as u8).take_while(|&b| b != 0).collect();
    if bytes.is_empty() {
        return String::new();
    }
    let flags = MULTI_BYTE_TO_WIDE_CHAR_FLAGS(0);
    // SAFETY: 入出力スライスは有効。長さは関数が返す必要量で確保する
    let n = unsafe { MultiByteToWideChar(CP_ACP, flags, &bytes, None) };
    if n <= 0 {
        return String::from_utf8_lossy(&bytes).into_owned();
    }
    let mut wide = vec![0u16; n as usize];
    let n = unsafe { MultiByteToWideChar(CP_ACP, flags, &bytes, Some(&mut wide)) };
    String::from_utf16_lossy(&wide[..n.max(0) as usize])
}

fn is_asio_host(_host: cpal::HostId) -> bool {
    #[cfg(windows)]
    {
        _host == cpal::HostId::Asio
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// 開いたデバイス。ASIO の場合は共有レジストリからの貸し出しで、drop で返却する。
///
/// これから作ったストリームは、この値より先に drop すること(ドライバ解放の順序)。
pub struct OpenedDevice {
    pub device: cpal::Device,
    asio_driver: Option<String>,
}

impl OpenedDevice {
    /// ASIO デバイスか。ASIO は既定設定(全チャンネル)のまま開かないこと(`stream_config` を使う)。
    pub fn is_asio(&self) -> bool {
        self.asio_driver.is_some()
    }

    /// ストリーム設定と、使うチャンネル(ストリーム内の 0 始まりの位置)。
    ///
    /// `channels` は使いたいチャンネル(0 始まり)。空なら既定:
    /// WASAPI は全チャンネル、ASIO は入力が 1ch 目・出力が 1/2ch 目(1ch しか無ければ 1ch 目)。
    /// ASIO は先頭から「最大チャンネル + 1」本を開き、バッファはドライバ設定
    /// (ASIO コントロールパネルの値)に従う。範囲外のチャンネルはエラー。
    pub fn stream_config(
        &self,
        supported: &cpal::SupportedStreamConfig,
        channels: &[u16],
        input: bool,
    ) -> Result<(cpal::StreamConfig, Vec<usize>)> {
        let mut config = supported.config();
        let total = config.channels;
        let pick: Vec<u16> = if !channels.is_empty() {
            channels.to_vec()
        } else if !self.is_asio() {
            (0..total).collect()
        } else if input || total < 2 {
            vec![0]
        } else {
            vec![0, 1]
        };
        if let Some(&bad) = pick.iter().find(|&&c| c >= total) {
            return Err(anyhow!(
                "チャンネル {} はありません(このデバイスは {total} チャンネル)",
                bad + 1
            ));
        }
        if self.is_asio() {
            config.channels = pick.iter().max().map_or(1, |&m| m + 1);
            config.buffer_size = cpal::BufferSize::Default;
        }
        Ok((config, pick.into_iter().map(usize::from).collect()))
    }

    pub fn name(&self) -> String {
        let name = self
            .device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_default();
        if self.is_asio() { format!("ASIO: {name}") } else { name }
    }
}

impl Drop for OpenedDevice {
    fn drop(&mut self) {
        if self.asio_driver.is_some() {
            let mut g = ASIO.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(s) = g.as_mut() {
                s.users = s.users.saturating_sub(1);
                if s.users == 0 {
                    // 共有インスタンスを落とす(他に参照が無ければここでドライバが解放される)
                    *g = None;
                }
            }
        }
    }
}

/// `id` は `DeviceInfo::id`。None で既定ホストの既定入力。
pub fn open_input_device(id: Option<&str>) -> Result<OpenedDevice> {
    open(id, Direction::Input)
}

/// `id` は `DeviceInfo::id`。None で既定ホストの既定出力。
pub fn open_output_device(id: Option<&str>) -> Result<OpenedDevice> {
    open(id, Direction::Output)
}

fn open(id: Option<&str>, dir: Direction) -> Result<OpenedDevice> {
    let kind = match dir {
        Direction::Input => "入力",
        Direction::Output => "出力",
    };
    let Some(id) = id else {
        let host = cpal::default_host();
        let device = match dir {
            Direction::Input => host.default_input_device(),
            Direction::Output => host.default_output_device(),
        }
        .ok_or_else(|| anyhow!("既定の{kind}デバイスがありません"))?;
        return Ok(OpenedDevice { device, asio_driver: None });
    };
    let parsed: cpal::DeviceId = id
        .parse()
        .map_err(|e| anyhow!("{kind}デバイス ID が不正です: {id} ({e})"))?;
    if is_asio_host(parsed.0) {
        return open_asio(&parsed.1, kind);
    }
    let host = cpal::host_from_id(parsed.0).with_context(|| format!("{} ホストを使えません", parsed.0))?;
    let device = host
        .device_by_id(&parsed)
        .ok_or_else(|| anyhow!("{kind}デバイスが見つかりません: {id}"))?;
    Ok(OpenedDevice { device, asio_driver: None })
}

struct AsioShared {
    driver: String,
    device: cpal::Device,
    users: usize,
}

/// 使用中の ASIO デバイス(プロセスで高々 1 つ)。
static ASIO: Mutex<Option<AsioShared>> = Mutex::new(None);

fn open_asio(driver: &str, kind: &str) -> Result<OpenedDevice> {
    let mut g = ASIO.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(s) = g.as_mut() {
        if s.driver != driver {
            return Err(anyhow!(
                "ASIO ドライバ「{}」を使用中のため「{driver}」を開けません(ASIO は同時に 1 ドライバのみ)",
                s.driver
            ));
        }
        s.users += 1;
        return Ok(OpenedDevice { device: s.device.clone(), asio_driver: Some(driver.to_string()) });
    }
    #[cfg(windows)]
    {
        let host = cpal::host_from_id(cpal::HostId::Asio).context("ASIO ホストを使えません")?;
        let device = host
            .devices()
            .context("ASIO ドライバを列挙できません")?
            .find(|d| d.id().is_ok_and(|i| i.1 == driver))
            .ok_or_else(|| anyhow!("ASIO {kind}デバイス「{driver}」をロードできません(未接続か、他のアプリが使用中)"))?;
        *g = Some(AsioShared { driver: driver.to_string(), device: device.clone(), users: 1 });
        Ok(OpenedDevice { device, asio_driver: Some(driver.to_string()) })
    }
    #[cfg(not(windows))]
    {
        Err(anyhow!("ASIO {kind}デバイスはこの環境では使えません: {driver}"))
    }
}
