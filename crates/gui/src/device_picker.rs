//! 入出力デバイスの 2 段選択(ドライバ → 候補)。
//!
//! 1 段目はドライバ: `WASAPI` と、ASIO ドライバごとの `ASIO: {ドライバ名}`。
//! 2 段目の候補は、WASAPI なら「既定」+ 各デバイス、ASIO ならそのドライバのチャンネル
//! (入力は 1ch ずつと隣り合う 2ch の組、出力は 2ch の組と 1ch ずつ)。
//!
//! 選択の保存は (ドライバ名, 候補ラベル)。ラベルはドライバ内で一意。

use sttts_i18n::tr;
use sttts_protocol::AudioDeviceInfo;

pub const WASAPI_DRIVER: &str = "WASAPI";
const ASIO_PREFIX: &str = "ASIO: ";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dir {
    Input,
    Output,
}

/// 2 段目の候補 1 つ。`device_id` が None ならシステム既定。
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub label: String,
    pub device_id: Option<String>,
    /// 使うチャンネル(0 始まり)。WASAPI は空(全チャンネル)
    pub channels: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct DevicePicker {
    dir: Dir,
    devices: Vec<AudioDeviceInfo>,
}

impl DevicePicker {
    pub fn new(dir: Dir, devices: Vec<AudioDeviceInfo>) -> Self {
        Self { dir, devices }
    }

    fn default_label(&self) -> &'static str {
        match self.dir {
            Dir::Input => tr!("Default input device", "既定の入力デバイス", "默认输入设备"),
            Dir::Output => tr!("Default output device", "既定の出力デバイス", "默认输出设备"),
        }
    }

    /// 1 段目の項目。WASAPI(システム既定を含む)は常に先頭。
    pub fn drivers(&self) -> Vec<String> {
        let mut out = vec![WASAPI_DRIVER.to_string()];
        out.extend(
            self.devices
                .iter()
                .filter(|d| is_asio(d))
                .map(|d| format!("{ASIO_PREFIX}{}", d.name)),
        );
        out
    }

    /// 2 段目の候補。未知のドライバは空。
    pub fn choices(&self, driver: &str) -> Vec<Choice> {
        if driver == WASAPI_DRIVER {
            let mut out = vec![Choice { label: self.default_label().into(), device_id: None, channels: Vec::new() }];
            for d in self.devices.iter().filter(|d| !is_asio(d)) {
                // 同名デバイスはラベルを番号で区別する(ラベルで保存・復元するため)
                let mut label = d.name.clone();
                let mut n = 2;
                while out.iter().any(|c| c.label == label) {
                    label = format!("{} ({n})", d.name);
                    n += 1;
                }
                out.push(Choice { label, device_id: Some(d.id.clone()), channels: Vec::new() });
            }
            return out;
        }
        let Some(name) = driver.strip_prefix(ASIO_PREFIX) else { return Vec::new() };
        let Some(d) = self.devices.iter().find(|d| is_asio(d) && d.name == name) else { return Vec::new() };
        let singles = (0..d.channels.len() as u16).map(|c| vec![c]);
        let pairs = (0..d.channels.len() as u16 / 2).map(|p| vec![2 * p, 2 * p + 1]);
        let sets: Vec<Vec<u16>> = match self.dir {
            // マイクは 1 本ずつ使うのが基本。ステレオ入力は 2ch の平均
            Dir::Input => singles.chain(pairs).collect(),
            // 再生はステレオの組が基本。1ch ずつはモノラル出力
            Dir::Output => pairs.chain(singles).collect(),
        };
        sets.into_iter()
            .map(|chs| Choice {
                label: channel_label(&d.channels, &chs),
                device_id: Some(d.id.clone()),
                channels: chs,
            })
            .collect()
    }

    /// 保存済みの (ドライバ, ラベル) を探す。ドライバ未保存は WASAPI 扱い。
    /// 見つからなければ None(呼び出し側で既定へ)。
    pub fn find(&self, driver: Option<&str>, label: Option<&str>) -> Option<(String, Choice)> {
        let driver = driver.unwrap_or(WASAPI_DRIVER);
        let choices = self.choices(driver);
        let choice = match label {
            Some(l) => choices.into_iter().find(|c| c.label == l)?,
            None => choices.into_iter().next()?,
        };
        Some((driver.to_string(), choice))
    }

    /// システム既定(WASAPI の「既定」)。
    pub fn system_default(&self) -> (String, Choice) {
        let choice = self.choices(WASAPI_DRIVER).remove(0);
        (WASAPI_DRIVER.to_string(), choice)
    }
}

fn is_asio(d: &AudioDeviceInfo) -> bool {
    d.host.eq_ignore_ascii_case("ASIO")
}

/// `1: Mic 1` / `1+2: Mic 1 / Mic 2`(番号は 1 始まり)
fn channel_label(names: &[String], chs: &[u16]) -> String {
    let nums: Vec<String> = chs.iter().map(|c| (c + 1).to_string()).collect();
    let names: Vec<&str> = chs.iter().map(|&c| names[c as usize].as_str()).collect();
    format!("{}: {}", nums.join("+"), names.join(" / "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(id: &str, host: &str, name: &str, channels: &[&str]) -> AudioDeviceInfo {
        AudioDeviceInfo {
            id: id.into(),
            host: host.into(),
            name: name.into(),
            default_rate: Some(48000),
            is_default: false,
            channels: channels.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn sample() -> Vec<AudioDeviceInfo> {
        vec![
            dev("wasapi:{a}", "WASAPI", "マイク (USB)", &[]),
            dev("wasapi:{b}", "WASAPI", "マイク (USB)", &[]),
            dev("asio:MOTU MicroBook ASIO", "ASIO", "MOTU MicroBook ASIO", &["Mic 1", "Mic 2", "Guitar"]),
            dev("asio:Realtek ASIO", "ASIO", "Realtek ASIO", &["L", "R"]),
        ]
    }

    #[test]
    fn drivers_list_wasapi_first_then_each_asio_driver() {
        let p = DevicePicker::new(Dir::Input, sample());
        assert_eq!(p.drivers(), ["WASAPI", "ASIO: MOTU MicroBook ASIO", "ASIO: Realtek ASIO"]);
        // ASIO が無くても WASAPI(システム既定)は選べる
        assert_eq!(DevicePicker::new(Dir::Input, Vec::new()).drivers(), ["WASAPI"]);
    }

    #[test]
    fn wasapi_choices_are_default_plus_devices_with_unique_labels() {
        let p = DevicePicker::new(Dir::Input, sample());
        let c = p.choices("WASAPI");
        let labels: Vec<_> = c.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, ["既定の入力デバイス", "マイク (USB)", "マイク (USB) (2)"]);
        assert_eq!(c[0].device_id, None);
        assert_eq!(c[2].device_id.as_deref(), Some("wasapi:{b}"));
        assert!(c.iter().all(|c| c.channels.is_empty()));
    }

    #[test]
    fn asio_input_choices_are_each_channel_then_pairs() {
        let p = DevicePicker::new(Dir::Input, sample());
        let c = p.choices("ASIO: MOTU MicroBook ASIO");
        let got: Vec<_> = c.iter().map(|c| (c.label.as_str(), c.channels.clone())).collect();
        assert_eq!(
            got,
            [
                ("1: Mic 1", vec![0]),
                ("2: Mic 2", vec![1]),
                ("3: Guitar", vec![2]),
                ("1+2: Mic 1 / Mic 2", vec![0, 1]),
            ]
        );
        assert!(c.iter().all(|c| c.device_id.as_deref() == Some("asio:MOTU MicroBook ASIO")));
    }

    #[test]
    fn asio_output_choices_put_stereo_pairs_first() {
        let p = DevicePicker::new(Dir::Output, sample());
        let labels: Vec<_> = p.choices("ASIO: Realtek ASIO").into_iter().map(|c| c.label).collect();
        assert_eq!(labels, ["1+2: L / R", "1: L", "2: R"]);
    }

    #[test]
    fn find_restores_saved_selection_or_none() {
        let p = DevicePicker::new(Dir::Input, sample());
        let (d, c) = p.find(Some("ASIO: MOTU MicroBook ASIO"), Some("3: Guitar")).unwrap();
        assert_eq!((d.as_str(), c.channels), ("ASIO: MOTU MicroBook ASIO", vec![2]));
        // ドライバ未保存(以前の設定)は WASAPI のデバイス名として扱う
        let (d, c) = p.find(None, Some("マイク (USB)")).unwrap();
        assert_eq!((d.as_str(), c.device_id.as_deref()), ("WASAPI", Some("wasapi:{a}")));
        // ドライバだけ分かればその先頭候補
        assert_eq!(p.find(Some("ASIO: Realtek ASIO"), None).unwrap().1.label, "1: L");
        // 消えたドライバ・チャンネル
        assert!(p.find(Some("ASIO: Gone"), Some("1: L")).is_none());
        assert!(p.find(Some("ASIO: Realtek ASIO"), Some("9: X")).is_none());
        assert_eq!(p.system_default().1.device_id, None);
    }
}
