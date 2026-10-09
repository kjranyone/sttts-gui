//! オーディオデバイスの 2 段選択(ドライバ → 候補)の状態と適用。
//!
//! 候補の組み立ては `crate::device_picker`。ここは Select との同期と、
//! 選択をエンジン(入力)・再生ストリーム(出力)へ反映する部分。

use gpui_kit::component::IndexPath;
use gpui_kit::component::select::SelectState;
use gpui_kit::*;
use sttts_i18n::{tr, trf};
use sttts_protocol::{AudioConfig, AudioDeviceInfo, GuiMessage};

use super::{MicTransition, StttsApp};
use crate::audio;
use crate::device_picker::{Choice, DevicePicker, Dir, WASAPI_DRIVER};

/// 1 方向(入力 or 出力)のドライバ Select と候補 Select。
pub(super) struct DeviceSelect {
    pub picker: DevicePicker,
    pub driver_select: Entity<SelectState<Vec<String>>>,
    pub choice_select: Entity<SelectState<Vec<String>>>,
    pub driver: String,
    pub choice: Choice,
    /// Select の項目更新は window が要るため、render で反映する(`sync`)
    dirty: bool,
}

impl DeviceSelect {
    pub fn new(
        dir: Dir,
        devices: Vec<AudioDeviceInfo>,
        saved: (Option<&str>, Option<&str>),
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        let picker = DevicePicker::new(dir, devices);
        let (driver, choice) = picker.find(saved.0, saved.1).unwrap_or_else(|| picker.system_default());
        let drivers = picker.drivers();
        let labels = labels(&picker, &driver);
        let d_ix = drivers.iter().position(|d| d == &driver).map(IndexPath::new);
        let c_ix = labels.iter().position(|l| l == &choice.label).map(IndexPath::new);
        let driver_select = cx.new(|cx| SelectState::new(drivers, d_ix, window, cx));
        let choice_select = cx.new(|cx| SelectState::new(labels, c_ix, window, cx));
        Self { picker, driver_select, choice_select, driver, choice, dirty: false }
    }

    /// デバイス一覧を差し替える。`saved` があればそれを復元し、無ければ現在の選択を保つ
    /// (消えていたら既定へ)。選択が変わったら true。
    pub fn set_devices(&mut self, devices: Vec<AudioDeviceInfo>, dir: Dir, saved: Option<(Option<String>, Option<String>)>) -> bool {
        self.picker = DevicePicker::new(dir, devices);
        let found = match &saved {
            Some((d, l)) => self.picker.find(d.as_deref(), l.as_deref()),
            None => self.picker.find(Some(&self.driver), Some(&self.choice.label)),
        };
        let (driver, choice) = found.unwrap_or_else(|| self.picker.system_default());
        let changed = driver != self.driver || choice != self.choice;
        self.driver = driver;
        self.choice = choice;
        self.dirty = true;
        changed
    }

    /// ドライバを選ぶ。候補はそのドライバの先頭(WASAPI は既定、ASIO は 1ch 目 / 1+2ch)。
    pub fn select_driver(&mut self, driver: &str) -> bool {
        let Some((driver, choice)) = self.picker.find(Some(driver), None) else { return false };
        self.driver = driver;
        self.choice = choice;
        self.dirty = true;
        true
    }

    /// システム既定へ戻す(保存済みのデバイスを開けなかったとき)。
    pub fn select_default(&mut self) {
        (self.driver, self.choice) = self.picker.system_default();
        self.dirty = true;
    }

    pub fn select_choice(&mut self, label: &str) -> bool {
        let Some(choice) = self.picker.choices(&self.driver).into_iter().find(|c| c.label == label) else {
            return false;
        };
        self.choice = choice;
        true
    }

    /// 保存用: (ドライバ, 候補ラベル)。WASAPI・システム既定は None。
    pub fn saved(&self) -> (Option<String>, Option<String>) {
        let driver = (self.driver != WASAPI_DRIVER).then(|| self.driver.clone());
        let label = self.choice.device_id.is_some().then(|| self.choice.label.clone());
        (driver, label)
    }

    /// 2 段目の見出し(WASAPI はデバイス、ASIO はチャンネル)。
    pub fn choice_caption(&self) -> &'static str {
        if self.driver == WASAPI_DRIVER {
            tr!("Device", "デバイス", "设备")
        } else {
            tr!("Channel", "チャンネル", "声道")
        }
    }

    /// 表示言語の切替後: 「既定の〜デバイス」の文言を今の言語で作り直す
    /// (保存するのはデバイスの選択だけで、既定の文言は保存しない)。
    pub fn relocalize(&mut self) {
        if self.choice.device_id.is_none() {
            self.choice = self.picker.system_default().1;
        }
        self.dirty = true;
    }

    /// 表示用の短い名前(ライブ欄など)。
    pub fn display(&self) -> String {
        if self.driver == WASAPI_DRIVER {
            self.choice.label.clone()
        } else {
            format!("{} — {}", self.driver, self.choice.label)
        }
    }

    pub fn sync(&mut self, window: &mut Window, cx: &mut App) {
        if !std::mem::take(&mut self.dirty) {
            return;
        }
        let drivers = self.picker.drivers();
        let labels = labels(&self.picker, &self.driver);
        let (driver, label) = (self.driver.clone(), self.choice.label.clone());
        self.driver_select.update(cx, |s, cx| {
            s.set_items(drivers, window, cx);
            s.set_selected_value(&driver, window, cx);
        });
        self.choice_select.update(cx, |s, cx| {
            s.set_items(labels, window, cx);
            s.set_selected_value(&label, window, cx);
        });
    }
}

fn labels(picker: &DevicePicker, driver: &str) -> Vec<String> {
    picker.choices(driver).into_iter().map(|c| c.label).collect()
}

/// sttts-audio の列挙結果をプロトコルの型へ(入力はエンジンから同じ型で届く)。
pub(super) fn to_protocol(devices: Vec<sttts_audio::DeviceInfo>) -> Vec<AudioDeviceInfo> {
    devices
        .into_iter()
        .map(|d| AudioDeviceInfo {
            id: d.id,
            host: d.host,
            name: d.name,
            default_rate: Some(d.default_rate),
            is_default: d.is_default,
            channels: d.channels,
        })
        .collect()
}

impl StttsApp {
    /// エンジンから入力デバイス一覧が届いた。保存済みの選択を復元できたらエンジンへ反映する。
    pub(super) fn on_input_devices(&mut self, inputs: Vec<AudioDeviceInfo>, cx: &mut Context<Self>) {
        let saved = self.saved_input.take();
        let restoring = saved.is_some();
        self.input_dev.set_devices(inputs, Dir::Input, saved);
        if restoring && self.input_dev.choice.device_id.is_some() {
            self.send_input_config();
        }
        cx.notify();
    }

    pub(super) fn apply_input_driver(&mut self, driver: String, cx: &mut Context<Self>) {
        if self.input_dev.driver != driver && self.input_dev.select_driver(&driver) {
            self.commit_input(cx);
        }
    }

    pub(super) fn apply_input_choice(&mut self, label: String, cx: &mut Context<Self>) {
        if self.input_dev.choice.label != label && self.input_dev.select_choice(&label) {
            self.commit_input(cx);
        }
    }

    fn send_input_config(&mut self) {
        let c = self.input_dev.choice.clone();
        self.send(GuiMessage::Configure {
            tts: None,
            asr: None,
            audio: Some(AudioConfig { input_device: c.device_id, input_channels: c.channels }),
            voice: None,
            pipeline: None,
        });
    }

    /// 入力選択の適用。ライブ中は停止して、再開は利用者に任せる。
    fn commit_input(&mut self, cx: &mut Context<Self>) {
        self.send_input_config();
        if self.mic_running && self.mic_transition == MicTransition::None {
            // 停止→即再開はbackendの再開クールダウンに拒否されるうえ、デバイスの
            // 短時間反復 open/close はドライバクラッシュの原因。停止のみ送り、
            // 再開は利用者の操作(ライブ開始)に任せる。
            self.mic_transition = MicTransition::Stopping;
            self.send(GuiMessage::StopSession);
            self.push_log(
                tr!(
                    "Live stopped because the input device changed. Press \"Start live\" to use the new device",
                    "入力デバイスを変更したためライブを停止しました。新しいデバイスで「ライブ開始」を押してください",
                    "输入设备已更改,直播已停止。请按「开始直播」使用新设备"
                )
                .into(),
            );
        }
        self.persist_settings(cx);
        cx.notify();
    }

    pub(super) fn apply_output_driver(&mut self, driver: String, cx: &mut Context<Self>) {
        if self.output_dev.driver != driver && self.output_dev.select_driver(&driver) {
            self.commit_output(cx);
        }
    }

    pub(super) fn apply_output_choice(&mut self, label: String, cx: &mut Context<Self>) {
        if self.output_dev.choice.label != label && self.output_dev.select_choice(&label) {
            self.commit_output(cx);
        }
    }

    /// 出力選択の適用(ストリームを張り直す。未再生キューは破棄)。
    fn commit_output(&mut self, cx: &mut Context<Self>) {
        // 旧ストリームを先に閉じる(ASIO は同時に 1 ドライバしか開けないため、張り替え時に重ねない)
        if let Some(old) = self.audio.take() {
            old.clear();
        }
        self.playback.clear();
        let c = self.output_dev.choice.clone();
        match audio::AudioOut::open(c.device_id.as_deref(), &c.channels) {
            Ok(out) => {
                self.audio = Some(out);
                let name = self.output_dev.display();
                self.push_log(trf!("Output device: {name}", "出力デバイスを切替: {name}", "已切换输出设备:{name}"));
            }
            Err(e) => self.push_log(trf!(
                "[error:audio] Failed to switch the output device: {e:#}",
                "[error:audio] 出力デバイスの切替に失敗: {e:#}",
                "[error:audio] 切换输出设备失败:{e:#}"
            )),
        }
        self.persist_settings(cx);
        cx.notify();
    }
}
