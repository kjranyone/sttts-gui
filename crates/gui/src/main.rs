//! sttts-gui — GPUI(gpui-kit)クライアントのエントリポイント。
//!
//! - バックエンド(Python)を子プロセス起動し stdio NDJSON で往復
//! - テキスト入力 → チャンク分割 → TTS 合成 → rodio で逐次再生(疑似ストリーミング)
//! - マイク → ASR の部分/確定文字起こし表示、自動発話

mod audio;
mod backend;
mod settings;

use std::collections::VecDeque;
use std::path::PathBuf;

use gpui_kit::component::button::*;
use gpui_kit::component::input::{Input, InputState, Textarea, TextareaState};
use gpui_kit::component::select::{Select, SelectEvent, SelectState};
use gpui_kit::component::switch::Switch;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder;
use gpui_kit::*;
use sttts_protocol::{
    AnyMessage, AudioConfig, AudioDeviceInfo, BackendMessage, EngineState, GuiMessage, ModelInfo,
    PipelineConfig, TtsConfig, VoiceConfig,
};

const DEFAULT_INPUT_LABEL: &str = "既定の入力デバイス";
const DEFAULT_OUTPUT_LABEL: &str = "既定のデバイス";

/// UI に並べる文字起こし1件。
#[derive(Debug, Clone)]
struct TranscriptEntry {
    utterance: u64,
    final_text: Option<String>,
    partial: Option<String>,
}

/// 生成済みチャンク(履歴)。
#[derive(Debug, Clone)]
struct HistoryItem {
    text: String,
    path: Option<String>,
    gen_ms: u64,
}

pub struct StttsApp {
    mock: bool,
    backend: Option<backend::BackendHandle>,
    models: Vec<ModelInfo>,
    selected_model_id: String,
    tts_state: EngineState,
    asr_state: EngineState,
    mic_running: bool,

    speak_input: Entity<TextareaState>,
    caption_input: Entity<InputState>,
    seed_input: Entity<InputState>,
    random_seed: bool,
    auto_speak: bool,

    input_select: Entity<SelectState<Vec<String>>>,
    output_select: Entity<SelectState<Vec<String>>>,
    input_devices: Vec<AudioDeviceInfo>,
    /// デバイス一覧到着後に render(windowあり)で Select へ反映するための保留領域
    pending_input_items: Option<Vec<String>>,
    selected_input_name: Option<String>,
    selected_output_name: Option<String>,
    saved_input_device: Option<String>,
    mic_level_db: f32,

    transcript: Vec<TranscriptEntry>,
    history: Vec<HistoryItem>,
    /// 現在合成中/直前のチャンクテキスト(tts_chunk_start で設定、tts_audio で履歴へ)
    pending_chunk_text: Option<String>,
    logs: VecDeque<String>,
    last_gen_ms: Option<u64>,
    speaking: bool,
    /// 受付済み最大 request id(speak_accepted)
    last_accepted_request: u64,
    /// キャンセル時点の last_accepted_request。これ以下の request の音声は捨てる
    /// (キャンセル前に送出済みでパイプ上にあった tts_audio を鳴らさないため)
    cancelled_upto: u64,
    audio: Option<crate::audio::AudioOut>,
    /// Select 等のイベント購読(gpui は Subscription を drop すると購読解除になる)
    subscriptions: Vec<gpui::Subscription>,
    status_hint: String,
    root: PathBuf,
}

impl StttsApp {
    fn new(mock: bool, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let root = backend::repo_root();
        let saved = settings::AppSettings::load(&root);
        let auto_speak = saved.auto_speak.unwrap_or(true);
        let random_seed = saved.random_seed.unwrap_or(true);

        let speak_input = cx.new(|cx| {
            TextareaState::new(window, cx)
                .placeholder("話させたいテキストを入力(送信は発話ボタン)")
                .auto_grow(3, 8)
        });
        let caption_input = cx.new(|cx| {
            let mut state = InputState::new(window, cx)
                .placeholder("Voice Design キャプション(例: 落ち着いた、近い距離感の女性話者)");
            if let Some(caption) = &saved.caption {
                state.set_value(caption.as_str(), window, cx);
            }
            state
        });
        let seed_input = cx.new(|cx| InputState::new(window, cx).placeholder("seed(空欄=ランダム)"));

        // --- 入出力デバイス選択
        let saved_input_device = saved.input_device.clone();
        let output_devices = crate::audio::list_output_devices();
        let mut output_items = vec![DEFAULT_OUTPUT_LABEL.to_string()];
        output_items.extend(output_devices.iter().cloned());
        let saved_output_device = saved
            .output_device
            .clone()
            .filter(|n| output_devices.iter().any(|d| d == n));
        let output_sel_ix = saved_output_device
            .as_ref()
            .and_then(|n| output_devices.iter().position(|d| d == n))
            .map(|p| IndexPath::new(p + 1));
        let output_select = cx.new(|cx| {
            SelectState::new(output_items, output_sel_ix, window, cx)
        });
        let input_select = cx.new(|cx| SelectState::new(Vec::new(), None, window, cx));

        let audio = match crate::audio::AudioOut::open(saved_output_device.as_deref())
            .or_else(|_| crate::audio::AudioOut::open(None))
        {
            Ok(a) => (Some(a), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };
        let (audio, audio_error) = audio;

        let selected_model_id = saved.tts_model.unwrap_or_else(|| "v4.1-small-mf".into());

        let mut app = Self {
            mock,
            backend: None,
            models: Vec::new(),
            selected_model_id,
            tts_state: EngineState {
                phase: "idle".into(),
                detail: None,
                model: None,
            },
            asr_state: EngineState {
                phase: "idle".into(),
                detail: None,
                model: None,
            },
            mic_running: false,
            speak_input,
            caption_input,
            seed_input,
            random_seed,
            auto_speak,
            input_select,
            output_select,
            input_devices: Vec::new(),
            pending_input_items: None,
            selected_input_name: None,
            selected_output_name: saved_output_device,
            saved_input_device,
            mic_level_db: -100.0,
            transcript: Vec::new(),
            history: Vec::new(),
            pending_chunk_text: None,
            logs: VecDeque::new(),
            last_gen_ms: None,
            speaking: false,
            last_accepted_request: 0,
            cancelled_upto: 0,
            audio,
            subscriptions: Vec::new(),
            status_hint: "バックエンドを起動中…".into(),
            root,
        };
        if let Some(err) = audio_error {
            app.push_log(format!("出力デバイスを開けませんでした: {err}"));
        }
        app.start_backend(cx);

        // デバイス選択イベント(選択は window スコープで届く)。
        // Subscription は drop すると購読解除になるため保持する。
        let weak_input = cx.weak_entity();
        app.subscriptions.push(window.subscribe(
            &app.input_select,
            cx,
            move |_, event, window, cx| {
                if let (Some(app), SelectEvent::Confirm(Some(name))) =
                    (weak_input.upgrade(), event)
                {
                    app.update(cx, |app, cx| {
                        app.apply_input_device(name.clone(), window, cx)
                    });
                }
            },
        ));
        let weak_output = cx.weak_entity();
        app.subscriptions.push(window.subscribe(
            &app.output_select,
            cx,
            move |_, event, _window, cx| {
                if let (Some(app), SelectEvent::Confirm(Some(name))) =
                    (weak_output.upgrade(), event)
                {
                    app.update(cx, |app, cx| app.apply_output_device(name.clone(), cx));
                }
            },
        ));

        // 初期設定を backend へ反映
        app.send(GuiMessage::Configure {
            tts: Some(TtsConfig {
                model: Some(app.selected_model_id.clone()),
                ..Default::default()
            }),
            asr: None,
            audio: None,
            voice: Some(VoiceConfig {
                no_ref: Some(true),
                ..Default::default()
            }),
            pipeline: Some(PipelineConfig {
                auto_speak: Some(app.auto_speak),
                ..Default::default()
            }),
        });
        app
    }

    fn start_backend(&mut self, cx: &mut Context<Self>) {
        let output_dir: PathBuf = self.root.join("output");
        let _ = std::fs::create_dir_all(&output_dir);
        let spawn_cfg = backend::default_spawn(self.mock, &output_dir);

        let (tx_events, rx_events) = async_channel::unbounded::<AnyMessage>();
        let (tx_stderr, rx_stderr) = async_channel::unbounded::<String>();

        let handle = match backend::BackendHandle::spawn(spawn_cfg, tx_events, tx_stderr) {
            Ok(h) => h,
            Err(e) => {
                self.push_log(format!("バックエンド起動エラー: {e:#}"));
                self.status_hint = "バックエンド起動エラー".into();
                return;
            }
        };
        self.backend = Some(handle);
        self.status_hint = "バックエンド接続待ち…".into();

        // backend → UI の取り込みループ
        cx.spawn(async move |this, cx| {
            while let Ok(msg) = rx_events.recv().await {
                if this
                    .update(cx, |app, cx| app.on_backend_message(msg, cx))
                    .is_err()
                {
                    return;
                }
            }
            // チャネル閉鎖 = バックエンド終了
            let _ = this.update(cx, |app, cx| {
                app.push_log("バックエンドとの接続が切れました".into());
                app.status_hint = "バックエンド停止".into();
                app.mic_running = false;
                app.speaking = false;
                cx.notify();
            });
        })
        .detach();

        // stderr(人間可読ログ)の取り込みループ
        cx.spawn(async move |this, cx| {
            while let Ok(line) = rx_stderr.recv().await {
                if this
                    .update(cx, |app, _cx| app.push_log(format!("[py] {line}")))
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
    }

    fn send(&mut self, msg: GuiMessage) {
        if let Some(b) = &self.backend {
            if let Err(e) = b.send(&msg) {
                self.push_log(format!("送信失敗: {e}"));
            }
        } else {
            self.push_log("バックエンド未接続".into());
        }
    }

    fn push_log(&mut self, line: String) {
        self.logs.push_back(line);
        while self.logs.len() > 300 {
            self.logs.pop_front();
        }
    }

    fn persist_settings(&self, cx: &App) {
        let caption = self.caption_input.read(cx).value().to_string();
        let saved = settings::AppSettings {
            mock: Some(self.mock),
            tts_model: Some(self.selected_model_id.clone()),
            caption: Some(caption),
            auto_speak: Some(self.auto_speak),
            random_seed: Some(self.random_seed),
            input_device: self.selected_input_name.clone(),
            output_device: self.selected_output_name.clone(),
        };
        saved.save(&self.root);
    }

    fn on_backend_message(&mut self, any: AnyMessage, cx: &mut Context<Self>) {
        let msg = match any {
            AnyMessage::Known(m) => m,
            AnyMessage::Unknown(v) => {
                self.push_log(format!("[?] {}", serde_json::to_string(&v).unwrap_or_default()));
                return;
            }
        };
        match msg {
            BackendMessage::Hello { protocol, mock, models, .. } => {
                if protocol != sttts_protocol::PROTOCOL_VERSION {
                    self.push_log(format!(
                        "プロトコル不一致: backend={protocol} gui={}",
                        sttts_protocol::PROTOCOL_VERSION
                    ));
                }
                self.mock = mock;
                self.models = models;
                // backend(再)起動で request id は 1 から振り直される
                self.last_accepted_request = 0;
                self.cancelled_upto = 0;
                self.status_hint = if mock {
                    "モック接続".into()
                } else {
                    "実エンジン接続".into()
                };
            }
            BackendMessage::State { tts, asr, mic_running } => {
                self.tts_state = tts;
                self.asr_state = asr;
                self.mic_running = mic_running;
            }
            BackendMessage::Log { level, message } => {
                self.push_log(format!("[{level}] {message}"));
            }
            BackendMessage::MicLevel { db, .. } => {
                self.mic_level_db = db;
            }
            BackendMessage::AsrPartial { utterance, text } => {
                self.upsert_transcript(utterance, None, Some(text));
            }
            BackendMessage::AsrFinal { utterance, text } => {
                self.upsert_transcript(utterance, Some(text), None);
            }
            BackendMessage::SpeakAccepted { request, origin, .. } => {
                self.last_accepted_request = self.last_accepted_request.max(request);
                self.speaking = true;
                self.push_log(format!("発話受付 request={request} origin={origin}"));
            }
            BackendMessage::TtsChunkStart { text, .. } => {
                self.pending_chunk_text = Some(text);
            }
            BackendMessage::TtsAudio {
                request,
                wav_base64,
                gen_ms,
                path,
                ..
            } => {
                if request <= self.cancelled_upto {
                    self.pending_chunk_text = None;
                    self.push_log(format!("キャンセル済み request={request} の音声を破棄"));
                    return;
                }
                self.last_gen_ms = Some(gen_ms);
                self.history.push(HistoryItem {
                    text: self.pending_chunk_text.take().unwrap_or_default(),
                    path,
                    gen_ms,
                });
                if self.history.len() > 100 {
                    self.history.remove(0);
                }
                if let Some(audio) = &self.audio {
                    if let Err(e) = audio.enqueue_wav_base64(&wav_base64) {
                        self.push_log(format!("音声キュー追加失敗: {e}"));
                    }
                }
            }
            BackendMessage::TtsChunkDone { .. } => {}
            BackendMessage::SpeakDone { request, chunks, cancelled, failed } => {
                self.speaking = false;
                self.push_log(format!(
                    "発話完了 request={request} chunks={chunks} cancelled={cancelled} failed={failed}"
                ));
            }
            BackendMessage::Error { scope, message, .. } => {
                self.push_log(format!("[error:{scope}] {message}"));
            }
            BackendMessage::Pong { .. } => {}
            BackendMessage::Devices { inputs, .. } => {
                self.input_devices = inputs;
                let mut items = vec![DEFAULT_INPUT_LABEL.to_string()];
                items.extend(self.input_devices.iter().map(|d| d.name.clone()));
                // 保存済みデバイスが一覧にあれば復元し、バックエンドへも反映する
                let restored = self
                    .saved_input_device
                    .take()
                    .filter(|n| self.input_devices.iter().any(|d| &d.name == n));
                if let Some(name) = &restored {
                    self.selected_input_name = Some(name.clone());
                    let index = self
                        .input_devices
                        .iter()
                        .find(|d| &d.name == name)
                        .map(|d| d.index);
                    self.send(GuiMessage::Configure {
                        tts: None,
                        asr: None,
                        audio: Some(AudioConfig {
                            input_device_index: index,
                        }),
                        voice: None,
                        pipeline: None,
                    });
                }
                self.pending_input_items = Some(items);
            }
        }
        cx.notify();
    }

    fn upsert_transcript(
        &mut self,
        utterance: u64,
        final_text: Option<String>,
        partial: Option<String>,
    ) {
        if let Some(entry) = self
            .transcript
            .iter_mut()
            .rev()
            .find(|e| e.utterance == utterance)
        {
            if final_text.is_some() {
                entry.final_text = final_text;
                entry.partial = None;
            } else {
                entry.partial = partial;
            }
            return;
        }
        self.transcript.push(TranscriptEntry {
            utterance,
            final_text,
            partial,
        });
        while self.transcript.len() > 500 {
            self.transcript.remove(0);
        }
    }

    // ---------- アクション ----------

    fn speak_from_input(&mut self, _ev: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        let text = self.speak_input.read(cx).value().to_string();
        let text = text.trim().to_string();
        if text.is_empty() {
            self.push_log("テキストが空です".into());
            return;
        }
        let caption = {
            let c = self.caption_input.read(cx).value().to_string();
            let c = c.trim().to_string();
            (!c.is_empty()).then_some(c)
        };
        let seed = if self.random_seed {
            None
        } else {
            let s = self.seed_input.read(cx).value().trim().to_string();
            s.parse::<i64>().ok()
        };

        self.send(GuiMessage::Speak {
            text,
            caption,
            ref_wavs: None,
            seed,
            tag: None,
        });
        self.speak_input.update(cx, |s, cx| {
            s.set_value("", window, cx);
        });
        self.persist_settings(cx);
    }

    fn replay_history(&mut self, index: usize, _ev: &ClickEvent, _w: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = self.history.get(index) else {
            return;
        };
        let Some(path) = &item.path else {
            self.push_log("この履歴にはファイルがありません".into());
            return;
        };
        if let Some(audio) = &self.audio {
            match std::fs::read(path) {
                Ok(bytes) => match audio.enqueue_wav_bytes(bytes) {
                    Ok(()) => self.push_log(format!("再生: {path}")),
                    Err(e) => self.push_log(format!("再生失敗: {e}")),
                },
                Err(e) => self.push_log(format!("ファイル読込失敗: {e}")),
            }
        }
        let _ = cx;
    }

    fn select_model(&mut self, index: usize, _ev: &ClickEvent, _w: &mut Window, cx: &mut Context<Self>) {
        let model_id = match self.models.get(index) {
            Some(m) => m.id.clone(),
            None => return,
        };
        let label = self
            .models
            .get(index)
            .map(|m| m.label.clone())
            .unwrap_or_else(|| model_id.clone());
        self.selected_model_id.clone_from(&model_id);
        self.send(GuiMessage::Configure {
            tts: Some(TtsConfig {
                model: Some(model_id.clone()),
                ..Default::default()
            }),
            asr: None,
            audio: None,
            voice: None,
            pipeline: None,
        });
        self.push_log(format!("モデル切替: {label}"));
        self.persist_settings(cx);
    }

    fn toggle_auto_speak(&mut self, checked: &bool, _w: &mut Window, _cx: &mut Context<Self>) {
        self.auto_speak = *checked;
        self.send(GuiMessage::Configure {
            tts: None,
            asr: None,
            audio: None,
            voice: None,
            pipeline: Some(PipelineConfig {
                auto_speak: Some(*checked),
                ..Default::default()
            }),
        });
        self.persist_settings(_cx);
    }

    fn toggle_random_seed(&mut self, checked: &bool, _w: &mut Window, _cx: &mut Context<Self>) {
        self.random_seed = *checked;
        self.persist_settings(_cx);
    }

    fn toggle_mic(&mut self, _ev: &ClickEvent, _window: &mut Window, _cx: &mut Context<Self>) {
        if self.mic_running {
            self.send(GuiMessage::StopSession);
        } else {
            self.send(GuiMessage::StartSession);
        }
    }

    /// 入力デバイス選択の適用。マイク実行中はセッションを張り直す。
    fn apply_input_device(&mut self, name: String, _window: &mut Window, cx: &mut Context<Self>) {
        let index = if name == DEFAULT_INPUT_LABEL {
            None
        } else {
            self.input_devices
                .iter()
                .find(|d| d.name == name)
                .map(|d| d.index)
        };
        self.selected_input_name = (index.is_some()).then_some(name);
        self.send(GuiMessage::Configure {
            tts: None,
            asr: None,
            audio: Some(AudioConfig {
                input_device_index: index,
            }),
            voice: None,
            pipeline: None,
        });
        if self.mic_running {
            self.send(GuiMessage::StopSession);
            self.send(GuiMessage::StartSession);
            self.push_log("入力デバイスを変更したためマイクセッションを再起動します".into());
        }
        self.persist_settings(cx);
    }

    /// 出力デバイス選択の適用(ストリームを張り直す。未再生キューは破棄)。
    fn apply_output_device(&mut self, name: String, cx: &mut Context<Self>) {
        let preferred = (name != DEFAULT_OUTPUT_LABEL).then_some(name.clone());
        match crate::audio::AudioOut::open(preferred.as_deref()) {
            Ok(out) => {
                if let Some(old) = self.audio.as_ref() {
                    old.clear();
                }
                self.audio = Some(out);
                self.selected_output_name = (preferred.is_some()).then_some(name.clone());
                self.push_log(format!("出力デバイスを切替: {name}"));
            }
            Err(e) => {
                self.push_log(format!("出力デバイスの切替に失敗: {e:#}"));
            }
        }
        self.persist_settings(cx);
    }

    fn cancel_speak(&mut self, _ev: &ClickEvent, _window: &mut Window, _cx: &mut Context<Self>) {
        self.send(GuiMessage::CancelSpeak);
        self.cancelled_upto = self.last_accepted_request;
        if let Some(audio) = &self.audio {
            // clear() は内部で play() し直す(rodio の clear は Sink を pause するため)
            audio.clear();
        }
    }

    fn quit(&mut self, _ev: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        self.persist_settings(cx);
        if let Some(b) = &self.backend {
            b.shutdown();
        }
        cx.quit();
    }

    fn phase_label(state: &EngineState) -> String {
        match state.phase.as_str() {
            "ready" => "準備完了".into(),
            "loading" => format!("ロード中 {}", state.detail.clone().unwrap_or_default()),
            "error" => format!("エラー: {}", state.detail.clone().unwrap_or_default()),
            _ => "未ロード".into(),
        }
    }
}

impl Render for StttsApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // デバイス一覧は window を要求する API なので render で Select へ反映する
        if let Some(items) = self.pending_input_items.take() {
            let selected = self
                .selected_input_name
                .clone()
                .unwrap_or_else(|| DEFAULT_INPUT_LABEL.to_string());
            self.input_select.update(cx, |s, cx| {
                s.set_items(items, window, cx);
                s.set_selected_value(&selected, window, cx);
            });
        }

        // 入力レベルメータ(-60..0dB を 0..1 へマップ)
        let level_frac = ((self.mic_level_db + 60.0) / 60.0).clamp(0.0, 1.0);
        let level_color = if !self.mic_running {
            rgb(0x3a3f46)
        } else if level_frac < 0.5 {
            rgb(0x3fb950)
        } else if level_frac < 0.8 {
            rgb(0xd29922)
        } else {
            rgb(0xf85149)
        };
        let level_db_text = if self.mic_running {
            format!("{:+.0} dB", self.mic_level_db.max(-99.0))
        } else {
            "—".to_string()
        };

        let mock_label = if self.mock { " [mock]" } else { "" };

        let header = h_flex()
            .gap_2()
            .items_center()
            .px_3()
            .py_2()
            .border_b_1()
            .child(
                div()
                    .font_weight(FontWeight::BOLD)
                    .child(format!("sttts-gui{mock_label}")),
            )
            .child(div().text_sm().text_color(rgb(0x9aa0a6)).child(format!(
                "TTS: {} / ASR: {}",
                Self::phase_label(&self.tts_state),
                Self::phase_label(&self.asr_state)
            )))
            .child(div().flex_1())
            .child(
                Button::new("mic")
                    .label(if self.mic_running { "マイク停止" } else { "マイク開始" })
                    .on_click(cx.listener(Self::toggle_mic)),
            )
            .child(
                Button::new("cancel")
                    .label("キュー取消")
                    .on_click(cx.listener(Self::cancel_speak)),
            )
            .child(Button::new("quit").label("終了").on_click(cx.listener(Self::quit)));

        // 左: 文字起こし
        let transcript_items: Vec<_> = self
            .transcript
            .iter()
            .map(|e| {
                let (text, color) = if let Some(f) = &e.final_text {
                    (f.clone(), rgb(0xe8eaed))
                } else {
                    (e.partial.clone().unwrap_or_default(), rgb(0x9aa0a6))
                };
                div().child(text).text_color(color).py_0p5().into_any_element()
            })
            .collect();
        let empty_hint = self.transcript.is_empty().then(|| {
            div()
                .text_color(rgb(0x6b7075))
                .child("まだ文字起こしはありません。「マイク開始」で話しかけてください。")
                .into_any_element()
        });

        // 右: 発話パネル
        let model_buttons: Vec<_> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let selected = m.id == self.selected_model_id;
                Button::new(SharedString::from(format!("model-{i}")))
                    .label(if selected { format!("● {}", m.id) } else { m.id.clone() })
                    .when(selected, |b| b.primary())
                    .on_click(cx.listener(move |this, ev, w, cx| this.select_model(i, ev, w, cx)))
                    .into_any_element()
            })
            .collect();
        let models_loaded = !self.models.is_empty();

        let history_items: Vec<_> = self
            .history
            .iter()
            .enumerate()
            .rev()
            .take(15)
            .map(|(i, h)| {
                h_flex()
                    .gap_2()
                    .items_center()
                    .py_0p5()
                    .child(
                        div()
                            .flex_1()
                            .text_sm()
                            .text_color(rgb(0xbdc1c6))
                            .child(truncate(&h.text, 24)),
                    )
                    .child(
                        div().text_xs().text_color(rgb(0x6b7075)).child(format!("{}ms", h.gen_ms)),
                    )
                    .child(
                        Button::new(SharedString::from(format!("replay-{i}")))
                            .label("再生")
                            .compact()
                            .on_click(cx.listener(move |this, ev, w, cx| {
                                this.replay_history(i, ev, w, cx)
                            })),
                    )
                    .into_any_element()
            })
            .collect();

        let last_gen_line = match self.last_gen_ms {
            Some(ms) => format!("直近チャンク生成: {ms}ms"),
            None => "直近の生成なし".into(),
        };

        // Switch の on_click は &mut App を受けるため弱参照経由で Self を更新する
        let weak_auto_speak = cx.weak_entity();
        let weak_random_seed = cx.weak_entity();

        div()
            .size_full()
            .bg(rgb(0x1e1f22))
            .text_color(rgb(0xe8eaed))
            .child(
                v_flex()
                    .size_full()
                    .child(header)
                    .child(
                        h_flex()
                            .flex_1()
                            .gap_2()
                            .p_2()
                            .overflow_hidden()
                            .child(
                                v_flex()
                                    .id("transcript")
                                    .flex_1()
                                    .h_full()
                                    .p_2()
                                    .rounded_md()
                                    .bg(rgb(0x26282c))
                                    .overflow_y_scroll()
                                    .text_sm()
                                    .children(empty_hint.into_iter().chain(transcript_items)),
                            )
                            .child(
                                v_flex()
                                    .id("speak-panel")
                                    .w(px(430.))
                                    .h_full()
                                    .p_2()
                                    .rounded_md()
                                    .bg(rgb(0x26282c))
                                    .overflow_y_scroll()
                                    .gap_2()
                                    .child(
                                        div()
                                            .font_weight(FontWeight::BOLD)
                                            .text_sm()
                                            .child("テキストから発話"),
                                    )
                                    .child(Textarea::new(&self.speak_input).text_sm())
                                    .child(
                                        // 入出力デバイス + レベルメータ
                                        v_flex()
                                            .gap_1()
                                            .child(
                                                h_flex()
                                                    .gap_2()
                                                    .items_center()
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(rgb(0x9aa0a6))
                                                            .w(px(28.))
                                                            .child("入力"),
                                                    )
                                                    .child(
                                                        div().flex_1().child(
                                                            Select::new(&self.input_select)
                                                                .placeholder(DEFAULT_INPUT_LABEL)
                                                                .text_sm(),
                                                        ),
                                                    ),
                                            )
                                            .child(
                                                h_flex()
                                                    .gap_2()
                                                    .items_center()
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(rgb(0x9aa0a6))
                                                            .w(px(28.))
                                                            .child("出力"),
                                                    )
                                                    .child(
                                                        div().flex_1().child(
                                                            Select::new(&self.output_select)
                                                                .placeholder(DEFAULT_OUTPUT_LABEL)
                                                                .text_sm(),
                                                        ),
                                                    ),
                                            )
                                            .child(
                                                // レベルメータ(ASR 入力レベル)
                                                h_flex()
                                                    .gap_2()
                                                    .items_center()
                                                    .child(
                                                        div().w(px(28.)).child(""),
                                                    )
                                                    .child(
                                                        div()
                                                            .flex_1()
                                                            .h(px(10.))
                                                            .rounded_sm()
                                                            .bg(rgb(0x30343a))
                                                            .overflow_hidden()
                                                            .child(
                                                                div()
                                                                    .h_full()
                                                                    .w(relative(level_frac))
                                                                    .bg(level_color),
                                                            ),
                                                    )
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(rgb(0x9aa0a6))
                                                            .child(level_db_text),
                                                    ),
                                            ),
                                    )
                                    .child(
                                        div().text_sm().child(Input::new(&self.caption_input).text_sm()),
                                    )
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(
                                                div().w(px(180.)).child(
                                                    Input::new(&self.seed_input).text_sm(),
                                                ),
                                            )
                                            .child(
                                                Switch::new("random-seed")
                                                    .checked(self.random_seed)
                                                    .label("ランダムseed")
                                                    .on_click(move |checked, window, cx| {
                                                        if let Some(app) = weak_random_seed.upgrade() {
                                                            app.update(cx, |this, cx| {
                                                                this.toggle_random_seed(checked, window, cx)
                                                            });
                                                        }
                                                    }),
                                            ),
                                    )
                                    .child({
                                        let weak = weak_auto_speak.clone();
                                        Switch::new("auto-speak")
                                            .checked(self.auto_speak)
                                            .label("ASR確定文を自動発話")
                                            .on_click(move |checked, window, cx| {
                                                if let Some(app) = weak.upgrade() {
                                                    app.update(cx, |this, cx| {
                                                        this.toggle_auto_speak(checked, window, cx)
                                                    });
                                                }
                                            })
                                    })
                                    .child(
                                        Button::new("speak")
                                            .primary()
                                            .label(if self.speaking { "発話中…" } else { "発話" })
                                            .on_click(cx.listener(Self::speak_from_input)),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x6b7075))
                                            .child(if models_loaded {
                                                "モデル".to_string()
                                            } else {
                                                "モデル(バックエンド接続後に表示)".to_string()
                                            }),
                                    )
                                    .child(
                                        h_flex().flex_wrap().gap_1().children(model_buttons),
                                    )
                                    .child(div().flex_1())
                                    .child(
                                        div().text_xs().text_color(rgb(0x6b7075)).child(last_gen_line),
                                    )
                                    .child(
                                        div()
                                            .font_weight(FontWeight::BOLD)
                                            .text_sm()
                                            .child("生成履歴"),
                                    )
                                    .children(history_items),
                            ),
                    )
                    .child(
                        v_flex()
                            .id("log")
                            .h(px(130.))
                            .m_2()
                            .p_2()
                            .rounded_md()
                            .bg(rgb(0x16171a))
                            .overflow_y_scroll()
                            .text_xs()
                            .font_family("Consolas")
                            .text_color(rgb(0x8a9199))
                            .children(self.logs.iter().rev().take(12).rev().cloned()),
                    )
                    .child(
                        h_flex()
                            .px_3()
                            .py_1()
                            .border_t_1()
                            .text_xs()
                            .text_color(rgb(0x9aa0a6))
                            .gap_3()
                            .child(self.status_hint.clone())
                            .child(div().flex_1())
                            .child(format!(
                                "再生キュー: {} / 履歴: {}",
                                self.audio.as_ref().map(|a| a.pending_chunks()).unwrap_or(0),
                                self.history.len()
                            )),
                    ),
            )
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

fn main() {
    // --mock / --real で明示。無指定なら設定ファイルの mock を踏襲(既定 true)。
    let args: Vec<String> = std::env::args().collect();
    let explicit_mock = args.iter().any(|a| a == "--mock");
    let explicit_real = args.iter().any(|a| a == "--real");
    let mock = if explicit_mock {
        true
    } else if explicit_real {
        false
    } else {
        let root = backend::repo_root();
        settings::AppSettings::load(&root).mock.unwrap_or(true)
    };

    gpui_kit::application().run(move |cx: &mut App| {
        gpui_kit::init(cx);
        let bounds = Bounds::centered(None, size(px(1220.), px(780.)), cx);
        gpui_kit::open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            cx,
            |window, cx| cx.new(|cx| StttsApp::new(mock, window, cx)),
        )
        .expect("ウィンドウ生成に失敗");
        cx.activate(true);
    });
}
