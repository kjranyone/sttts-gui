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
const DEFAULT_VOICE_LABEL: &str = "既定の声(自動)";

/// マイク開始/停止の遷移中状態(楽観的UI)。連打によるデバイスの短時間反復 open/close を防ぐ。
#[derive(Debug, Clone, Copy, PartialEq)]
enum MicTransition {
    None,
    Starting,
    Stopping,
}

/// 会話の1発話(あなた=ASR確定文 / 音声=TTS生成)。時系列で1本のビューに並べる。
#[derive(Debug, Clone)]
struct ConversationEntry {
    kind: ConversationKind,
    text: String,
    /// ASR: 発話id(partial → final の置換用)。TTS: None
    utterance: Option<u64>,
    /// ASR partial(確定前の灰色表示)
    partial: bool,
    /// TTS: 合成時間
    gen_ms: Option<u64>,
    /// TTS: 生成wavパス(再生ボタン用)
    path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ConversationKind {
    User,
    Assistant,
}

pub struct StttsApp {
    mock: bool,
    backend: Option<backend::BackendHandle>,
    models: Vec<ModelInfo>,
    selected_model_id: String,
    tts_state: EngineState,
    asr_state: EngineState,
    mic_running: bool,
    /// マイク開始/停止要求を送ってから State が戻るまでの楽観的状態。
    /// 押下即時にフィードバック(ボタン無効化+モーダル)を出し、遷移中の再操作を防ぐ。
    mic_transition: MicTransition,

    speak_input: Entity<TextareaState>,
    caption_input: Entity<InputState>,
    seed_input: Entity<InputState>,
    random_seed: bool,
    auto_speak: bool,

    input_select: Entity<SelectState<Vec<String>>>,
    output_select: Entity<SelectState<Vec<String>>>,
    /// 声バンク(data/voices の wav ファイル)
    voice_select: Entity<SelectState<Vec<String>>>,
    voices: Vec<(String, PathBuf)>,
    selected_voice_name: Option<String>,
    input_devices: Vec<AudioDeviceInfo>,
    /// デバイス一覧到着後に render(windowあり)で Select へ反映するための保留領域
    pending_input_items: Option<Vec<String>>,
    selected_input_name: Option<String>,
    selected_output_name: Option<String>,
    saved_input_device: Option<String>,
    mic_level_db: f32,

    conversation: Vec<ConversationEntry>,
    /// 現在合成中/直前のチャンクテキスト(tts_chunk_start で設定、tts_audio で履歴へ)
    pending_chunk_text: Option<String>,
    logs: VecDeque<String>,
    last_gen_ms: Option<u64>,
    /// 発話終了 → 初音(先頭チャンクを再生キューに積むまで)の直近値と履歴(中央値表示用)
    last_e2e_ms: Option<u64>,
    e2e_history: VecDeque<u64>,
    last_asr_ms: Option<u64>,
    last_first_chunk_ms: Option<u64>,
    last_rtf: Option<f64>,
    speaking: bool,
    /// 現在の発話リクエストの合成進行(TtsChunkStart/TtsAudio で数える。受付時にリセット)
    synth_started: u32,
    synth_done: u32,
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

        // --- 声バンク(data/voices の wav を参照音声として選択できる)
        let voices = scan_voice_bank(&root);
        let saved_voice = saved
            .voice
            .clone()
            .filter(|n| voices.iter().any(|(vn, _)| vn == n));
        let mut voice_items = vec![DEFAULT_VOICE_LABEL.to_string()];
        voice_items.extend(voices.iter().map(|(n, _)| n.clone()));
        let voice_sel_ix = saved_voice
            .as_ref()
            .and_then(|n| voices.iter().position(|(vn, _)| vn == n))
            .map(|p| IndexPath::new(p + 1));
        let voice_select = cx.new(|cx| SelectState::new(voice_items, voice_sel_ix, window, cx));

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
            mic_transition: MicTransition::None,
            speak_input,
            caption_input,
            seed_input,
            random_seed,
            auto_speak,
            input_select,
            output_select,
            voice_select,
            voices,
            selected_voice_name: saved_voice.clone(),
            input_devices: Vec::new(),
            pending_input_items: None,
            selected_input_name: None,
            selected_output_name: saved_output_device,
            saved_input_device,
            mic_level_db: -100.0,
            conversation: Vec::new(),
            pending_chunk_text: None,
            logs: VecDeque::new(),
            last_gen_ms: None,
            last_e2e_ms: None,
            e2e_history: VecDeque::new(),
            last_asr_ms: None,
            last_first_chunk_ms: None,
            last_rtf: None,
            speaking: false,
            synth_started: 0,
            synth_done: 0,
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
        let weak_voice = cx.weak_entity();
        app.subscriptions.push(window.subscribe(
            &app.voice_select,
            cx,
            move |_, event, _window, cx| {
                if let (Some(app), SelectEvent::Confirm(Some(name))) =
                    (weak_voice.upgrade(), event)
                {
                    app.update(cx, |app, _cx| app.apply_voice(name.clone()));
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
            voice: Some(app.selected_voice_config()),
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
                app.mic_transition = MicTransition::None;
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

    /// 選択中の声に対応する voice 設定(バンク選択時は参照音声、既定は no_ref)。
    fn selected_voice_config(&self) -> VoiceConfig {
        match self
            .selected_voice_name
            .as_ref()
            .and_then(|n| self.voices.iter().find(|(vn, _)| vn == n))
        {
            Some((_, path)) => VoiceConfig {
                ref_wavs: Some(vec![path.to_string_lossy().into_owned()]),
                no_ref: Some(false),
                ..Default::default()
            },
            None => VoiceConfig {
                no_ref: Some(true),
                ..Default::default()
            },
        }
    }

    /// 声バンクの選択適用。参照音声が変わるとウォームアップもやり直される。
    fn apply_voice(&mut self, name: String) {
        self.selected_voice_name = (name != DEFAULT_VOICE_LABEL).then_some(name);
        self.send(GuiMessage::Configure {
            tts: None,
            asr: None,
            audio: None,
            voice: Some(self.selected_voice_config()),
            pipeline: None,
        });
        match &self.selected_voice_name {
            Some(n) => self.push_log(format!("声を切替: {n}(参照音声で合成します)")),
            None => self.push_log("声を既定に戻しました(キャプション/自動音質で合成)".into()),
        }
    }

    /// 声バンクフォルダ(data/voices)を Explorer で開き、内容を再読込する。
    fn open_voice_folder(&mut self, _ev: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        let dir = self.root.join("data").join("voices");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
        // フォルダを開いた後に追加された wav も選択できるように再スキャン
        self.voices = scan_voice_bank(&self.root);
        let mut items = vec![DEFAULT_VOICE_LABEL.to_string()];
        items.extend(self.voices.iter().map(|(n, _)| n.clone()));
        let selected = self
            .selected_voice_name
            .clone()
            .unwrap_or_else(|| DEFAULT_VOICE_LABEL.to_string());
        self.voice_select.update(cx, |s, cx| {
            s.set_items(items, window, cx);
            s.set_selected_value(&selected, window, cx);
        });
        self.push_log(format!(
            "声フォルダを開きました: {}(wav を置いたら再読込されます)",
            dir.display()
        ));
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
            voice: self.selected_voice_name.clone(),
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
                self.asr_state = asr.clone();
                self.mic_running = mic_running;
                // 停止中は State 応答で解除。開始中は ASR が ready/error になるまで
                // モーダルを維持し、ロードの様子を見せる(モーダル内に進捗も出す)。
                match self.mic_transition {
                    MicTransition::Stopping => self.mic_transition = MicTransition::None,
                    MicTransition::Starting => {
                        if matches!(asr.phase.as_str(), "ready" | "error") {
                            self.mic_transition = MicTransition::None;
                        }
                    }
                    MicTransition::None => {}
                }
            }
            BackendMessage::Log { level, message } => {
                self.push_log(format!("[{level}] {message}"));
            }
            BackendMessage::MicLevel { db, .. } => {
                self.mic_level_db = db;
            }
            BackendMessage::AsrPartial { utterance, text, .. } => {
                self.upsert_transcript(utterance, text, false);
            }
            BackendMessage::AsrFinal { utterance, text, asr_ms, .. } => {
                if asr_ms.is_some() {
                    self.last_asr_ms = asr_ms;
                }
                self.upsert_transcript(utterance, text, true);
            }
            BackendMessage::SpeakAccepted { request, origin, .. } => {
                self.last_accepted_request = self.last_accepted_request.max(request);
                self.speaking = true;
                self.synth_started = 0;
                self.synth_done = 0;
                self.push_log(format!("発話受付 request={request} origin={origin}"));
            }
            BackendMessage::TtsChunkStart { text, .. } => {
                self.synth_started += 1;
                self.pending_chunk_text = Some(text);
            }
            BackendMessage::TtsAudio {
                request,
                wav_base64,
                gen_ms,
                path,
                first_chunk,
                rtf,
                first_chunk_ms,
                e2e_ms,
                ..
            } => {
                let received = std::time::Instant::now();
                self.synth_done += 1;
                if request <= self.cancelled_upto {
                    self.pending_chunk_text = None;
                    self.push_log(format!("キャンセル済み request={request} の音声を破棄"));
                    return;
                }
                self.last_gen_ms = Some(gen_ms);
                let chunk_text = self.pending_chunk_text.take().unwrap_or_default();
                self.push_assistant_entry(chunk_text, gen_ms, path);
                if let Some(audio) = &self.audio {
                    if let Err(e) = audio.enqueue_wav_base64(&wav_base64) {
                        self.push_log(format!("音声キュー追加失敗: {e}"));
                    }
                }
                if first_chunk {
                    // backend 側(話し終わり→送出)+ GUI 側(受信→再生キュー投入)
                    let local_ms = received.elapsed().as_millis() as u64;
                    self.last_first_chunk_ms = first_chunk_ms;
                    self.last_rtf = rtf;
                    if let Some(e2e) = e2e_ms {
                        let total = e2e + local_ms;
                        self.last_e2e_ms = Some(total);
                        self.e2e_history.push_back(total);
                        if self.e2e_history.len() > 20 {
                            self.e2e_history.pop_front();
                        }
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
                if scope == "asr" {
                    // マイク/ASR 起動・停止失敗: 遷移中のまま固めない
                    self.mic_transition = MicTransition::None;
                }
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

    /// ASR の partial/final を会話ビューへ反映。final は同発話の partial を確定化する。
    fn upsert_transcript(&mut self, utterance: u64, text: String, is_final: bool) {
        if let Some(entry) = self
            .conversation
            .iter_mut()
            .rev()
            .find(|e| e.utterance == Some(utterance))
        {
            entry.text = text;
            entry.partial = !is_final;
            return;
        }
        self.conversation.push(ConversationEntry {
            kind: ConversationKind::User,
            text,
            utterance: Some(utterance),
            partial: !is_final,
            gen_ms: None,
            path: None,
        });
        self.trim_conversation();
    }

    /// TTS 生成を会話ビューへ追加(音声の発話として時系列に並ぶ)。
    fn push_assistant_entry(&mut self, text: String, gen_ms: u64, path: Option<String>) {
        self.conversation.push(ConversationEntry {
            kind: ConversationKind::Assistant,
            text,
            utterance: None,
            partial: false,
            gen_ms: Some(gen_ms),
            path,
        });
        self.trim_conversation();
    }

    fn trim_conversation(&mut self) {
        while self.conversation.len() > 500 {
            self.conversation.remove(0);
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

    fn replay_conversation(&mut self, index: usize, _ev: &ClickEvent, _w: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.conversation.get(index) else {
            return;
        };
        let Some(path) = &entry.path else {
            self.push_log("この発話には音声ファイルがありません".into());
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

    fn toggle_mic(&mut self, _ev: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        // 遷移中(開始中/停止中)の再操作は無視: マイクの短時間反復 open/close は
        // USB オーディオドライバを落とすことがある(2026-10-09 に BugCheck 0xD1 を2度発生)。
        if self.mic_transition != MicTransition::None {
            return;
        }
        let (transition, msg, timeout_log) = if self.mic_running {
            (MicTransition::Stopping, GuiMessage::StopSession, "マイク停止がタイムアウトしました")
        } else {
            (MicTransition::Starting, GuiMessage::StartSession, "マイク開始がタイムアウトしました(もう一度お試しください)")
        };
        self.mic_transition = transition;
        self.send(msg);
        let started_at = std::time::Instant::now();
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(std::time::Duration::from_secs(15)).await;
            let _ = this.update(cx, |app, cx| {
                if app.mic_transition != MicTransition::None
                    && started_at.elapsed() >= std::time::Duration::from_secs(15)
                {
                    app.mic_transition = MicTransition::None;
                    app.push_log(timeout_log.into());
                    cx.notify();
                }
            });
        })
        .detach();
    }

    /// 入力デバイス選択の適用。マイク実行中は停止して再開はユーザーに任せる。
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
            // 停止→即再開はbackendの再開クールダウンに拒否されるうえ、デバイスの
            // 短時間反復 open/close はドライバクラッシュの原因。停止のみ送り、
            // 再開はユーザー操作(マイク開始)に任せる。
            self.mic_transition = MicTransition::Stopping;
            self.send(GuiMessage::StopSession);
            self.push_log(
                "入力デバイスを変更したためマイクを停止しました。新しいデバイスで使うには「マイク開始」を押してください"
                    .into(),
            );
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

    /// ヘッダの主 KPI: 話し終えてから音声が鳴り始めるまで(=応答速度)。直近値と直近20回の中央値。
    fn latency_label(&self) -> String {
        match self.last_e2e_ms {
            Some(ms) => {
                let mut v: Vec<u64> = self.e2e_history.iter().copied().collect();
                v.sort_unstable();
                let median = v.get(v.len() / 2).copied().unwrap_or(ms);
                format!("応答 {ms}ms(中央値 {median}ms)")
            }
            None => "応答 —".into(),
        }
    }

    fn latency_detail(&self) -> String {
        let fmt = |v: Option<u64>| v.map(|x| format!("{x}ms")).unwrap_or_else(|| "—".into());
        let rtf = self.last_rtf.map(|r| format!("{r:.2}")).unwrap_or_else(|| "—".into());
        format!("文字 {} / 音声 {} / 速度 {}",
            fmt(self.last_asr_ms),
            fmt(self.last_first_chunk_ms),
            rtf)
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

    /// ヘッダ表示用の短ラベル(固定語のみ。detail のような長い文字列はログで確認する)
    fn phase_label(state: &EngineState) -> &'static str {
        match state.phase.as_str() {
            "ready" => "準備完了",
            "loading" => "ロード中…",
            "error" => "エラー",
            _ => "未ロード",
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

        // 発話パイプラインの状態(合成 → 再生キュー → 再生)を1行に集約する
        let play_queue = self.audio.as_ref().map(|a| a.pending_chunks()).unwrap_or(0);
        let (pipeline_status, pipeline_color) = if self.speaking {
            let synth = if self.synth_started > 0 {
                format!("合成 {}/{}", self.synth_done, self.synth_started)
            } else {
                "合成待ち…".to_string()
            };
            if play_queue > 0 {
                (format!("♪ 再生中(キュー {play_queue})・{synth}"), rgb(0xffb1cf))
            } else if self.synth_done < self.synth_started || self.synth_started == 0 {
                (format!("◈ {synth}"), rgb(0xf0c987))
            } else {
                ("✓ 合成完了・再生待ち".to_string(), rgb(0x8ef0c0))
            }
        } else if play_queue > 0 {
            (format!("♪ 再生中(キュー {play_queue})"), rgb(0xffb1cf))
        } else {
            ("― 待機中".to_string(), rgb(0x8d86ad))
        };

        // ---- 会話ビュー(チャット泡)
        let conversation_items: Vec<_> = self
            .conversation
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let is_user = e.kind == ConversationKind::User;
                let bubble = if is_user {
                    // あなた: 桜の泡(右)
                    v_flex()
                        .max_w(px(420.))
                        .px_3()
                        .py_2()
                        .rounded_lg()
                        .bg(linear_gradient(120., linear_color_stop(rgba(0xffd6e7ff), 0.), linear_color_stop(rgba(0xffb1cfff), 1.)))
                        .shadow_sm()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::BOLD)
                                .text_color(rgb(0x9c5f80))
                                .child(if e.partial { "あなた …" } else { "あなた" }),
                        )
                        .child(div().text_sm().text_color(rgb(0x53324a)).child(e.text.clone()))
                } else {
                    // 音声: 藤の泡(左)+ 再生
                    h_flex()
                        .max_w(px(460.))
                        .items_end()
                        .gap_2()
                        .child(
                            v_flex()
                                .px_3()
                                .py_2()
                                .rounded_lg()
                                .bg(linear_gradient(120., linear_color_stop(rgba(0x6d5bd0ff), 0.), linear_color_stop(rgba(0x8d6fe8ff), 1.)))
                                .shadow_sm()
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(FontWeight::BOLD)
                                        .text_color(rgb(0xd9d0ff))
                                        .child("♪ 音声"),
                                )
                                .child(
                                    div().text_sm().text_color(rgb(0xf4f1ff)).child(e.text.clone()),
                                ),
                        )
                        .child(
                            v_flex().gap_1().child(
                                div().text_xs().text_color(rgb(0x8d86ad)).child(
                                    e.gen_ms.map(|ms| format!("{ms}ms")).unwrap_or_default(),
                                ),
                            ),
                        )
                        .when(e.path.is_some(), |row| {
                            row.child(
                                Button::new(SharedString::from(format!("replay-{i}")))
                                    .label("▶")
                                    .compact()
                                    .on_click(cx.listener(move |this, ev, w, cx| {
                                        this.replay_conversation(i, ev, w, cx)
                                    })),
                            )
                        })
                };
                let row = h_flex().w_full().py_1_5();
                let row = if is_user {
                    // 右寄せ(残り幅を左に置く)
                    row.child(div().flex_1()).child(bubble)
                } else {
                    row.child(bubble).child(div().flex_1())
                };
                row.into_any_element()
            })
            .collect();
        let empty_hint = self.conversation.is_empty().then(|| {
            v_flex()
                .h_full()
                .w_full()
                .items_center()
                .justify_center()
                .gap_2()
                .child(div().text_xl().text_color(rgb(0xffb1cf)).child("♪"))
                .child(
                    div()
                        .text_sm()
                        .text_color(rgb(0x8d86ad))
                        .child("「マイク開始」で話しかけると、ここに会話が並びます"),
                )
                .into_any_element()
        });

        // ---- 右パネル: モデルボタン
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

        // Switch の on_click は &mut App を受けるため弱参照経由で Self を更新する
        let weak_auto_speak = cx.weak_entity();
        let weak_random_seed = cx.weak_entity();

        // ---- ヘッダ
        let (mic_dot, mic_label) = if self.mic_running {
            (rgb(0x8ef0c0), "listening")
        } else {
            (rgb(0x8d86ad), "idle")
        };
        let header = h_flex()
            .gap_3()
            .items_center()
            .px_4()
            .py_2()
            .child(
                div()
                    .text_base()
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0xfff2fa))
                    .child("sttts"),
            )
            .child(div().text_sm().text_color(rgb(0xc9a8dd)).child("音声対話"))
            .child(
                h_flex()
                    .gap_1()
                    .items_center()
                    .px_2()
                    .py_1()
                    .rounded_full()
                    .bg(rgba(0xffffff10))
                    .child(
                        div().text_xs().text_color(rgb(0xb9b1d6)).child(format!(
                            "TTS {} / ASR {}",
                            Self::phase_label(&self.tts_state),
                            Self::phase_label(&self.asr_state)
                        )),
                    ),
            )
            .child(
                h_flex()
                    .gap_1()
                    .items_center()
                    .px_2()
                    .py_1()
                    .rounded_full()
                    .bg(rgba(0xffffff10))
                    .child(div().size(px(8.)).rounded_full().bg(mic_dot))
                    .child(div().text_xs().text_color(rgb(0xb9b1d6)).child(mic_label)),
            )
            .child(div().flex_1())
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0xffb1cf))
                    .child(self.latency_label()),
            )
            .child(div().text_xs().text_color(rgb(0x8d86ad)).child(self.latency_detail()))
            .child(
                Button::new("mic")
                    .label(match self.mic_transition {
                        MicTransition::Starting => "マイク開始中…",
                        MicTransition::Stopping => "マイク停止中…",
                        MicTransition::None if self.mic_running => "マイク停止",
                        MicTransition::None => "マイク開始",
                    })
                    .disabled(self.mic_transition != MicTransition::None)
                    .on_click(cx.listener(Self::toggle_mic)),
            )
            .child(
                Button::new("cancel")
                    .label("発話を中止")
                    .on_click(cx.listener(Self::cancel_speak)),
            )
            .child(Button::new("quit").label("終了").on_click(cx.listener(Self::quit)));

        // ---- レベルメーター(緑→桜グラデ)
        let level_meter = h_flex()
            .gap_2()
            .items_center()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x8d86ad))
                    .w(px(28.))
                    .child("音量"),
            )
            .child(
                div()
                    .flex_1()
                    .h(px(12.))
                    .rounded_full()
                    .bg(rgba(0x00000075))
                    .border_1()
                    .border_color(rgba(0xffffff26))
                    .overflow_hidden()
                    .child(
                        div()
                            .h_full()
                            .w(relative(level_frac))
                            .rounded_full()
                            .bg(linear_gradient(90., linear_color_stop(rgba(0x7ef0b2ff), 0.), linear_color_stop(rgba(0xff8fb8ff), 1.))),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .font_family("Consolas")
                    .text_color(rgb(0xb9b1d6))
                    .child(if self.mic_running {
                        format!("{:+.0} dB", self.mic_level_db.max(-99.0))
                    } else {
                        "—".to_string()
                    }),
            );

        // ---- レイアウト
        // 通常UI(の全体)を relative ラッパーの中に置き、遷移中は同じラッパー内の
        // absolute 兄弟として全画面オーバーレイを重ねる(CSS と同じ意味論)。
        let app_ui = div()
            .size_full()
            .bg(linear_gradient(160., linear_color_stop(rgba(0x2b1e4fff), 0.), linear_color_stop(rgba(0x0f1633ff), 1.)))
            .text_color(rgb(0xf4f1ff))
            .child(
                v_flex()
                    .size_full()
                    .child(header)
                    .child(
                        h_flex()
                            .flex_1()
                            .gap_3()
                            .p_3()
                            .overflow_hidden()
                            // 左: 会話(ガラス風カード)
                            .child(
                                v_flex()
                                    .id("conversation")
                                    .flex_1()
                                    .h_full()
                                    .p_4()
                                    .rounded_lg()
                                    .bg(rgba(0xffffff08))
                                    .border_1()
                                    .border_color(rgba(0xffffff1c))
                                    .shadow_sm()
                                    .overflow_y_scroll()
                                    .text_sm()
                                    .children(empty_hint.into_iter().chain(conversation_items)),
                            )
                            // 右: 発話パネル(ガラス風カード)
                            .child(
                                v_flex()
                                    .id("speak-panel")
                                    .w(px(430.))
                                    .h_full()
                                    .p_4()
                                    .rounded_lg()
                                    .bg(rgba(0xffffff08))
                                    .border_1()
                                    .border_color(rgba(0xffffff1c))
                                    .shadow_sm()
                                    .overflow_y_scroll()
                                    .gap_2()
                                    .child(
                                        div()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(0xfff2fa))
                                            .child("テキストから発話"),
                                    )
                                    .child(Textarea::new(&self.speak_input).text_sm())
                                    .child(
                                        v_flex()
                                            .gap_1()
                                            .child(
                                                h_flex()
                                                    .gap_2()
                                                    .items_center()
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(rgb(0x8d86ad))
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
                                                            .text_color(rgb(0x8d86ad))
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
                                                // 声バンク(参照音声による声質指定)
                                                h_flex()
                                                    .gap_2()
                                                    .items_center()
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(rgb(0x8d86ad))
                                                            .w(px(28.))
                                                            .child("声"),
                                                    )
                                                    .child(
                                                        div().flex_1().child(
                                                            Select::new(&self.voice_select)
                                                                .placeholder(DEFAULT_VOICE_LABEL)
                                                                .text_sm(),
                                                        ),
                                                    )
                                                    .child(
                                                        Button::new("open-voices")
                                                            .label("📁")
                                                            .compact()
                                                            .on_click(cx.listener(Self::open_voice_folder)),
                                                    ),
                                            )
                                            .child(level_meter),
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
                                    // 発話ボタン: 桜→藤グラデ
                                    .child(
                                        div()
                                            .id("speak-button")
                                            .rounded_md()
                                            .py_2()
                                            .w_full()
                                            .flex()
                                            .justify_center()
                                            .text_sm()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(0xffffff))
                                            .bg(linear_gradient(120., linear_color_stop(rgba(0xff8fb8ff), 0.), linear_color_stop(rgba(0x8d6fe8ff), 1.)))
                                            .shadow_md()
                                            .child("発話")
                                            .on_click(cx.listener(Self::speak_from_input)),
                                    )
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_family("Consolas")
                                            .text_color(pipeline_color)
                                            .child(pipeline_status),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x8d86ad))
                                            .child(if models_loaded {
                                                "モデル".to_string()
                                            } else {
                                                "モデル(バックエンド接続後に表示)".to_string()
                                            }),
                                    )
                                    .child(h_flex().flex_wrap().gap_1().children(model_buttons))
                                    .child(div().flex_1()),
                            ),
                    )
                    // ログ(黒ガラス)
                    .child(
                        v_flex()
                            .id("log")
                            .h(px(120.))
                            .mx_3()
                            .mb_1()
                            .p_2()
                            .rounded_lg()
                            .bg(rgba(0x00000055))
                            .border_1()
                            .border_color(rgba(0xffffff14))
                            .overflow_y_scroll()
                            .text_xs()
                            .font_family("Consolas")
                            .text_color(rgb(0x9d96bd))
                            .children(self.logs.iter().rev().take(12).rev().cloned()),
                    )
                    .child(
                        h_flex()
                            .px_4()
                            .py_2()
                            .text_xs()
                            .text_color(rgb(0x8d86ad))
                            .gap_3()
                            .child(self.status_hint.clone())
                            .child(div().flex_1())
                            .child(format!("会話: {} 件", self.conversation.len())),
                    ),
            );

        // 遷移中は通常UIの上に全画面の半透明スクリーン+中央カードを absolute で重ねる
        let overlay = (self.mic_transition != MicTransition::None).then(|| {
            let (title, sub) = match self.mic_transition {
                MicTransition::Stopping => (
                    "マイクを停止中…",
                    "デバイスの解放を待っています(直後の再開はしばらく受け付けません)".to_string(),
                ),
                _ => (
                    "マイクを準備中…",
                    format!(
                        "TTS {} / ASR {}",
                        Self::phase_label(&self.tts_state),
                        Self::phase_label(&self.asr_state)
                    ),
                ),
            };
            div()
                .id("mic-transition-overlay")
                .occlude()
                .absolute()
                .left_0()
                .top_0()
                .w_full()
                .h_full()
                .bg(rgba(0x0f0a1e99))
                .flex()
                .items_center()
                .justify_center()
                .with_animation(
                    "mic-transition-fade",
                    Animation::new(std::time::Duration::from_millis(180)).with_easing(ease_in_out),
                    |el, delta| el.opacity(0.5 + 0.5 * delta),
                )
                .child(
                    v_flex()
                        .id("mic-transition-card")
                        .gap_2()
                        .px_10()
                        .py_8()
                        .rounded_lg()
                        .bg(rgba(0x2b2d31ee))
                        .border_1()
                        .border_color(rgba(0xffffff2b))
                        .shadow_lg()
                        .items_center()
                        .child(
                            div()
                                .text_xl()
                                .font_weight(FontWeight::BOLD)
                                .text_color(rgb(0xfff2fa))
                                .child(title),
                        )
                        .child(div().text_sm().text_color(rgb(0xb9b1d6)).child(sub)),
                )
        });

        div()
            .relative()
            .size_full()
            .child(app_ui)
            .children(overlay)
            .into_any_element()
    }
}


/// data/voices の wav を声バンクとして読み込む(ファイル名=話者名)。
fn scan_voice_bank(root: &std::path::Path) -> Vec<(String, PathBuf)> {
    let dir = root.join("data").join("voices");
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            let is_wav = p
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("wav"))
                .unwrap_or(false);
            if is_wav {
                if let Some(name) = p.file_stem().and_then(|s| s.to_str()) {
                    if !name.is_empty() {
                        out.push((name.to_string(), p));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
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
