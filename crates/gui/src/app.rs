//! アプリの状態とふるまい(描画は子モジュール)。
//!
//! # 責務と画面構成
//! sttts は「話した内容を、選んだ声で届ける」ための中継器。主画面は次の3つを分けて見せる
//! (docs/acting-reconstruction-design.md「UI への反映」):
//!
//! - **何を話したか** — ストリーム(中央)。1ターン = 入力1件と、それを届けた声の対応
//! - **どう伝えるか** — 右レールの「音声キュー」(自動再生 ON/OFF、テンポと間の再現)
//! - **どの声で届けるか** — 右レールの「声」
//!
//! 環境で一度決まる設定(入出力デバイス、TTS モデル、seed)は「詳細設定」シート
//! (既定は閉)に置き、主画面に出さない(AGENTS.md「設計の前提」)。

mod chrome;
mod help;
mod kit;
mod rail;
mod sheet;
mod stream;
mod title_bar;

use std::collections::VecDeque;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use gpui_kit::component::input::{InputEvent, InputState, TextareaState};
use gpui_kit::component::select::{SelectEvent, SelectState};
use gpui_kit::component::IndexPath;
use gpui_kit::*;
use sttts_protocol::{
    AnyMessage, AudioConfig, AudioDeviceInfo, BackendMessage, EngineState, GuiMessage, ModelInfo,
    PipelineConfig, TtsConfig, VoiceConfig,
};

use crate::turns::{Turns, tag_for};
use crate::{audio, backend, secret, settings, sysmon};

pub(crate) const DEFAULT_INPUT_LABEL: &str = "既定の入力デバイス";
pub(crate) const DEFAULT_OUTPUT_LABEL: &str = "既定の出力デバイス";
pub(crate) const DEFAULT_VOICE_LABEL: &str = "既定の声";

/// ASR プロバイダ選択(表示名, asr.engine 値)。ローカルとクラウドを選べる。
pub(crate) const ASR_PROVIDERS: &[(&str, &str)] = &[
    ("クラウド(Gemini)", "gemini"),
    ("ローカル(Nemotron)", "nemotron"),
    ("ローカル(kotoba)", "kotoba"),
];

/// Gemini API キーの発行ページ(Google AI Studio)
pub(crate) const GEMINI_KEY_URL: &str = "https://aistudio.google.com/apikey";

/// マイク開始/停止要求の応答待ちの上限。超えたら遷移中表示を解除する。
const MIC_TRANSITION_TIMEOUT: Duration = Duration::from_secs(15);

/// マイク開始/停止の遷移中状態(楽観的UI)。連打によるデバイスの短時間反復 open/close を防ぐ。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MicTransition {
    None,
    Starting,
    Stopping,
}

pub struct StttsApp {
    mock: bool,
    backend: Option<backend::BackendHandle>,
    /// hello を受け取った(backend と会話できる)
    connected: bool,
    models: Vec<ModelInfo>,
    selected_model_id: String,
    tts_state: EngineState,
    asr_state: EngineState,
    mic_running: bool,
    /// マイク開始/停止要求を送ってから State が戻るまでの楽観的状態。
    /// 押下即時にフィードバック(ボタン無効化+オーバーレイ)を出し、遷移中の再操作を防ぐ。
    mic_transition: MicTransition,
    mic_level_db: f32,

    // ---- ストリーム(何を話したか → 届けた声)
    turns: Turns,
    stream_scroll: gpui::ScrollHandle,
    composer: Entity<TextareaState>,

    // ---- 音声キュー
    /// 確定文を自動で音声キューへ流す(false = カードで止めて手動で発話)
    auto_speak: bool,
    performance_enabled: bool,

    // ---- 声
    voices: Vec<(String, PathBuf)>,
    /// ファイル選択ダイアログの結果。Window が要るので render で取り込む。
    pending_voice_import: Option<Vec<PathBuf>>,
    /// PC リソース(RAM / GPU 専用メモリ)の最新サンプル
    sys: Option<sysmon::SysSample>,
    backend_pid: Arc<AtomicU32>,
    /// エンジンが loading になった時刻(経過秒の表示用)
    tts_loading_since: Option<Instant>,
    asr_loading_since: Option<Instant>,
    selected_voice_name: Option<String>,
    voice_select: Entity<SelectState<Vec<String>>>,
    /// 話し方の指示(Irodori の caption)
    caption_input: Entity<InputState>,

    // ---- 認識
    asr_select: Entity<SelectState<Vec<String>>>,
    selected_asr_engine: String,
    /// Gemini API キー入力欄(伏せ字)。確定は Enter / フォーカスアウト
    gemini_key_input: Entity<InputState>,
    /// backend へ送ったキー(空 = GUI では未設定)
    gemini_api_key: String,
    /// 保存用の暗号化済みキー(secret::protect の出力)
    gemini_key_protected: Option<String>,

    // ---- 詳細設定(環境で一度決まるもの)
    settings_open: bool,
    /// 「?」から開いている解説
    help_topic: Option<help::HelpTopic>,
    input_select: Entity<SelectState<Vec<String>>>,
    output_select: Entity<SelectState<Vec<String>>>,
    input_devices: Vec<AudioDeviceInfo>,
    /// デバイス一覧到着後に render(windowあり)で Select へ反映するための保留領域
    pending_input_items: Option<Vec<String>>,
    selected_input_name: Option<String>,
    selected_output_name: Option<String>,
    saved_input_device: Option<String>,
    seed_input: Entity<InputState>,
    random_seed: bool,

    // ---- 診断
    log_open: bool,
    log_scroll: gpui::ScrollHandle,
    logs: VecDeque<String>,
    /// ログ欄を閉じている間に増えたエラー行(ステータスバーのバッジ)
    unread_errors: usize,
    /// data/gui.log(起動ごとに作り直す。ログ欄は選択・コピーできないため、エージェント/人間が事後に読む)
    log_file: Option<std::fs::File>,
    /// 発話終了 → 初音(先頭チャンクを再生キューに積むまで)の直近値と履歴(中央値表示用)
    last_e2e_ms: Option<u64>,
    e2e_history: VecDeque<u64>,
    last_asr_ms: Option<u64>,
    last_first_chunk_ms: Option<u64>,
    last_rtf: Option<f64>,

    /// 受付済み最大 request id(speak_accepted)
    last_accepted_request: u64,
    /// キャンセル時点の last_accepted_request。これ以下の request の音声は捨てる
    /// (キャンセル前に送出済みでパイプ上にあった tts_audio を鳴らさないため)
    cancelled_upto: u64,
    audio: Option<audio::AudioOut>,
    /// Select 等のイベント購読(gpui は Subscription を drop すると購読解除になる)
    subscriptions: Vec<gpui::Subscription>,
    status_hint: String,
    root: PathBuf,
}

impl StttsApp {
    pub fn new(mock: bool, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let root = backend::repo_root();
        let saved = settings::AppSettings::load(&root);
        let auto_speak = saved.auto_speak.unwrap_or(true);
        let performance_enabled = saved.performance_enabled.unwrap_or(true);
        let random_seed = saved.random_seed.unwrap_or(true);

        let composer = cx.new(|cx| {
            TextareaState::new(window, cx)
                .placeholder("文字で話す")
                .auto_grow(1, 6)
        });
        let caption_input = cx.new(|cx| {
            let mut state = InputState::new(window, cx);
            if let Some(caption) = &saved.caption {
                state.set_value(caption.as_str(), window, cx);
            }
            state
        });
        let seed_input = cx.new(|cx| InputState::new(window, cx).placeholder("seed"));

        // --- 入出力デバイス選択
        let saved_input_device = saved.input_device.clone();
        let output_devices = audio::list_output_devices();
        let mut output_items = vec![DEFAULT_OUTPUT_LABEL.to_string()];
        output_items.extend(output_devices.iter().cloned());
        let saved_output_device = saved
            .output_device
            .clone()
            .filter(|n| output_devices.iter().any(|d| d == n));
        let output_sel_ix = saved_output_device
            .as_ref()
            .and_then(|n| output_devices.iter().position(|d| d == n))
            .map(|p| IndexPath::new(p + 1))
            .or(Some(IndexPath::new(0)));
        let output_select = cx.new(|cx| SelectState::new(output_items, output_sel_ix, window, cx));
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
            .map(|p| IndexPath::new(p + 1))
            .or(Some(IndexPath::new(0)));
        let voice_select = cx.new(|cx| SelectState::new(voice_items, voice_sel_ix, window, cx));

        // --- ASR プロバイダ選択(ローカル/クラウド)
        let asr_items: Vec<String> = ASR_PROVIDERS.iter().map(|(l, _)| l.to_string()).collect();
        let saved_asr = saved.asr_provider.clone().unwrap_or_else(|| "nemotron".into());
        let asr_sel_ix = ASR_PROVIDERS
            .iter()
            .position(|(_, e)| *e == saved_asr)
            .map(IndexPath::new);
        let asr_select = cx.new(|cx| SelectState::new(asr_items, asr_sel_ix, window, cx));

        // --- Gemini API キー(data/config.json に DPAPI 暗号化で保存)
        let saved_gemini_key = saved.gemini_api_key_protected.as_deref().map(secret::unprotect);
        let gemini_key_unreadable = matches!(saved_gemini_key, Some(None));
        let gemini_api_key = saved_gemini_key.flatten().unwrap_or_default();
        let gemini_key_protected = (!gemini_api_key.is_empty())
            .then(|| saved.gemini_api_key_protected.clone())
            .flatten();
        let gemini_key_input = cx.new(|cx| {
            let mut state = InputState::new(window, cx)
                .masked(true)
                .placeholder("AIza…");
            state.set_value(gemini_api_key.as_str(), window, cx);
            state
        });

        let (audio, audio_error) = match audio::AudioOut::open(saved_output_device.as_deref())
            .or_else(|_| audio::AudioOut::open(None))
        {
            Ok(a) => (Some(a), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };

        let selected_model_id = saved.tts_model.unwrap_or_else(|| "v4.1-small-mf".into());

        let mut app = Self {
            mock,
            backend: None,
            connected: false,
            models: Vec::new(),
            selected_model_id,
            tts_state: idle_state(),
            asr_state: idle_state(),
            mic_running: false,
            mic_transition: MicTransition::None,
            mic_level_db: -100.0,
            turns: Turns::default(),
            stream_scroll: gpui::ScrollHandle::new(),
            composer,
            auto_speak,
            performance_enabled,
            voices,
            pending_voice_import: None,
            sys: None,
            backend_pid: Arc::new(AtomicU32::new(0)),
            tts_loading_since: None,
            asr_loading_since: None,
            selected_voice_name: saved_voice,
            voice_select,
            caption_input,
            asr_select,
            selected_asr_engine: saved_asr,
            gemini_key_input,
            gemini_api_key,
            gemini_key_protected,
            settings_open: false,
            help_topic: None,
            input_select,
            output_select,
            input_devices: Vec::new(),
            pending_input_items: None,
            selected_input_name: None,
            selected_output_name: saved_output_device,
            saved_input_device,
            seed_input,
            random_seed,
            log_open: false,
            log_scroll: gpui::ScrollHandle::new(),
            logs: VecDeque::new(),
            unread_errors: 0,
            log_file: open_log_file(&root),
            last_e2e_ms: None,
            e2e_history: VecDeque::new(),
            last_asr_ms: None,
            last_first_chunk_ms: None,
            last_rtf: None,
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
        if gemini_key_unreadable {
            app.push_log(
                "保存済みの Gemini API キーを復号できませんでした(別のPC/ユーザーで保存されたもの)。再入力してください"
                    .into(),
            );
        }
        app.start_backend(cx);
        app.start_sysmon(cx);

        // OS タイトルバーを持たないため、終了はウィンドウの × に一本化する。
        // 閉じる前に設定保存とバックエンド停止(マイク解放を含む)を済ませる。
        let weak_close = cx.weak_entity();
        window.on_window_should_close(cx, move |_window, cx| {
            let _ = weak_close.update(cx, |this, cx| this.shutdown(cx));
            true
        });

        app.subscribe_inputs(window, cx);

        // 初期設定を backend へ反映(engine とキーを同じパッチで送り、プリロードを1回で済ませる)
        app.send(GuiMessage::Configure {
            tts: Some(TtsConfig {
                model: Some(app.selected_model_id.clone()),
                ..Default::default()
            }),
            asr: Some(sttts_protocol::AsrConfig {
                engine: Some(app.selected_asr_engine.clone()),
                gemini_api_key: (!app.gemini_api_key.is_empty()).then(|| app.gemini_api_key.clone()),
                ..Default::default()
            }),
            audio: None,
            voice: Some(app.selected_voice_config()),
            pipeline: Some(PipelineConfig {
                auto_speak: Some(app.auto_speak),
                performance_enabled: Some(app.performance_enabled),
                ..Default::default()
            }),
        });
        app
    }

    /// Select / 入力欄のイベント購読(選択は window スコープで届く)。
    fn subscribe_inputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        fn on_confirm(
            app: WeakEntity<StttsApp>,
            f: impl Fn(&mut StttsApp, String, &mut Window, &mut Context<StttsApp>) + 'static,
        ) -> impl FnMut(Entity<SelectState<Vec<String>>>, &SelectEvent<Vec<String>>, &mut Window, &mut App)
        + 'static {
            move |_, event, window, cx| {
                if let SelectEvent::Confirm(Some(name)) = event {
                    let _ = app.update(cx, |app, cx| f(app, name.clone(), window, cx));
                }
            }
        }

        let weak = cx.weak_entity();
        let subs = vec![
            window.subscribe(&self.input_select, cx, on_confirm(weak.clone(), |a, n, w, cx| a.apply_input_device(n, w, cx))),
            window.subscribe(&self.output_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_output_device(n, cx))),
            window.subscribe(&self.voice_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_voice(n, cx))),
            window.subscribe(&self.asr_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_asr_provider(n, cx))),
            window.subscribe(&self.gemini_key_input, cx, {
                let weak = weak.clone();
                move |_, event: &InputEvent, _window, cx| {
                    if matches!(event, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                        let _ = weak.update(cx, |app, cx| app.apply_gemini_key(cx));
                    }
                }
            }),
            window.subscribe(&self.caption_input, cx, {
                let weak = weak.clone();
                move |_, event: &InputEvent, _window, cx| {
                    if matches!(event, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                        let _ = weak.update(cx, |app, cx| app.persist_settings(cx));
                    }
                }
            }),
            window.subscribe(&self.composer, cx, {
                let weak = weak.clone();
                move |_, event: &InputEvent, window, cx| {
                    if let InputEvent::PressEnter { secondary: true, .. } = event {
                        let _ = weak.update(cx, |app, cx| app.speak_from_composer(window, cx));
                    }
                }
            }),
        ];
        self.subscriptions.extend(subs);
    }

    // ---------- backend ----------

    fn start_backend(&mut self, cx: &mut Context<Self>) {
        // 待ち時間・読み込み時間の表示を毎秒更新する(待ちが無いときは再描画しない)
        cx.spawn(async move |this, cx| loop {
            cx.background_executor().timer(Duration::from_secs(1)).await;
            if this
                .update(cx, |app, cx| {
                    let (queued, speaking) = app.turns.queue_counts();
                    if queued + speaking > 0 || app.tts_loading_since.is_some() || app.asr_loading_since.is_some() {
                        cx.notify();
                    }
                })
                .is_err()
            {
                break;
            }
        })
        .detach();

        let (tx_events, rx_events) = async_channel::unbounded::<AnyMessage>();
        self.status_hint = "バックエンド起動中…".into();
        self.launch_backend(tx_events, rx_events, cx);
    }

    /// RAM / GPU 専用メモリの監視(読み取り専用。デバイスには触れない)。
    fn start_sysmon(&mut self, cx: &mut Context<Self>) {
        let (tx, rx) = async_channel::unbounded::<sysmon::SysSample>();
        sysmon::spawn(tx, self.backend_pid.clone());
        cx.spawn(async move |this, cx| {
            while let Ok(sample) = rx.recv().await {
                if this
                    .update(cx, |app, cx| {
                        app.sys = Some(sample);
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
    }

    fn launch_backend(
        &mut self,
        tx_events: async_channel::Sender<AnyMessage>,
        rx_events: async_channel::Receiver<AnyMessage>,
        cx: &mut Context<Self>,
    ) {
        let output_dir: PathBuf = self.root.join("output");
        let _ = std::fs::create_dir_all(&output_dir);
        let handle = backend::BackendHandle::start(self.mock, &self.root, &output_dir, tx_events);
        self.backend_pid.store(handle.pid(), Ordering::Relaxed);
        self.backend = Some(handle);
        self.status_hint = "バックエンド接続待ち…".into();

        // backend → UI の取り込みループ
        cx.spawn(async move |this, cx| {
            while let Ok(msg) = rx_events.recv().await {
                if this.update(cx, |app, cx| app.on_backend_message(msg, cx)).is_err() {
                    return;
                }
            }
            // チャネル閉鎖 = バックエンド終了
            let _ = this.update(cx, |app, cx| {
                app.push_log("バックエンドとの接続が切れました".into());
                app.status_hint = "バックエンド停止".into();
                app.connected = false;
                app.mic_running = false;
                app.mic_transition = MicTransition::None;
                app.turns.backend_restarted();
                cx.notify();
            });
        })
        .detach();
    }

    fn send(&mut self, msg: GuiMessage) {
        if let Some(b) = &self.backend {
            b.send(&msg);
        } else {
            self.push_log("バックエンド未接続".into());
        }
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
                self.connected = true;
                self.mock = mock;
                self.models = models;
                // 保存されていたモデルが今のエンジンで扱えない(以前の版で選んだ大型モデル等)ときは既定へ戻す
                if !self.models.iter().any(|m| m.id == self.selected_model_id)
                    && let Some(first) = self.models.first().cloned()
                {
                    self.push_log(format!(
                        "音声合成モデル {} は使えないため {} に切り替えました",
                        self.selected_model_id, first.label
                    ));
                    self.selected_model_id.clone_from(&first.id);
                    self.send(GuiMessage::Configure {
                        tts: Some(TtsConfig { model: Some(first.id), ..Default::default() }),
                        asr: None,
                        audio: None,
                        voice: None,
                        pipeline: None,
                    });
                }
                // backend(再)起動で request id は 1 から振り直される
                self.last_accepted_request = 0;
                self.cancelled_upto = 0;
                self.turns.backend_restarted();
                self.status_hint = if mock { "モック接続".into() } else { "実エンジン接続".into() };
            }
            BackendMessage::State { tts, asr, mic_running } => {
                if self.mic_running && !mic_running {
                    self.turns.mic_stopped();
                }
                self.mic_running = mic_running;
                // 停止中は State 応答で解除。開始中は TTS と ASR の両方が準備完了/エラーに
                // なるまでオーバーレイを維持し、ロードの様子を見せる(ASR だけ見ると
                // TTS ロード中に閉じる)。
                match self.mic_transition {
                    MicTransition::Stopping => self.mic_transition = MicTransition::None,
                    MicTransition::Starting => {
                        let settled = |s: &EngineState| matches!(s.phase.as_str(), "ready" | "error");
                        if settled(&tts) && settled(&asr) {
                            self.mic_transition = MicTransition::None;
                        }
                    }
                    MicTransition::None => {}
                }
                let since = |loading: bool, prev: Option<Instant>| {
                    if loading { prev.or_else(|| Some(Instant::now())) } else { None }
                };
                self.tts_loading_since = since(tts.phase == "loading", self.tts_loading_since);
                self.asr_loading_since = since(asr.phase == "loading", self.asr_loading_since);
                self.tts_state = tts;
                self.asr_state = asr;
            }
            BackendMessage::Log { level, message } => {
                self.push_log(format!("[{level}] {message}"));
            }
            BackendMessage::MicLevel { db, .. } => {
                self.mic_level_db = db;
            }
            BackendMessage::AsrPartial { utterance, text, .. } => {
                self.turns.asr_partial(utterance, text);
                self.stream_scroll.scroll_to_bottom();
            }
            BackendMessage::AsrFinal { utterance, text, asr_ms, delivery, .. } => {
                if asr_ms.is_some() {
                    self.last_asr_ms = asr_ms;
                }
                self.turns.asr_final(utterance, text, asr_ms, self.auto_speak);
                self.turns.set_delivery(utterance, delivery);
                self.stream_scroll.scroll_to_bottom();
            }
            BackendMessage::SpeakAccepted { request, origin, tag, utterance, .. } => {
                self.last_accepted_request = self.last_accepted_request.max(request);
                self.turns.speak_accepted(request, tag.as_deref(), utterance, self.selected_voice_name.clone());
                self.push_log(format!("発話受付 request={request} origin={origin}"));
                self.stream_scroll.scroll_to_bottom();
            }
            BackendMessage::TtsChunkStart { request, chunk, text } => {
                self.turns.chunk_start(request, chunk, text);
            }
            BackendMessage::TtsAudio {
                request,
                chunk,
                wav_base64,
                duration_ms,
                path,
                first_chunk,
                rtf,
                first_chunk_ms,
                e2e_ms,
                ..
            } => {
                let received = Instant::now();
                if request <= self.cancelled_upto {
                    self.push_log(format!("キャンセル済み request={request} の音声を破棄"));
                    return;
                }
                if let Some(audio) = &self.audio {
                    if let Err(e) = audio.enqueue_wav_base64(&wav_base64) {
                        self.push_log(format!("音声キュー追加失敗: {e}"));
                    }
                }
                let mut total_e2e = None;
                if first_chunk {
                    // backend 側(話し終わり→送出)+ GUI 側(受信→再生キュー投入)
                    let local_ms = received.elapsed().as_millis() as u64;
                    self.last_first_chunk_ms = first_chunk_ms;
                    self.last_rtf = rtf;
                    if let Some(e2e) = e2e_ms {
                        let total = e2e + local_ms;
                        total_e2e = Some(total);
                        self.last_e2e_ms = Some(total);
                        self.e2e_history.push_back(total);
                        if self.e2e_history.len() > 20 {
                            self.e2e_history.pop_front();
                        }
                    }
                }
                self.turns.chunk_audio(request, chunk, path, duration_ms, total_e2e);
            }
            BackendMessage::TtsChunkDone { .. } => {}
            BackendMessage::SpeakDone { request, chunks, cancelled, failed } => {
                self.turns.speak_done(request, cancelled, failed);
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
                if let Some(s) = self.sys {
                    // 失敗の瞬間のリソースをログに残す(原因調査用)
                    self.push_log(format!("[sys] {}", format_sample(&s)));
                }
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
                    let index = self.input_devices.iter().find(|d| &d.name == name).map(|d| d.index);
                    self.send(GuiMessage::Configure {
                        tts: None,
                        asr: None,
                        audio: Some(AudioConfig { input_device_index: index }),
                        voice: None,
                        pipeline: None,
                    });
                }
                self.pending_input_items = Some(items);
            }
        }
        cx.notify();
    }

    // ---------- ライブ(マイク) ----------

    fn toggle_mic(&mut self, cx: &mut Context<Self>) {
        // 遷移中(開始中/停止中)の再操作は無視: マイクの短時間反復 open/close は
        // USB オーディオドライバを落とすことがある(2026-10-09 に BugCheck 0xD1 を2度発生)。
        if self.mic_transition != MicTransition::None {
            return;
        }
        let (transition, msg, timeout_log) = if self.mic_running {
            (MicTransition::Stopping, GuiMessage::StopSession, "ライブの停止がタイムアウトしました")
        } else {
            (
                MicTransition::Starting,
                GuiMessage::StartSession,
                "ライブの開始がタイムアウトしました(もう一度お試しください)",
            )
        };
        self.mic_transition = transition;
        self.send(msg);
        let started_at = Instant::now();
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(MIC_TRANSITION_TIMEOUT).await;
            let _ = this.update(cx, |app, cx| {
                if app.mic_transition != MicTransition::None && started_at.elapsed() >= MIC_TRANSITION_TIMEOUT {
                    app.mic_transition = MicTransition::None;
                    app.push_log(timeout_log.into());
                    cx.notify();
                }
            });
        })
        .detach();
        cx.notify();
    }

    // ---------- 発話 ----------

    /// Speak を送る(話し方・seed は現在の設定)。tag で speak_accepted をターンへ戻す。
    fn send_speak(&mut self, text: String, turn_id: u64, cx: &mut Context<Self>) {
        let caption = {
            let c = self.caption_input.read(cx).value().trim().to_string();
            (!c.is_empty()).then_some(c)
        };
        let seed = if self.random_seed {
            None
        } else {
            self.seed_input.read(cx).value().trim().parse::<i64>().ok()
        };
        self.send(GuiMessage::Speak {
            text,
            caption,
            ref_wavs: None,
            seed,
            tag: Some(tag_for(turn_id)),
            delivery: self.turns.get(turn_id).and_then(|turn| turn.delivery.clone()),
        });
        self.stream_scroll.scroll_to_bottom();
    }

    fn speak_from_composer(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let text = self.composer.read(cx).value().trim().to_string();
        if text.is_empty() {
            self.composer.update(cx, |s, cx| s.focus(window, cx));
            return;
        }
        let id = self.turns.push_typed(text.clone());
        self.send_speak(text, id, cx);
        self.composer.update(cx, |s, cx| s.set_value("", window, cx));
        self.persist_settings(cx);
        cx.notify();
    }

    fn insert_annotation(&mut self, emoji: &'static str, window: &mut Window, cx: &mut Context<Self>) {
        self.composer.update(cx, |s, cx| {
            // 選択が空ならカーソル位置へ挿入し、範囲があればそこを置き換える。
            s.replace(emoji, window, cx);
            s.focus(window, cx);
        });
        cx.notify();
    }

    /// 確認待ちのターンをそのまま話す。
    fn confirm_turn(&mut self, id: u64, cx: &mut Context<Self>) {
        let Some(text) = self.turns.get(id).map(|t| t.text.clone()) else { return };
        self.turns.mark_queued(id);
        self.send_speak(text, id, cx);
        cx.notify();
    }

    /// ターンの文を入力欄へ移して訂正する(確認待ちだったものは発話しない扱いにする)。
    fn edit_turn(&mut self, id: u64, window: &mut Window, cx: &mut Context<Self>) {
        let Some(text) = self.turns.get(id).map(|t| t.text.clone()) else { return };
        self.turns.dismiss(id);
        self.composer.update(cx, |s, cx| {
            s.set_value(text.as_str(), window, cx);
            s.focus(window, cx);
        });
        cx.notify();
    }

    fn dismiss_turn(&mut self, id: u64, cx: &mut Context<Self>) {
        self.turns.dismiss(id);
        cx.notify();
    }

    /// 同じ文を今の声・話し方でもう一度話す(新しいターンになる)。
    fn respeak_turn(&mut self, id: u64, cx: &mut Context<Self>) {
        let Some((text, delivery)) = self.turns.get(id).map(|t| (t.text.clone(), t.delivery.clone())) else { return };
        if text.trim().is_empty() {
            return;
        }
        let new_id = self.turns.push_typed(text.clone());
        self.turns.set_delivery_for_id(new_id, delivery);
        self.send_speak(text, new_id, cx);
        cx.notify();
    }

    /// 届けた音声(保存済み WAV)を再生し直す。
    fn replay_turn(&mut self, id: u64) {
        let Some(turn) = self.turns.get(id) else { return };
        let paths: Vec<String> = turn.audio_paths().into_iter().map(String::from).collect();
        let Some(audio) = &self.audio else {
            self.push_log("出力デバイスが開かれていません".into());
            return;
        };
        let mut errors = Vec::new();
        for path in &paths {
            match std::fs::read(path) {
                Ok(bytes) => {
                    if let Err(e) = audio.enqueue_wav_bytes(bytes) {
                        errors.push(format!("再生失敗: {e}"));
                    }
                }
                Err(e) => errors.push(format!("ファイル読込失敗 {path}: {e}")),
            }
        }
        for e in errors {
            self.push_log(e);
        }
    }

    fn cancel_speak(&mut self, cx: &mut Context<Self>) {
        self.send(GuiMessage::CancelSpeak);
        self.cancelled_upto = self.last_accepted_request;
        if let Some(audio) = &self.audio {
            // clear() は内部で play() し直す(rodio の clear は Sink を pause するため)
            audio.clear();
        }
        self.turns.cancel_active();
        cx.notify();
    }

    /// 合成中・再生中・受付待ちのいずれか(「止める」を出す条件)
    fn is_delivering(&self) -> bool {
        let playing = self.audio.as_ref().is_some_and(|a| a.pending_chunks() > 0);
        playing || self.turns.iter().any(|t| t.status.is_active())
    }

    // ---------- 音声キュー・声・認識 ----------

    fn set_auto_speak(&mut self, on: bool, cx: &mut Context<Self>) {
        if self.auto_speak == on {
            return;
        }
        self.auto_speak = on;
        self.send(GuiMessage::Configure {
            tts: None,
            asr: None,
            audio: None,
            voice: None,
            pipeline: Some(PipelineConfig {
                auto_speak: Some(on),
                ..Default::default()
            }),
        });
        self.persist_settings(cx);
        cx.notify();
    }

    fn set_performance_enabled(&mut self, on: bool, cx: &mut Context<Self>) {
        if self.performance_enabled == on {
            return;
        }
        self.performance_enabled = on;
        self.send(GuiMessage::Configure {
            tts: None,
            asr: None,
            audio: None,
            voice: None,
            pipeline: Some(PipelineConfig {
                performance_enabled: Some(on),
                ..Default::default()
            }),
        });
        self.persist_settings(cx);
        cx.notify();
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
    fn apply_voice(&mut self, name: String, cx: &mut Context<Self>) {
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
            None => self.push_log("声を既定に戻しました(話し方の指示/自動音質で合成)".into()),
        }
        self.persist_settings(cx);
    }

    /// 声ライブラリへ取り込む(ドロップ/ファイル選択の共通入口)。
    /// 音声(wav/flac)は data/voices へコピーして選択し、画像は選択中の声のアイコンにする。
    /// 音声を先に処理するので、音声と画像を同時に渡せば新しい声に画像が付く。
    fn import_voice_files(&mut self, paths: &[PathBuf], window: &mut Window, cx: &mut Context<Self>) {
        let dir = self.root.join("data").join("voices");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.push_log(format!("[error:voice] 声フォルダを作れません: {e}"));
            return;
        }
        let (audio, rest): (Vec<_>, Vec<_>) = paths.iter().partition(|p| is_voice_audio(p));
        let mut imported = None;
        for src in audio {
            match copy_into_voice_bank(src, &dir) {
                Ok(name) => {
                    self.push_log(format!("声を追加: {name}"));
                    imported = Some(name);
                }
                Err(e) => self.push_log(format!("[error:voice] {}: {e}", src.display())),
            }
        }
        self.refresh_voices(imported.as_deref(), window, cx);
        if let Some(name) = imported {
            self.apply_voice(name, cx);
        }
        for src in rest {
            if !is_voice_image(src) {
                self.push_log(format!("[error:voice] 非対応のファイル: {}", src.display()));
                continue;
            }
            let Some(name) = self.selected_voice_name.clone() else {
                self.push_log("[error:voice] 画像は声を選んでから追加してください".into());
                continue;
            };
            match set_voice_image(src, &dir, &name) {
                Ok(()) => self.push_log(format!("声のアイコンを設定: {name}")),
                Err(e) => self.push_log(format!("[error:voice] {}: {e}", src.display())),
            }
        }
        cx.notify();
    }

    /// ファイル選択ダイアログから声を追加する。
    fn pick_voice_files(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: None,
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                let _ = this.update(cx, |this, cx| {
                    // Window が要るので、取り込みは次の render で実行する
                    this.pending_voice_import = Some(paths);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    /// 選択中の声をライブラリから削除し、既定の声に戻す。
    fn delete_selected_voice(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(name) = self.selected_voice_name.clone() else { return };
        let dir = self.root.join("data").join("voices");
        if let Some((_, path)) = self.voices.iter().find(|(n, _)| *n == name) {
            let _ = std::fs::remove_file(path);
        }
        remove_voice_images(&dir, &name);
        self.push_log(format!("声を削除: {name}"));
        self.refresh_voices(None, window, cx);
        self.apply_voice(DEFAULT_VOICE_LABEL.to_string(), cx);
        cx.notify();
    }

    /// data/voices を再スキャンして選択肢を更新する(`select` が None なら現在の選択を維持)。
    fn refresh_voices(&mut self, select: Option<&str>, window: &mut Window, cx: &mut Context<Self>) {
        self.voices = scan_voice_bank(&self.root);
        let mut items = vec![DEFAULT_VOICE_LABEL.to_string()];
        items.extend(self.voices.iter().map(|(n, _)| n.clone()));
        let selected = select
            .map(str::to_string)
            .or_else(|| self.selected_voice_name.clone())
            .filter(|n| self.voices.iter().any(|(vn, _)| vn == n))
            .unwrap_or_else(|| DEFAULT_VOICE_LABEL.to_string());
        self.voice_select.update(cx, |s, cx| {
            s.set_items(items, window, cx);
            s.set_selected_value(&selected, window, cx);
        });
    }

    /// 声のアイコン画像(あれば)。
    pub(super) fn voice_image(&self, name: &str) -> Option<PathBuf> {
        find_voice_image(&self.root.join("data").join("voices"), name)
    }

    /// ASR プロバイダの選択適用(ローカル/クラウド)。次回のライブ開始から新エンジンで動く
    /// (preload も組み直される)。
    fn apply_asr_provider(&mut self, label: String, cx: &mut Context<Self>) {
        let Some((_, engine)) = ASR_PROVIDERS.iter().find(|(l, _)| *l == label) else {
            return;
        };
        self.selected_asr_engine = engine.to_string();
        self.send(GuiMessage::Configure {
            tts: None,
            asr: Some(sttts_protocol::AsrConfig {
                engine: Some(engine.to_string()),
                ..Default::default()
            }),
            audio: None,
            voice: None,
            pipeline: None,
        });
        self.push_log(if *engine == "gemini" {
            "認識をクラウド(Gemini Live API)に切替しました(発話ごとにクラウドへ送信されます)".into()
        } else {
            format!("認識をローカル({engine})に切替しました")
        });
        self.persist_settings(cx);
        cx.notify();
    }

    /// Gemini API キー入力欄の確定(Enter / フォーカスアウト)。変更があれば暗号化して
    /// data/config.json に保存し、backend へ送る(backend はエンジンを作り直す)。
    fn apply_gemini_key(&mut self, cx: &mut Context<Self>) {
        let key = self.gemini_key_input.read(cx).value().trim().to_string();
        if key == self.gemini_api_key {
            return;
        }
        self.gemini_key_protected = if key.is_empty() { None } else { secret::protect(&key) };
        if !key.is_empty() && self.gemini_key_protected.is_none() {
            self.push_log("Gemini API キーを暗号化保存できませんでした(この起動中のみ有効)".into());
        }
        self.gemini_api_key = key;
        // 空文字 = GUI では未設定(backend は環境変数にフォールバック)
        self.send(GuiMessage::Configure {
            tts: None,
            asr: Some(sttts_protocol::AsrConfig {
                gemini_api_key: Some(self.gemini_api_key.clone()),
                ..Default::default()
            }),
            audio: None,
            voice: None,
            pipeline: None,
        });
        self.push_log(if self.gemini_api_key.is_empty() {
            "Gemini API キーを削除しました".into()
        } else {
            "Gemini API キーを保存しました(このPCのユーザーでのみ復号できる形で暗号化)".into()
        });
        self.persist_settings(cx);
        cx.notify();
    }

    // ---------- 詳細設定 ----------

    fn select_model(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(model) = self.models.get(index) else { return };
        let (model_id, label) = (model.id.clone(), model.label.clone());
        self.selected_model_id.clone_from(&model_id);
        self.send(GuiMessage::Configure {
            tts: Some(TtsConfig {
                model: Some(model_id),
                ..Default::default()
            }),
            asr: None,
            audio: None,
            voice: None,
            pipeline: None,
        });
        self.push_log(format!("モデル切替: {label}"));
        self.persist_settings(cx);
        cx.notify();
    }

    fn set_random_seed(&mut self, on: bool, cx: &mut Context<Self>) {
        self.random_seed = on;
        self.persist_settings(cx);
        cx.notify();
    }

    /// 入力デバイス選択の適用。ライブ中は停止して、再開は利用者に任せる。
    fn apply_input_device(&mut self, name: String, _window: &mut Window, cx: &mut Context<Self>) {
        let index = if name == DEFAULT_INPUT_LABEL {
            None
        } else {
            self.input_devices.iter().find(|d| d.name == name).map(|d| d.index)
        };
        self.selected_input_name = index.is_some().then_some(name);
        self.send(GuiMessage::Configure {
            tts: None,
            asr: None,
            audio: Some(AudioConfig { input_device_index: index }),
            voice: None,
            pipeline: None,
        });
        if self.mic_running && self.mic_transition == MicTransition::None {
            // 停止→即再開はbackendの再開クールダウンに拒否されるうえ、デバイスの
            // 短時間反復 open/close はドライバクラッシュの原因。停止のみ送り、
            // 再開は利用者の操作(ライブ開始)に任せる。
            self.mic_transition = MicTransition::Stopping;
            self.send(GuiMessage::StopSession);
            self.push_log("入力デバイスを変更したためライブを停止しました。新しいデバイスで「ライブ開始」を押してください".into());
        }
        self.persist_settings(cx);
        cx.notify();
    }

    /// 出力デバイス選択の適用(ストリームを張り直す。未再生キューは破棄)。
    fn apply_output_device(&mut self, name: String, cx: &mut Context<Self>) {
        let preferred = (name != DEFAULT_OUTPUT_LABEL).then_some(name.clone());
        match audio::AudioOut::open(preferred.as_deref()) {
            Ok(out) => {
                if let Some(old) = self.audio.as_ref() {
                    old.clear();
                }
                self.audio = Some(out);
                self.selected_output_name = preferred;
                self.push_log(format!("出力デバイスを切替: {name}"));
            }
            Err(e) => self.push_log(format!("出力デバイスの切替に失敗: {e:#}")),
        }
        self.persist_settings(cx);
        cx.notify();
    }

    /// data フォルダ(gui.log / config.json)を開く。
    fn open_data_folder(&mut self) {
        let dir = self.root.join("data");
        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
    }

    // ---------- 診断・永続化 ----------

    fn push_log(&mut self, line: String) {
        if let Some(f) = self.log_file.as_mut() {
            let _ = writeln!(f, "{line}");
        }
        if !self.log_open && is_error_line(&line) {
            self.unread_errors += 1;
        }
        self.logs.push_back(line);
        while self.logs.len() > 300 {
            self.logs.pop_front();
        }
        self.log_scroll.scroll_to_bottom();
    }

    fn toggle_log(&mut self, cx: &mut Context<Self>) {
        self.log_open = !self.log_open;
        if self.log_open {
            self.unread_errors = 0;
            self.log_scroll.scroll_to_bottom();
        }
        cx.notify();
    }

    fn persist_settings(&self, cx: &App) {
        let caption = self.caption_input.read(cx).value().to_string();
        let saved = settings::AppSettings {
            mock: Some(self.mock),
            tts_model: Some(self.selected_model_id.clone()),
            caption: Some(caption),
            auto_speak: Some(self.auto_speak),
            performance_enabled: Some(self.performance_enabled),
            random_seed: Some(self.random_seed),
            input_device: self.selected_input_name.clone(),
            output_device: self.selected_output_name.clone(),
            voice: self.selected_voice_name.clone(),
            asr_provider: Some(self.selected_asr_engine.clone()),
            gemini_api_key_protected: self.gemini_key_protected.clone(),
        };
        saved.save(&self.root);
    }

    /// 設定を保存してバックエンドを停止する。take() で二重停止を防ぐ。
    fn shutdown(&mut self, cx: &mut Context<Self>) {
        self.persist_settings(cx);
        if let Some(b) = self.backend.take() {
            b.shutdown();
        }
    }

    // ---------- 表示用の要約 ----------

    /// 話し終えてから音声が鳴り始めるまで(=応答速度)。直近値と直近20回の中央値。
    fn latency_summary(&self) -> Option<(u64, u64)> {
        let last = self.last_e2e_ms?;
        let mut v: Vec<u64> = self.e2e_history.iter().copied().collect();
        v.sort_unstable();
        Some((last, v.get(v.len() / 2).copied().unwrap_or(last)))
    }

    /// 今選んでいる声の言い方(「既定の声」/「さくら の声」)
    fn voice_phrase(&self) -> String {
        voice_phrase(self.selected_voice_name.as_deref())
    }

    fn asr_provider_label(&self) -> &'static str {
        ASR_PROVIDERS
            .iter()
            .find(|(_, e)| *e == self.selected_asr_engine)
            .map(|(l, _)| *l)
            .unwrap_or("ローカル")
    }
}

fn idle_state() -> EngineState {
    EngineState {
        phase: "idle".into(),
        detail: None,
        model: None,
    }
}

/// 声の表示名(None = 既定の声)。「〜で届けます」等に続けて使う
pub(crate) fn voice_phrase(name: Option<&str>) -> String {
    match name {
        Some(n) => format!("{n} の声"),
        None => DEFAULT_VOICE_LABEL.to_string(),
    }
}

/// エラー表示(赤字・未読バッジ)の対象行
pub(crate) fn is_error_line(line: &str) -> bool {
    line.starts_with("[error")
        || line.starts_with("[ERROR")
        || line.starts_with("[WARN")
        || line.contains("エラー")
        || line.contains("失敗")
        || line.contains("切れました")
}

/// エンジン状態の短い表示(固定語のみ。detail のような長い文字列はログで確認する)
pub(crate) fn phase_label(state: &EngineState) -> &'static str {
    match state.phase.as_str() {
        "ready" => "準備完了",
        "loading" => "読み込み中…",
        "error" => "エラー",
        _ => "未読み込み",
    }
}

/// 起動時に data/gui.log を空にして開く(古いログは残さない)。
fn open_log_file(root: &std::path::Path) -> Option<std::fs::File> {
    let dir = root.join("data");
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::File::create(dir.join("gui.log")).ok()
}

const VOICE_AUDIO_EXT: &[&str] = &["wav", "flac"];
const VOICE_IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "webp"];

fn has_ext(p: &std::path::Path, exts: &[&str]) -> bool {
    p.extension()
        .and_then(|x| x.to_str())
        .is_some_and(|x| exts.iter().any(|e| x.eq_ignore_ascii_case(e)))
}

fn is_voice_audio(p: &std::path::Path) -> bool {
    has_ext(p, VOICE_AUDIO_EXT)
}

fn is_voice_image(p: &std::path::Path) -> bool {
    has_ext(p, VOICE_IMAGE_EXT)
}

/// data/voices の音声(wav/flac)を声バンクとして読み込む(ファイル名=話者名)。
fn scan_voice_bank(root: &std::path::Path) -> Vec<(String, PathBuf)> {
    let dir = root.join("data").join("voices");
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            if is_voice_audio(&p) {
                if let Some(name) = p.file_stem().and_then(|s| s.to_str()).filter(|n| !n.is_empty()) {
                    out.push((name.to_string(), p.clone()));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn find_voice_image(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    VOICE_IMAGE_EXT
        .iter()
        .map(|e| dir.join(format!("{name}.{e}")))
        .find(|p| p.is_file())
}

fn remove_voice_images(dir: &std::path::Path, name: &str) {
    for e in VOICE_IMAGE_EXT {
        let _ = std::fs::remove_file(dir.join(format!("{name}.{e}")));
    }
}

/// 音声を声バンクへコピーし、声の名前(拡張子なし)を返す。同名があれば連番を付ける。
fn copy_into_voice_bank(src: &std::path::Path, dir: &std::path::Path) -> std::io::Result<String> {
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.trim().replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_"))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "voice".to_string());
    let ext = src
        .extension()
        .and_then(|x| x.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "wav".to_string());
    let taken = |n: &str| VOICE_AUDIO_EXT.iter().any(|e| dir.join(format!("{n}.{e}")).exists());
    let mut name = stem.clone();
    let mut i = 2;
    while taken(&name) {
        name = format!("{stem} ({i})");
        i += 1;
    }
    std::fs::copy(src, dir.join(format!("{name}.{ext}")))?;
    Ok(name)
}

/// 画像を声のアイコンとして data/voices/<name>.<ext> へコピーする(既存のアイコンは置換)。
fn set_voice_image(src: &std::path::Path, dir: &std::path::Path, name: &str) -> std::io::Result<()> {
    let ext = src
        .extension()
        .and_then(|x| x.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "png".to_string());
    remove_voice_images(dir, name);
    std::fs::copy(src, dir.join(format!("{name}.{ext}")))?;
    Ok(())
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// "VRAM 5.0/10.0GB (アプリ 3.1GB) · RAM 33.2/47.7GB"
pub(crate) fn format_sample(s: &sysmon::SysSample) -> String {
    let gb = |b: u64| b as f64 / GIB;
    let mut parts = Vec::new();
    if s.vram_total > 0 {
        let app = s.app_vram.map(|a| format!(" (アプリ {:.1}GB)", gb(a))).unwrap_or_default();
        parts.push(format!("VRAM {:.1}/{:.1}GB{app}", gb(s.vram_used), gb(s.vram_total)));
    }
    if s.ram_total > 0 {
        parts.push(format!("RAM {:.1}/{:.1}GB", gb(s.ram_used), gb(s.ram_total)));
    }
    parts.join(" · ")
}
