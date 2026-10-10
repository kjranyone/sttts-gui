//! アプリの状態とふるまい(描画は子モジュール)。
//!
//! # 責務と画面構成
//! sttts は「話した内容を、選んだ声で届ける」ための中継器。主画面は次の3つを分けて見せる
//! (docs/acting-reconstruction-design.md「UI への反映」):
//!
//! - **何を話したか** — ストリーム(中央)。1ターン = 入力1件と、それを届けた声の対応
//! - **どう伝えるか** — 右レールの「音声キュー」(自動再生 ON/OFF、テンポと間の再現)
//! - **どの声で届けるか** — 右レールの「声」(Irodori の声・話し方・seed)
//!
//! 環境で一度決まる設定(入出力デバイス、TTS モデル)は「詳細設定」シート
//! (既定は閉。入口はタイトルバーの歯車のみ)に置き、主画面に出さない(AGENTS.md「設計の前提」)。

mod chrome;
mod devices;
mod help;
mod kit;
mod log;
mod rail;
mod sampling;
mod sheet;
mod stream;
mod title_bar;
mod voice_library;
mod voice_bank;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use gpui_kit::component::input::{InputEvent, InputState, TextareaState};
use gpui_kit::component::select::{SelectEvent, SelectState};
use gpui_kit::component::IndexPath;
use gpui_kit::*;
use sttts_protocol::{
    AnyMessage, AsrConfig, BackendMessage, EngineState, GuiMessage, ModelInfo,
    PipelineConfig, TtsConfig, VoiceConfig,
};

use sttts_i18n::{tr, trf};

use crate::turns::{Playback, Turns, tag_for};
use crate::device_picker::Dir;
use crate::locale::Lang;
use crate::{audio, backend, locale, secret, settings, sysmon};
use log::open_log_file;
pub(crate) use log::is_error_line;
use voice_bank::{default_voice_label, scan_voice_bank, voice_durations};
pub(crate) use voice_bank::voice_phrase;

/// ASR プロバイダ選択(表示名, asr.engine 値)。ローカルとクラウドを選べる。
pub(crate) fn asr_providers() -> [(&'static str, &'static str); 3] {
    [
        (tr!("Cloud (Gemini)", "クラウド(Gemini)", "云端(Gemini)"), "gemini"),
        (tr!("Local (Nemotron)", "ローカル(Nemotron)", "本地(Nemotron)"), "nemotron"),
        (tr!("Local (kotoba)", "ローカル(kotoba)", "本地(kotoba)"), "kotoba"),
    ]
}

/// 文字列を項目に持つ選択欄(声・認識・言語・デバイス)
pub(crate) type TextSelect = SelectState<Vec<String>>;
type TextSelectEvent = SelectEvent<Vec<String>>;

/// Gemini API キーの発行ページ(Google AI Studio)
pub(crate) const GEMINI_KEY_URL: &str = "https://aistudio.google.com/apikey";
/// Gemini API キー入力が止まってから適用するまでの待ち時間
const GEMINI_KEY_APPLY_DELAY: Duration = Duration::from_millis(800);

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
    /// 再生キュー上の (ターン, チャンク)。カードの「再生中」表示に使う
    playback: Playback,
    stream_scroll: gpui::ScrollHandle,
    composer: Entity<TextareaState>,

    // ---- 音声キュー
    /// 確定文を自動で音声キューへ流す(false = カードで止めて手動で発話)
    auto_speak: bool,
    performance_enabled: bool,

    // ---- 声
    voices: Vec<(String, PathBuf)>,
    /// 参照音声の長さ(秒。声のライブラリの表示用。refresh_voices で更新)
    voice_secs: std::collections::HashMap<String, f32>,
    /// 声のライブラリ(管理シート)を開いているか
    voice_library_open: bool,
    /// 削除の確認を出している声(1 つだけ)
    voice_delete_confirm: Option<String>,
    /// 試聴中の声
    previewing_voice: Option<String>,
    /// ファイル選択ダイアログの結果。Window が要るので render で取り込む。
    pending_voice_import: Option<Vec<PathBuf>>,
    /// PC リソース(RAM / GPU 専用メモリ)の最新サンプル
    sys: Option<sysmon::SysSample>,
    backend_pid: Arc<AtomicU32>,
    /// エンジンが loading になった時刻(経過秒の表示用)
    tts_loading_since: Option<Instant>,
    asr_loading_since: Option<Instant>,
    selected_voice_name: Option<String>,
    voice_select: Entity<TextSelect>,
    /// 話し方の指示(Irodori の caption)
    caption_input: Entity<InputState>,

    // ---- 認識
    asr_select: Entity<TextSelect>,
    selected_asr_engine: String,
    /// Gemini API キー入力欄(伏せ字)。確定は Enter / フォーカスアウト / 入力が止まって少し経ったとき
    gemini_key_input: Entity<InputState>,
    /// backend へ送ったキー(空 = GUI では未設定)
    gemini_api_key: String,
    /// 保存用の暗号化済みキー(secret::protect の出力)
    gemini_key_protected: Option<String>,
    /// キー入力の変更ごとに増やす。待ち時間後も同じなら(入力が止まった)適用する
    gemini_key_edit_seq: u64,

    // ---- 詳細設定(環境で一度決まるもの)
    settings_open: bool,
    /// 利用者が選んだ表示言語(None = OS の表示言語に従う。保存もしない)
    language: Option<Lang>,
    language_select: Entity<TextSelect>,
    /// 「?」から開いている解説
    help_topic: Option<help::HelpTopic>,
    /// 入力デバイス(ドライバ → デバイス / ASIO チャンネル)。一覧はエンジンから届く
    input_dev: devices::DeviceSelect,
    /// 出力デバイス(ドライバ → デバイス / ASIO チャンネル)。一覧は GUI が列挙する
    output_dev: devices::DeviceSelect,
    /// 保存済みの入力選択 (ドライバ, 候補)。最初のデバイス一覧の到着時に復元する
    saved_input: Option<(Option<String>, Option<String>)>,
    seed_input: Entity<InputState>,
    random_seed: bool,
    /// 合成パラメータ(Irodori の tts.sampling)の編集欄
    sampling: sampling::SamplingEditor,

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
                .placeholder(composer_placeholder())
                .auto_grow(1, 6)
        });
        let caption_input = cx.new(|cx| {
            let mut state = InputState::new(window, cx);
            if let Some(caption) = &saved.caption {
                state.set_value(caption.as_str(), window, cx);
            }
            state
        });
        let seed_input = cx.new(|cx| {
            let mut state = InputState::new(window, cx).placeholder("seed");
            if let Some(seed) = saved.seed {
                state.set_value(seed.to_string(), window, cx);
            }
            state
        });
        let sampling = sampling::SamplingEditor::new(&root, window, cx);

        // --- 入出力デバイス選択(ドライバ → 候補の 2 段)
        let saved_input = Some((saved.input_driver.clone(), saved.input_device.clone()));
        let input_dev = devices::DeviceSelect::new(Dir::Input, Vec::new(), (None, None), window, cx);
        let mut output_dev = devices::DeviceSelect::new(
            Dir::Output,
            devices::to_protocol(audio::list_output_devices()),
            (saved.output_driver.as_deref(), saved.output_device.as_deref()),
            window,
            cx,
        );

        // --- 声バンク(data/voices の wav を参照音声として選択できる)。同梱プリセットは未導入の分だけ先に書き出す
        let preset_error = sttts_engine::presets::install_presets(&root).err();
        let voices = scan_voice_bank(&root);
        let saved_voice = saved
            .voice
            .clone()
            .filter(|n| voices.iter().any(|(vn, _)| vn == n));
        let mut voice_items = vec![default_voice_label().to_string()];
        voice_items.extend(voices.iter().map(|(n, _)| n.clone()));
        let voice_sel_ix = saved_voice
            .as_ref()
            .and_then(|n| voices.iter().position(|(vn, _)| vn == n))
            .map(|p| IndexPath::new(p + 1))
            .or(Some(IndexPath::new(0)));
        let voice_select = cx.new(|cx| SelectState::new(voice_items, voice_sel_ix, window, cx));

        // --- ASR プロバイダ選択(ローカル/クラウド)
        let asr_items: Vec<String> = asr_providers().iter().map(|(l, _)| l.to_string()).collect();
        let saved_asr = saved.asr_provider.clone().unwrap_or_else(|| "nemotron".into());
        let asr_sel_ix = asr_providers()
            .iter()
            .position(|(_, e)| *e == saved_asr)
            .map(IndexPath::new);
        let asr_select = cx.new(|cx| SelectState::new(asr_items, asr_sel_ix, window, cx));

        // --- 表示言語(項目は各言語の自称なので、切り替えても作り直さない)
        let language = saved.language.as_deref().and_then(Lang::from_tag);
        let language_items: Vec<String> = Lang::ALL.iter().map(|l| l.native_name().to_string()).collect();
        let language_ix = Lang::ALL.iter().position(|l| *l == sttts_i18n::lang()).map(IndexPath::new);
        let language_select = cx.new(|cx| SelectState::new(language_items, language_ix, window, cx));

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

        let out_choice = output_dev.choice.clone();
        let (audio, audio_error) = match audio::AudioOut::open(out_choice.device_id.as_deref(), &out_choice.channels) {
            Ok(a) => (Some(a), None),
            Err(first) if out_choice.device_id.is_some() => {
                // 保存済みのデバイスを開けなければシステム既定で鳴らす
                output_dev.select_default();
                match audio::AudioOut::open(None, &[]) {
                    Ok(a) => (
                        Some(a),
                        Some(trf!(
                            "{first:#} (playing on the default output device)",
                            "{first:#}(既定の出力デバイスで再生します)",
                            "{first:#}(改用默认输出设备播放)"
                        )),
                    ),
                    Err(e) => (None, Some(format!("{e:#}"))),
                }
            }
            Err(e) => (None, Some(format!("{e:#}"))),
        };

        let selected_model_id = saved.tts_model.unwrap_or_else(|| sttts_engine::tts::DEFAULT_TTS_MODEL.into());

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
            playback: Playback::default(),
            stream_scroll: gpui::ScrollHandle::new(),
            composer,
            auto_speak,
            performance_enabled,
            voice_secs: voice_durations(&voices),
            voices,
            voice_library_open: false,
            voice_delete_confirm: None,
            previewing_voice: None,
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
            gemini_key_edit_seq: 0,
            settings_open: false,
            language,
            language_select,
            help_topic: None,
            input_dev,
            output_dev,
            saved_input,
            seed_input,
            random_seed,
            sampling,
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
            status_hint: tr!("Starting backend…", "バックエンドを起動中…", "正在启动后端…").into(),
            root,
        };
        if let Some(err) = audio_error {
            app.push_log(trf!(
                "[error:audio] Could not open the output device: {err}",
                "[error:audio] 出力デバイスを開けませんでした: {err}",
                "[error:audio] 无法打开输出设备:{err}"
            ));
        }
        if let Some(err) = preset_error {
            app.push_log(trf!(
                "[warn] Could not install the bundled voice presets: {err:#}",
                "[warn] 同梱の声プリセットを書き出せませんでした: {err:#}",
                "[warn] 无法写出内置的声音预设:{err:#}"
            ));
        }
        if gemini_key_unreadable {
            app.push_log(
                tr!(
                    "[warn] Could not decrypt the saved Gemini API key (it was saved on another PC or user). Please enter it again",
                    "[warn] 保存済みの Gemini API キーを復号できませんでした(別のPC/ユーザーで保存されたもの)。再入力してください",
                    "[warn] 无法解密已保存的 Gemini API 密钥(它是在其他电脑或用户下保存的)。请重新输入"
                ),
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
            asr: Some(AsrConfig {
                engine: Some(app.selected_asr_engine.clone()),
                gemini_api_key: (!app.gemini_api_key.is_empty()).then(|| app.gemini_api_key.clone()),
                ..Default::default()
            }),
            audio: None,
            voice: Some(app.voice_config(cx)),
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
        ) -> impl FnMut(Entity<TextSelect>, &TextSelectEvent, &mut Window, &mut App)
        + 'static {
            move |_, event, window, cx| {
                if let SelectEvent::Confirm(Some(name)) = event {
                    let _ = app.update(cx, |app, cx| f(app, name.clone(), window, cx));
                }
            }
        }

        let weak = cx.weak_entity();
        let subs = vec![
            window.subscribe(&self.input_dev.driver_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_input_driver(n, cx))),
            window.subscribe(&self.input_dev.choice_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_input_choice(n, cx))),
            window.subscribe(&self.output_dev.driver_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_output_driver(n, cx))),
            window.subscribe(&self.output_dev.choice_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_output_choice(n, cx))),
            window.subscribe(&self.voice_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_voice(n, cx))),
            window.subscribe(&self.asr_select, cx, on_confirm(weak.clone(), |a, n, _, cx| a.apply_asr_provider(n, cx))),
            window.subscribe(&self.language_select, cx, on_confirm(weak.clone(), |a, n, window, cx| {
                if let Some(lang) = Lang::from_native_name(&n) {
                    a.set_language(lang, window, cx);
                }
            })),
            window.subscribe(&self.gemini_key_input, cx, {
                let weak = weak.clone();
                move |_, event: &InputEvent, _window, cx| match event {
                    InputEvent::PressEnter { .. } | InputEvent::Blur => {
                        let _ = weak.update(cx, |app, cx| app.apply_gemini_key(cx));
                    }
                    // 右クリックメニューで貼り付け・余白クリックでは Enter も Blur も来ない。
                    // 入力が止まったら適用する(1文字ごとにエンジンを作り直さない)
                    InputEvent::Change => {
                        let _ = weak.update(cx, |app, cx| app.schedule_gemini_key_apply(cx));
                    }
                    _ => {}
                }
            }),
            // 話し方・seed は打つたびにエンジンへ送る(自動発話はエンジン側の設定で合成するため)。
            // 送るのは設定の差し替えだけで軽い。保存は確定時のみ。
            window.subscribe(&self.caption_input, cx, {
                let weak = weak.clone();
                move |_, event: &InputEvent, _window, cx| match event {
                    InputEvent::Change => {
                        let _ = weak.update(cx, |app, cx| app.send_voice_config(cx));
                    }
                    InputEvent::PressEnter { .. } | InputEvent::Blur => {
                        let _ = weak.update(cx, |app, cx| app.persist_settings(cx));
                    }
                    _ => {}
                }
            }),
            window.subscribe(&self.seed_input, cx, {
                let weak = weak.clone();
                move |_, event: &InputEvent, _window, cx| match event {
                    InputEvent::Change => {
                        let _ = weak.update(cx, |app, cx| app.send_voice_config(cx));
                    }
                    InputEvent::PressEnter { .. } | InputEvent::Blur => {
                        let _ = weak.update(cx, |app, cx| app.persist_settings(cx));
                    }
                    _ => {}
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
        self.subscribe_sampling_inputs(window, cx);
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

        // 再生位置の追従: Sink は残数しか通知しないので短い間隔で突き合わせ、
        // 鳴っているチャンクが変わったときだけ再描画する
        cx.spawn(async move |this, cx| loop {
            cx.background_executor().timer(Duration::from_millis(60)).await;
            if this
                .update(cx, |app, cx| {
                    if app.playback.is_empty() {
                        return;
                    }
                    let remaining = app.audio.as_ref().map_or(0, |a| a.pending_chunks());
                    if app.playback.sync(remaining) {
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
        self.status_hint = tr!("Starting backend…", "バックエンド起動中…", "正在启动后端…").into();
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
        self.status_hint = tr!("Waiting for backend…", "バックエンド接続待ち…", "等待后端连接…").into();

        // backend → UI の取り込みループ
        cx.spawn(async move |this, cx| {
            while let Ok(msg) = rx_events.recv().await {
                if this.update(cx, |app, cx| app.on_backend_message(msg, cx)).is_err() {
                    return;
                }
            }
            // チャネル閉鎖 = バックエンド終了
            let _ = this.update(cx, |app, cx| {
                app.push_log(
                    tr!(
                        "[error:backend] Lost connection to the backend",
                        "[error:backend] バックエンドとの接続が切れました",
                        "[error:backend] 与后端的连接已断开"
                    ),
                );
                app.status_hint = tr!("Backend stopped", "バックエンド停止", "后端已停止").into();
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
            self.push_log(tr!("[error:backend] Backend not connected", "[error:backend] バックエンド未接続", "[error:backend] 后端未连接"));
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
                    let gui = sttts_protocol::PROTOCOL_VERSION;
                    self.push_log(trf!(
                        "[warn] Protocol mismatch: backend={protocol} gui={gui}",
                        "[warn] プロトコル不一致: backend={protocol} gui={gui}",
                        "[warn] 协议不一致:backend={protocol} gui={gui}"
                    ));
                }
                self.connected = true;
                self.mock = mock;
                self.models = models;
                // 保存されていたモデルが今のエンジンで扱えない(以前の版で選んだ大型モデル等)ときは既定へ戻す
                if !self.models.iter().any(|m| m.id == self.selected_model_id)
                    && let Some(first) = self.models.first().cloned()
                {
                    let (old, new) = (&self.selected_model_id, &first.label);
                    self.push_log(trf!(
                        "TTS model {old} is not available; switched to {new}",
                        "音声合成モデル {old} は使えないため {new} に切り替えました",
                        "语音合成模型 {old} 不可用,已切换为 {new}"
                    ));
                    self.selected_model_id.clone_from(&first.id);
                    self.send(GuiMessage::configure_tts(TtsConfig { model: Some(first.id), ..Default::default() }));
                }
                // backend(再)起動で request id は 1 から振り直される
                self.last_accepted_request = 0;
                self.cancelled_upto = 0;
                self.turns.backend_restarted();
                self.status_hint = if mock {
                    tr!("Connected (mock)", "モック接続", "已连接(模拟)").into()
                } else {
                    tr!("Connected", "実エンジン接続", "已连接").into()
                };
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
            BackendMessage::AsrDiscarded { utterance } => {
                self.turns.asr_discarded(utterance);
            }
            BackendMessage::SpeakAccepted { request, origin, tag, utterance, .. } => {
                self.last_accepted_request = self.last_accepted_request.max(request);
                self.turns.speak_accepted(request, tag.as_deref(), utterance, self.selected_voice_name.clone());
                self.push_log(trf!(
                    "Speech accepted request={request} origin={origin}",
                    "発話受付 request={request} origin={origin}",
                    "已接受发话 request={request} origin={origin}"
                ));
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
                seed,
                first_chunk,
                rtf,
                first_chunk_ms,
                e2e_ms,
                ..
            } => {
                let received = Instant::now();
                if request <= self.cancelled_upto {
                    self.push_log(trf!(
                        "Discarded audio of cancelled request={request}",
                        "キャンセル済み request={request} の音声を破棄",
                        "已丢弃已取消 request={request} 的音频"
                    ));
                    return;
                }
                if let Some(audio) = &self.audio {
                    match audio.enqueue_wav_base64(&wav_base64) {
                        // ターン不明でも Sink の残数と揃えるため積む(id 0 はどのターンにも一致しない)
                        Ok(()) => self.playback.push(self.turns.id_for_request(request).unwrap_or(0), chunk),
                        Err(e) => self.push_log(trf!(
                            "[error:audio] Failed to queue audio: {e}",
                            "[error:audio] 音声キュー追加失敗: {e}",
                            "[error:audio] 加入音频队列失败:{e}"
                        )),
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
                self.turns.chunk_audio(request, chunk, path, duration_ms, total_e2e, seed);
            }
            BackendMessage::TtsChunkDone { .. } => {}
            BackendMessage::SpeakDone { request, chunks, cancelled, failed } => {
                self.turns.speak_done(request, cancelled, failed);
                self.push_log(trf!(
                    "Speech done request={request} chunks={chunks} cancelled={cancelled} failed={failed}",
                    "発話完了 request={request} chunks={chunks} cancelled={cancelled} failed={failed}",
                    "发话完成 request={request} chunks={chunks} cancelled={cancelled} failed={failed}"
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
            BackendMessage::Devices { inputs, .. } => self.on_input_devices(inputs, cx),
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
            (
                MicTransition::Stopping,
                GuiMessage::StopSession,
                tr!("[warn] Stopping live timed out", "[warn] ライブの停止がタイムアウトしました", "[warn] 停止直播超时"),
            )
        } else {
            (
                MicTransition::Starting,
                GuiMessage::StartSession,
                tr!(
                    "[warn] Starting live timed out (please try again)",
                    "[warn] ライブの開始がタイムアウトしました(もう一度お試しください)",
                    "[warn] 开始直播超时(请重试)"
                ),
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
                    app.push_log(timeout_log);
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
        let seed = self.fixed_seed(cx);
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
        let paths: Vec<(u32, String)> =
            turn.chunks.iter().filter_map(|c| Some((c.index, c.path.clone()?))).collect();
        let Some(audio) = &self.audio else {
            self.push_log(
                tr!("[error:audio] The output device is not open", "[error:audio] 出力デバイスが開かれていません", "[error:audio] 输出设备未打开"),
            );
            return;
        };
        let mut errors = Vec::new();
        for (chunk, path) in &paths {
            match std::fs::read(path) {
                Ok(bytes) => match audio.enqueue_wav_bytes(bytes) {
                    Ok(()) => self.playback.push(id, *chunk),
                    Err(e) => errors.push(trf!(
                        "[error:audio] Playback failed: {e}",
                        "[error:audio] 再生失敗: {e}",
                        "[error:audio] 播放失败:{e}"
                    )),
                },
                Err(e) => errors.push(trf!(
                    "[error:audio] Failed to read {path}: {e}",
                    "[error:audio] ファイル読込失敗 {path}: {e}",
                    "[error:audio] 读取文件失败 {path}:{e}"
                )),
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
        self.playback.clear();
        self.turns.cancel_active();
        cx.notify();
    }

    // ---------- 音声キュー・声・認識 ----------

    fn set_auto_speak(&mut self, on: bool, cx: &mut Context<Self>) {
        if self.auto_speak == on {
            return;
        }
        self.auto_speak = on;
        self.send(GuiMessage::configure_pipeline(PipelineConfig { auto_speak: Some(on), ..Default::default() }));
        self.persist_settings(cx);
        cx.notify();
    }

    fn set_performance_enabled(&mut self, on: bool, cx: &mut Context<Self>) {
        if self.performance_enabled == on {
            return;
        }
        self.performance_enabled = on;
        self.send(GuiMessage::configure_pipeline(PipelineConfig { performance_enabled: Some(on), ..Default::default() }));
        self.persist_settings(cx);
        cx.notify();
    }

    /// 右レール「声」の現在値から作る voice 設定(声・話し方・seed をまとめて送る)。
    /// マイクからの自動発話はこの設定で合成されるので、欄を変えたらすぐ送る。
    fn voice_config(&self, cx: &App) -> VoiceConfig {
        let ref_wav = self
            .selected_voice_name
            .as_ref()
            .and_then(|n| self.voices.iter().find(|(vn, _)| vn == n))
            .map(|(_, path)| path.to_string_lossy().into_owned());
        VoiceConfig {
            // 既定の声へ戻すときも空で送り、前の声の参照音声を残さない
            no_ref: Some(ref_wav.is_none()),
            ref_wavs: Some(ref_wav.into_iter().collect()),
            caption: Some(self.caption_input.read(cx).value().trim().to_string()),
            seed: self.fixed_seed(cx),
        }
    }

    /// 固定 seed(ランダム、または欄が空・不正なら None)
    fn fixed_seed(&self, cx: &App) -> Option<i64> {
        if self.random_seed {
            return None;
        }
        self.seed_input.read(cx).value().trim().parse::<i64>().ok()
    }

    fn send_voice_config(&mut self, cx: &App) {
        let voice = self.voice_config(cx);
        self.send(GuiMessage::configure_voice(voice));
    }

    /// ASR プロバイダの選択適用(ローカル/クラウド)。次回のライブ開始から新エンジンで動く
    /// (preload も組み直される)。
    fn apply_asr_provider(&mut self, label: String, cx: &mut Context<Self>) {
        let Some((_, engine)) = asr_providers().into_iter().find(|(l, _)| *l == label) else {
            return;
        };
        self.selected_asr_engine = engine.to_string();
        self.send(GuiMessage::configure_asr(AsrConfig { engine: Some(engine.to_string()), ..Default::default() }));
        self.push_log(if engine == "gemini" {
            tr!(
                "Recognition switched to the cloud (Gemini Live API). Each utterance is sent to the cloud",
                "認識をクラウド(Gemini Live API)に切替しました(発話ごとにクラウドへ送信されます)",
                "识别已切换到云端(Gemini Live API),每段发话都会发送到云端"
            )
            .into()
        } else {
            trf!(
                "Recognition switched to local ({engine})",
                "認識をローカル({engine})に切替しました",
                "识别已切换到本地({engine})"
            )
        });
        self.persist_settings(cx);
        cx.notify();
    }

    /// キー入力の変更から少し待って、その間に次の変更が無ければ確定する。
    fn schedule_gemini_key_apply(&mut self, cx: &mut Context<Self>) {
        self.gemini_key_edit_seq += 1;
        let seq = self.gemini_key_edit_seq;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(GEMINI_KEY_APPLY_DELAY).await;
            let _ = this.update(cx, |app, cx| {
                if app.gemini_key_edit_seq == seq {
                    app.apply_gemini_key(cx);
                }
            });
        })
        .detach();
    }

    /// Gemini API キー入力欄の確定(Enter / フォーカスアウト / 入力停止)。変更があれば暗号化して
    /// data/config.json に保存し、backend へ送る(backend はエンジンを作り直す)。
    fn apply_gemini_key(&mut self, cx: &mut Context<Self>) {
        let key = self.gemini_key_input.read(cx).value().trim().to_string();
        if key == self.gemini_api_key {
            return;
        }
        self.gemini_key_protected = if key.is_empty() { None } else { secret::protect(&key) };
        if !key.is_empty() && self.gemini_key_protected.is_none() {
            self.push_log(
                tr!(
                    "[warn] Could not save the Gemini API key encrypted (valid only until the app exits)",
                    "[warn] Gemini API キーを暗号化保存できませんでした(この起動中のみ有効)",
                    "[warn] 无法加密保存 Gemini API 密钥(仅在本次运行中有效)"
                ),
            );
        }
        self.gemini_api_key = key;
        // 空文字 = GUI では未設定(backend は環境変数にフォールバック)
        self.send(GuiMessage::configure_asr(AsrConfig {
            gemini_api_key: Some(self.gemini_api_key.clone()),
            ..Default::default()
        }));
        self.push_log(if self.gemini_api_key.is_empty() {
            tr!("Gemini API key removed", "Gemini API キーを削除しました", "已删除 Gemini API 密钥")
        } else {
            tr!(
                "Gemini API key saved (encrypted so only this user on this PC can decrypt it)",
                "Gemini API キーを保存しました(このPCのユーザーでのみ復号できる形で暗号化)",
                "已保存 Gemini API 密钥(已加密,仅本电脑的当前用户可解密)"
            )
        });
        self.persist_settings(cx);
        cx.notify();
    }

    // ---------- 詳細設定 ----------

    fn select_model(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(model) = self.models.get(index) else { return };
        let (model_id, label) = (model.id.clone(), model.label.clone());
        self.selected_model_id.clone_from(&model_id);
        self.send(GuiMessage::configure_tts(TtsConfig { model: Some(model_id), ..Default::default() }));
        self.push_log(trf!("Model: {label}", "モデル切替: {label}", "已切换模型:{label}"));
        self.persist_settings(cx);
        cx.notify();
    }

    /// 表示言語を切り替える。描画のたびに引く文言はすぐ変わる。選択欄の項目や入力欄の
    /// 案内文のように作成時に渡した文言は、ここで作り直す。
    fn set_language(&mut self, lang: Lang, window: &mut Window, cx: &mut Context<Self>) {
        self.language = Some(lang);
        if sttts_i18n::lang() != lang {
            locale::apply(lang);
            self.relocalize(window, cx);
        }
        self.persist_settings(cx);
        cx.notify();
    }

    fn relocalize(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.composer.update(cx, |s, cx| s.set_placeholder(composer_placeholder(), window, cx));
        self.refresh_voices(None, window, cx);
        let asr_items: Vec<String> = asr_providers().iter().map(|(l, _)| l.to_string()).collect();
        let asr_label = self.asr_provider_label().to_string();
        self.asr_select.update(cx, |s, cx| {
            s.set_items(asr_items, window, cx);
            s.set_selected_value(&asr_label, window, cx);
        });
        self.input_dev.relocalize();
        self.output_dev.relocalize();
        self.relocalize_sampling(window, cx);
        // モデルの表示名は hello で届いたもの。エンジンは同じプロセスなので、カタログを引き直す
        let catalog = sttts_engine::app::model_catalog();
        for m in &mut self.models {
            if let Some(fresh) = catalog.iter().find(|c| c.id == m.id) {
                m.clone_from(fresh);
            }
        }
    }

    /// 履歴の seed を固定 seed にする(ランダムで気に入った声を次からも使う)。
    fn use_seed(&mut self, seed: i64, window: &mut Window, cx: &mut Context<Self>) {
        self.seed_input.update(cx, |s, cx| s.set_value(seed.to_string(), window, cx));
        self.push_log(trf!("Fixed seed set to {seed}", "seed を {seed} に固定", "已将 seed 固定为 {seed}"));
        self.set_random_seed(false, cx);
    }

    fn set_random_seed(&mut self, on: bool, cx: &mut Context<Self>) {
        self.random_seed = on;
        self.send_voice_config(cx);
        self.persist_settings(cx);
        cx.notify();
    }

    /// data フォルダ(gui.log / config.json)を開く。
    fn open_data_folder(&mut self) {
        let dir = self.root.join("data");
        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
    }

    // ---------- 診断・永続化 ----------

    fn persist_settings(&self, cx: &App) {
        let caption = self.caption_input.read(cx).value().to_string();
        let saved = settings::AppSettings {
            language: self.language.map(|l| l.code().to_string()),
            mock: Some(self.mock),
            tts_model: Some(self.selected_model_id.clone()),
            caption: Some(caption),
            auto_speak: Some(self.auto_speak),
            performance_enabled: Some(self.performance_enabled),
            random_seed: Some(self.random_seed),
            seed: self.seed_input.read(cx).value().trim().parse::<i64>().ok(),
            input_driver: self.input_dev.saved().0,
            input_device: self.input_dev.saved().1,
            output_driver: self.output_dev.saved().0,
            output_device: self.output_dev.saved().1,
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
        asr_providers()
            .into_iter()
            .find(|(_, e)| *e == self.selected_asr_engine)
            .map(|(l, _)| l)
            .unwrap_or(tr!("Local", "ローカル", "本地"))
    }
}

fn composer_placeholder() -> &'static str {
    tr!("Type to speak", "文字で話す", "输入文字来说话")
}

fn idle_state() -> EngineState {
    EngineState {
        phase: "idle".into(),
        detail: None,
        model: None,
    }
}

/// エンジン状態の短い表示(固定語のみ。detail のような長い文字列はログで確認する)
pub(crate) fn phase_label(state: &EngineState) -> &'static str {
    match state.phase.as_str() {
        "ready" => tr!("Ready", "準備完了", "就绪"),
        "loading" => tr!("Loading…", "読み込み中…", "加载中…"),
        "error" => tr!("Error", "エラー", "错误"),
        _ => tr!("Not loaded", "未読み込み", "未加载"),
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// "VRAM 5.0/10.0GB (アプリ 3.1GB) · RAM 33.2/47.7GB"
pub(crate) fn format_sample(s: &sysmon::SysSample) -> String {
    let gb = |b: u64| b as f64 / GIB;
    let mut parts = Vec::new();
    if s.vram_total > 0 {
        let app = s
            .app_vram
            .map(|a| {
                let a = gb(a);
                trf!(" (app {a:.1}GB)", " (アプリ {a:.1}GB)", " (应用 {a:.1}GB)")
            })
            .unwrap_or_default();
        parts.push(format!("VRAM {:.1}/{:.1}GB{app}", gb(s.vram_used), gb(s.vram_total)));
    }
    if s.ram_total > 0 {
        parts.push(format!("RAM {:.1}/{:.1}GB", gb(s.ram_used), gb(s.ram_total)));
    }
    parts.join(" · ")
}
