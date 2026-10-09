//! バックエンド本体。GUI からのメッセージ(`GuiMessage`)を受け、`BackendMessage` を `Sink` へ送る。
//!
//! - TTS は専用ワーカースレッド + キュー(チャンク逐次合成=疑似ストリーミング)。
//! - マイク+ASR はセッション(`start_session` / `stop_session`)。
//! - GUI スレッドを止めないよう、メッセージは専用のディスパッチスレッドで順番に処理する。
//!
//! 外界(モデル・マイク・VAD・デバイス列挙)は `Platform` を通して注入する。本番は `RealPlatform`、
//! テストは偽物に差し替える。

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use base64::Engine as _;
use rand::Rng as _;
use serde_json::{Map, Value, json};
use sttts_protocol::{AudioDeviceInfo, BackendMessage, EngineState, GuiMessage, ModelInfo, PROTOCOL_VERSION};

use crate::asr::{AsrEngine, Progress};
use crate::chunker::{ChunkOptions, count_mora, split_chunks};
use crate::config::{default_config, default_user_config_path, get, get_bool, get_f64, get_i64, load_user_config, merge_config};
use crate::performance::{AcousticObservation, Delivery, observe, plan_delivery};
use crate::session::{AsrSource, AudioSource, LiveSession, OnBlock, SessionConfig, SessionHost, Timing, Vad};
use crate::sink::Sink;
use crate::tts::{MockTts, TtsEngine, TtsOutput, TtsRequest, check_sampling_overrides};
use crate::util::{lock, ms, now};

pub const IDLE: &str = "idle";
pub const LOADING: &str = "loading";
pub const READY: &str = "ready";
pub const ERROR: &str = "error";

/// 停止→再開の最小間隔(秒)。USB オーディオの短時間反復 open/close はドライバクラッシュを
/// 引き起こした実績あり(BugCheck 0xD1、2026-10 に2度)。
pub const SESSION_RESTART_COOLDOWN_S: f64 = 1.5;

const DEVICE_FATAL_MARKERS: [&str; 9] = [
    "DEVICE_LOST",
    "OUT_OF_RESOURCES",
    "OUT_OF_DEVICE_MEMORY",
    "device lost",
    "Device lost",
    "is lost",
    "Out of Memory",
    "OutOfMemory",
    "GPU からの読み戻しに失敗",
];

/// XPU/GPU のデバイス喪失・資源枯渇か(同一プロセスでは復帰できない種類の失敗)。
pub fn is_device_fatal(err: &str) -> bool {
    DEVICE_FATAL_MARKERS.iter().any(|m| err.contains(m))
}

/// GUI のモデル選択に出すカタログ
pub fn model_catalog() -> Vec<ModelInfo> {
    vec![ModelInfo {
        id: "v4.1-small-mf".into(),
        label: "Irodori v4.1 Small MeanFlow(高速・会話向け)".into(),
        size: Some("約766M / 4steps".into()),
        note: Some("ストリーミング会話の既定".into()),
    }]
}

// ---------------------------------------------------------------- 外界(注入)

pub type TtsProgress<'a> = &'a dyn Fn(&str);

/// 外界との接点。本番とテストで差し替える。
pub trait Platform: Send + Sync {
    fn create_tts(&self, cfg: &Value, progress: TtsProgress) -> Result<Arc<dyn TtsEngine>>;
    fn create_asr(&self, cfg: &Value, progress: Progress) -> Result<Arc<dyn AsrEngine>>;
    /// 音声ソース(マイク / WAV)。`input_wavs` が空でなければマイクの代わりに WAV を流す。
    fn open_source(&self, cfg: &Value, input_wavs: &[PathBuf], on_block: OnBlock, on_eof: Option<Box<dyn FnOnce() + Send>>) -> Result<Box<dyn AudioSource>>;
    fn create_vad(&self, cfg: &Value) -> Result<Box<dyn Vad>>;
    fn list_devices(&self) -> (Vec<AudioDeviceInfo>, Vec<AudioDeviceInfo>);
}

pub struct BackendOptions {
    /// 実モデル・実デバイスを使わない(GUI 開発用)
    pub mock: bool,
    /// 実 ASR + モック TTS(ベンチ用)
    pub mock_tts: bool,
    pub output_dir: PathBuf,
    pub save_wavs: bool,
    /// アプリのルート(`data/backend.json` の場所)
    pub root: PathBuf,
    pub config_path: Option<PathBuf>,
    pub load_user_file: bool,
    /// マイクの代わりに流す WAV(ベンチ用)
    pub input_wavs: Vec<PathBuf>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            mock: false,
            mock_tts: false,
            output_dir: PathBuf::from("output"),
            save_wavs: true,
            root: PathBuf::from("."),
            config_path: None,
            load_user_file: true,
            input_wavs: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------- 内部の型

#[derive(Debug, Clone, PartialEq)]
enum SpecStatus {
    Queued,
    Running,
    Done,
    Failed,
    Emitted,
    Discarded,
}

/// 投機的 TTS(確定前の安定 partial から先頭チャンクを先行合成)の状態。
///
/// `request` が設定される(=確定文の先頭チャンクと完全一致して束縛された)まで音声は決して送らない。
struct SpecEntry {
    text: String,
    seed: u64,
    voice_key: Value,
    status: SpecStatus,
    request: Option<u64>,
    result: Option<TtsOutput>,
}

#[derive(Default)]
struct SpecState {
    entries: HashMap<u64, SpecEntry>,
    /// utterance → entry id(最新の発話のみ保持)
    by_utterance: HashMap<u64, u64>,
    /// utterance → (先頭チャンク候補, 連続回数)
    track: HashMap<u64, (String, u32)>,
    next_id: u64,
}

struct TtsJob {
    request: u64,
    chunk: u32,
    text: String,
    caption: Option<String>,
    ref_wavs: Vec<String>,
    seed: Option<u64>,
    /// 投機ジョブの `SpecState::entries` の id
    spec: Option<u64>,
    /// ウォームアップ合成(結果は送らない)
    warmup: bool,
    enqueued: Option<Instant>,
    /// 発話単位の上書き(元の設定は変更しない)
    sampling: Option<Map<String, Value>>,
}

#[derive(Default)]
struct JobQueue {
    jobs: Mutex<(VecDeque<TtsJob>, bool)>,
    cv: Condvar,
}

impl JobQueue {
    fn push(&self, job: TtsJob) {
        lock(&self.jobs).0.push_back(job);
        self.cv.notify_one();
    }
    fn close(&self) {
        lock(&self.jobs).1 = true;
        self.cv.notify_all();
    }
    fn pop(&self) -> Option<TtsJob> {
        let mut g = lock(&self.jobs);
        loop {
            if g.1 {
                return None;
            }
            if let Some(j) = g.0.pop_front() {
                return Some(j);
            }
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
    /// キュー上の未合成チャンクを破棄する(ウォームアップは残す)
    fn drain_non_warmup(&self) {
        lock(&self.jobs).0.retain(|j| j.warmup);
    }
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        lock(&self.jobs).0.is_empty()
    }
    #[cfg(test)]
    fn take_all(&self) -> Vec<TtsJob> {
        lock(&self.jobs).0.drain(..).collect()
    }
}

struct PendingInfo {
    total: u32,
    done: u32,
    cancelled: bool,
    failed: bool,
    accepted: f64,
    speech_end: Option<f64>,
}

#[derive(Default)]
struct Pending {
    map: HashMap<u64, PendingInfo>,
    seq: u64,
}

#[derive(Default)]
struct AppState {
    tts_phase: String,
    tts_fatal: bool,
    tts_detail: Option<String>,
    tts_loaded_model: Option<String>,
    asr_phase: String,
    asr_detail: Option<String>,
    asr_loaded_model: Option<String>,
    mic_running: bool,
}

#[derive(Default)]
struct AsrCache {
    engine: Option<Arc<dyn AsrEngine>>,
    /// `engine` を作ったときの設定キー。現設定と一致するときだけ流用する
    engine_key: Option<Value>,
    preload_key: Option<Value>,
}

/// 並行解析(発話音声の観測)の結果待ち
#[derive(Default)]
struct PerfSlot {
    st: Mutex<(Option<AcousticObservation>, bool)>,
    cv: Condvar,
}

type PerfJob = Box<dyn FnOnce() + Send>;

pub trait SessionHandle: Send + Sync {
    fn stop(&self);
    fn wait_asr_idle(&self, _timeout: Duration) -> bool {
        true
    }
}

impl SessionHandle for LiveSession {
    fn stop(&self) {
        LiveSession::stop(self);
    }
    fn wait_asr_idle(&self, timeout: Duration) -> bool {
        LiveSession::wait_asr_idle(self, timeout)
    }
}

pub(crate) struct Inner {
    pub(crate) sink: Sink,
    platform: Arc<dyn Platform>,
    opts: BackendOptions,
    pub(crate) config: Mutex<Value>,
    state: Mutex<AppState>,
    engine: Mutex<Option<Arc<dyn TtsEngine>>>,
    queue: JobQueue,
    pending: Mutex<Pending>,
    session: Mutex<Option<Arc<dyn SessionHandle>>>,
    utterance_seq: AtomicU64,
    last_session_stop: Mutex<Option<Instant>>,
    spec: Mutex<SpecState>,
    warmup_model: Mutex<Option<String>>,
    asr: Mutex<AsrCache>,
    stop: AtomicBool,
    perf_tx: Mutex<Option<mpsc::Sender<PerfJob>>>,
    perf_pending: Mutex<HashMap<u64, Arc<PerfSlot>>>,
    recent_rates: Mutex<VecDeque<f64>>,
    emotion_missing_reported: AtomicBool,
    wav_counter: AtomicU64,
}

// ---------------------------------------------------------------- Backend

pub struct Backend {
    tx: mpsc::Sender<GuiMessage>,
    dispatcher: Mutex<Option<JoinHandle<()>>>,
    tts_thread: Mutex<Option<JoinHandle<()>>>,
}

impl Backend {
    pub fn start(opts: BackendOptions, platform: Arc<dyn Platform>, sink: Sink) -> Self {
        let inner = Inner::new(opts, platform, sink);
        let (tx, rx) = mpsc::channel::<GuiMessage>();

        let tts_thread = {
            let inner = Arc::clone(&inner);
            std::thread::Builder::new().name("tts-worker".into()).spawn(move || inner.tts_worker()).expect("spawn tts-worker")
        };
        let dispatcher = {
            let inner = Arc::clone(&inner);
            std::thread::Builder::new()
                .name("backend-dispatch".into())
                .spawn(move || {
                    inner.announce();
                    for msg in rx {
                        let shutdown = matches!(msg, GuiMessage::Shutdown);
                        let inner2 = Arc::clone(&inner);
                        // 1 件の失敗でディスパッチを止めない
                        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| inner2.dispatch(msg))).is_err() {
                            inner.sink.error("backend", "内部エラー(メッセージ処理中に panic しました)", true);
                        }
                        if shutdown || inner.stop.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                    inner.teardown();
                })
                .expect("spawn backend-dispatch")
        };
        Self { tx, dispatcher: Mutex::new(Some(dispatcher)), tts_thread: Mutex::new(Some(tts_thread)) }
    }

    /// GUI → バックエンドのメッセージ送信(即座に戻る)。
    pub fn send(&self, msg: GuiMessage) {
        let _ = self.tx.send(msg);
    }

    /// 終了要求 → 完了待ち。マイクと TTS ワーカーを止めてから戻る。
    pub fn shutdown(&self) {
        let _ = self.tx.send(GuiMessage::Shutdown);
        self.join();
    }

    fn join(&self) {
        if let Some(h) = lock(&self.dispatcher).take() {
            let _ = h.join();
        }
        if let Some(h) = lock(&self.tts_thread).take() {
            let _ = h.join();
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = self.tx.send(GuiMessage::Shutdown);
        self.join();
    }
}

// ---------------------------------------------------------------- Inner

impl Inner {
    pub(crate) fn new(opts: BackendOptions, platform: Arc<dyn Platform>, sink: Sink) -> Arc<Self> {
        let mut config = default_config();
        if opts.load_user_file {
            let path = opts.config_path.clone().unwrap_or_else(|| default_user_config_path(&opts.root));
            let warn = |m: &str| sink.warn(m.to_string());
            config = merge_config(&config, &load_user_config(&path, &warn));
        }
        let (perf_tx, perf_rx) = mpsc::channel::<PerfJob>();
        // 並行解析は 1 スレッド(混雑時は古い結果を捨てる)
        std::thread::Builder::new()
            .name("performance".into())
            .spawn(move || {
                for job in perf_rx {
                    job();
                }
            })
            .expect("spawn performance");
        Arc::new(Self {
            sink,
            platform,
            opts,
            config: Mutex::new(config),
            state: Mutex::new(AppState { tts_phase: IDLE.into(), asr_phase: IDLE.into(), ..Default::default() }),
            engine: Mutex::new(None),
            queue: JobQueue::default(),
            pending: Mutex::new(Pending::default()),
            session: Mutex::new(None),
            utterance_seq: AtomicU64::new(0),
            last_session_stop: Mutex::new(None),
            spec: Mutex::new(SpecState::default()),
            warmup_model: Mutex::new(None),
            asr: Mutex::new(AsrCache::default()),
            stop: AtomicBool::new(false),
            perf_tx: Mutex::new(Some(perf_tx)),
            perf_pending: Mutex::new(HashMap::new()),
            recent_rates: Mutex::new(VecDeque::new()),
            emotion_missing_reported: AtomicBool::new(false),
            wav_counter: AtomicU64::new(0),
        })
    }

    fn mock_tts(&self) -> bool {
        self.opts.mock || self.opts.mock_tts
    }

    /// 起動直後の通知(hello / state / devices)
    pub(crate) fn announce(&self) {
        self.sink.send(BackendMessage::Hello {
            protocol: PROTOCOL_VERSION,
            mock: self.opts.mock,
            python: None,
            backend_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            models: model_catalog(),
        });
        self.send_state();
        self.send_devices();
    }

    pub(crate) fn dispatch(self: &Arc<Self>, msg: GuiMessage) {
        match msg {
            GuiMessage::Configure { tts, asr, audio, voice, pipeline } => {
                let to_value = |v: Option<Value>| v;
                let mut patch = Map::new();
                for (k, v) in [
                    ("tts", tts.and_then(|c| serde_json::to_value(c).ok())),
                    ("asr", asr.and_then(|c| serde_json::to_value(c).ok())),
                    ("audio", audio.and_then(|c| serde_json::to_value(c).ok())),
                    ("voice", voice.and_then(|c| serde_json::to_value(c).ok())),
                    ("pipeline", pipeline.and_then(|c| serde_json::to_value(c).ok())),
                ] {
                    if let Some(v) = to_value(v) {
                        patch.insert(k.into(), v);
                    }
                }
                self.configure(&Value::Object(patch));
            }
            GuiMessage::StartSession => self.start_session(),
            GuiMessage::StopSession => self.stop_session(),
            GuiMessage::Speak { text, caption, ref_wavs, seed, tag, delivery } => {
                self.speak(SpeakParams { text, caption, ref_wavs, seed, tag, delivery: delivery.as_ref().map(Delivery::from_info), ..SpeakParams::manual() });
            }
            GuiMessage::CancelSpeak => self.cancel_speak(),
            GuiMessage::Ping { nonce } => self.sink.send(BackendMessage::Pong { nonce }),
            GuiMessage::Shutdown => self.stop.store(true, Ordering::SeqCst),
        }
    }

    // ---------- 出力系 ----------

    pub(crate) fn send_state(&self) {
        let msg = {
            let st = lock(&self.state);
            BackendMessage::State {
                tts: EngineState { phase: st.tts_phase.clone(), detail: st.tts_detail.clone(), model: st.tts_loaded_model.clone() },
                asr: EngineState { phase: st.asr_phase.clone(), detail: st.asr_detail.clone(), model: st.asr_loaded_model.clone() },
                mic_running: st.mic_running,
            }
        };
        self.sink.send(msg);
    }

    fn set_tts(&self, phase: &str, detail: Option<String>, fatal: bool) {
        {
            let mut st = lock(&self.state);
            st.tts_phase = phase.into();
            st.tts_detail = detail;
            st.tts_fatal = fatal;
        }
        self.send_state();
    }

    fn set_asr(&self, phase: &str, detail: Option<String>, model_id: Option<String>) {
        {
            let mut st = lock(&self.state);
            st.asr_phase = phase.into();
            st.asr_detail = detail;
            if model_id.is_some() {
                st.asr_loaded_model = model_id;
            }
        }
        self.send_state();
    }

    /// 入出力デバイス一覧を送る(列挙に失敗しても致命傷にしない)
    fn send_devices(&self) {
        let (inputs, outputs) = self.platform.list_devices();
        self.sink.send(BackendMessage::Devices { inputs, outputs });
    }

    // ---------- 設定 ----------

    pub(crate) fn configure(self: &Arc<Self>, patch: &Value) {
        {
            let mut cfg = lock(&self.config);
            *cfg = merge_config(&cfg, patch);
        }
        let keys: Vec<&String> = patch.as_object().map(|o| o.keys().collect()).unwrap_or_default();
        self.sink.debug(format!("configure 適用: {keys:?}"));
        if patch.get("tts").is_some() {
            self.maybe_schedule_warmup();
        }
        if patch.get("asr").is_some() {
            self.maybe_schedule_asr_preload();
        }
    }

    fn cfg(&self) -> Value {
        lock(&self.config).clone()
    }

    /// `tts.warmup` が有効なら、モデルのロード + 短文合成を TTS ワーカーに先行投入する。
    /// 初回発話時のロード待ち・初回カーネル起動/メモリ確保のコストを先払いする。
    fn maybe_schedule_warmup(&self) {
        let cfg = self.cfg();
        if !get_bool(&cfg, "tts", "warmup", true) {
            return;
        }
        let model = get(&cfg, "tts", "model").as_str().map(str::to_string);
        {
            let mut w = lock(&self.warmup_model);
            if *w == model {
                return;
            }
            w.clone_from(&model);
        }
        self.queue.push(TtsJob {
            request: 0,
            chunk: 0,
            text: String::new(),
            caption: None,
            ref_wavs: Vec::new(),
            seed: Some(0),
            spec: None,
            warmup: true,
            enqueued: None,
            sampling: None,
        });
    }

    /// エンジンを作り直すべき ASR 設定の組。Gemini はキー/モードの変更でも作り直す。
    fn asr_config_key(&self) -> Value {
        let cfg = self.cfg();
        let a = |k: &str| get(&cfg, "asr", k).clone();
        json!([a("engine"), a("model"), a("nemotron_model_dir"), a("nemotron_chunk_ms"), a("gemini_model"), a("gemini_api_key"), a("gemini_mode")])
    }

    /// `asr.preload` が有効なら ASR エンジンを起動時にロードしてキャッシュする。
    /// 「マイク開始」を押した瞬間から(ロード待ちなく)文字起こしが始まるようにする先払い。
    /// engine 設定が変わったら作り直す。
    fn maybe_schedule_asr_preload(self: &Arc<Self>) {
        let cfg = self.cfg();
        if !get_bool(&cfg, "asr", "preload", true) || self.opts.mock {
            return;
        }
        let key = self.asr_config_key();
        {
            let mut a = lock(&self.asr);
            if a.preload_key.as_ref() == Some(&key) {
                return;
            }
            a.preload_key = Some(key.clone());
        }
        let me = Arc::clone(self);
        std::thread::Builder::new()
            .name("asr-preload".into())
            .spawn(move || me.preload_asr(&key))
            .expect("spawn asr-preload");
    }

    fn preload_asr(self: &Arc<Self>, key: &Value) {
        let cfg = self.cfg();
        let progress = |m: &str| self.set_asr(LOADING, Some(m.to_string()), None);
        let engine = match self.platform.create_asr(&cfg, &progress) {
            Ok(e) => e,
            Err(e) => {
                {
                    let mut a = lock(&self.asr);
                    if a.preload_key.as_ref() != Some(key) {
                        return; // 既に別設定のプリロードへ移っている。古い失敗は報告しない
                    }
                    a.preload_key = None; // 再試験できるように
                }
                self.set_asr(ERROR, Some(format!("ASRプリロード失敗: {e:#}")), None);
                return;
            }
        };
        let (stale, old) = {
            let mut a = lock(&self.asr);
            if a.preload_key.as_ref() != Some(key) {
                // ロード中に設定が変わった(別のプリロードが走っている)。この結果は捨てる
                (Some(Arc::clone(&engine)), None)
            } else {
                let old = a.engine.replace(Arc::clone(&engine));
                a.engine_key = Some(key.clone());
                (None, old)
            }
        };
        if let Some(s) = stale {
            s.unload();
            return;
        }
        if let Some(o) = old {
            o.unload();
        }
        self.set_asr(READY, None, Some(engine.model_id()));
        self.sink.info(format!("ASR プリロード完了: {}", engine.model_id()));
    }

    // ---------- TTS ----------

    fn resolve_voice(&self, caption: Option<&str>, ref_wavs: Option<&[String]>) -> (Option<String>, Vec<String>) {
        let cfg = self.cfg();
        let caption = caption
            .map(str::to_string)
            .or_else(|| get(&cfg, "voice", "caption").as_str().filter(|c| !c.is_empty()).map(str::to_string));
        let refs = match ref_wavs {
            Some(r) => r.to_vec(),
            None => get(&cfg, "voice", "ref_wavs")
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
        };
        (caption, refs)
    }

    /// 投機結果を流用してよいかの判定キー(声・モデル・合成設定が同一であること)。
    fn voice_key(&self, caption: Option<&str>, ref_wavs: &[String]) -> Value {
        let cfg = self.cfg();
        json!([caption, ref_wavs, get(&cfg, "tts", "model"), get(&cfg, "tts", "num_steps"), get(&cfg, "tts", "sampling")])
    }

    fn chunk_options(&self) -> ChunkOptions {
        let cfg = self.cfg();
        let p = |k: &str, d: i64| get_i64(&cfg, "pipeline", k, d).max(0) as usize;
        ChunkOptions {
            min_chars: p("chunk_min_chars", 16),
            first_min_chars: p("first_chunk_min_chars", 1),
            max_chars: p("chunk_max_chars", 80),
            first_mora_min: get_f64(&cfg, "pipeline", "first_chunk_mora_min", 8.0),
            first_mora_max: get_f64(&cfg, "pipeline", "first_chunk_mora_max", 12.0),
        }
    }

    fn split(&self, text: &str) -> Vec<String> {
        split_chunks(text, &self.chunk_options())
    }

    /// リクエスト単位の seed。未指定(ランダム)ならここで1回だけ決めて全チャンクで共有する
    /// (チャンクごとに別 seed だと、参照音声なしの声質がチャンク間で変わるため)。
    fn request_seed(seed: Option<i64>) -> u64 {
        match seed {
            Some(s) => s as u64,
            None => u64::from(rand::rng().random::<u32>() >> 1),
        }
    }

    pub(crate) fn speak(&self, p: SpeakParams) {
        let (dead, detail) = {
            let st = lock(&self.state);
            (st.tts_phase == ERROR && st.tts_fatal, st.tts_detail.clone())
        };
        if dead {
            self.sink.error("tts", detail.unwrap_or_else(|| "音声合成デバイスが停止しています".into()), false);
            return;
        }
        let text = p.text.trim().to_string();
        if text.is_empty() {
            self.sink.error("tts", "空のテキストです", true);
            return;
        }
        let chunks = self.split(&text);
        if chunks.is_empty() {
            self.sink.error("tts", "チャンクに分割できませんでした", true);
            return;
        }

        let (mut caption, ref_wavs) = self.resolve_voice(p.caption.as_deref(), p.ref_wavs.as_deref());
        let delivery = p.delivery.clone();
        if let Some(d) = &delivery {
            caption = d.caption(caption.as_deref());
        }
        let cfg = self.cfg();
        let mut sampling: Map<String, Value> = get(&cfg, "tts", "sampling").as_object().cloned().unwrap_or_default();
        if let Some(scale) = delivery.as_ref().and_then(|d| d.duration_scale) {
            sampling.insert("duration_scale".into(), json!(scale));
        }

        let request = {
            let mut pend = lock(&self.pending);
            pend.seq += 1;
            let request = pend.seq;
            pend.map.insert(
                request,
                PendingInfo { total: chunks.len() as u32, done: 0, cancelled: false, failed: false, accepted: now(), speech_end: p.speech_end },
            );
            request
        };

        self.sink.send(BackendMessage::SpeakAccepted {
            request,
            origin: p.origin.clone(),
            tag: p.tag.clone(),
            utterance: p.utterance,
            speech_end_ms: p.speech_end.map(ms),
            delivery: delivery.as_ref().map(Delivery::summary),
        });
        let mut seed = Self::request_seed(p.seed);

        // 投機的 TTS の束縛: 確定文の先頭チャンクと完全一致し、声・設定が同じときだけ流用する
        let mut first = 0usize;
        if let (Some(utt), None, None) = (p.utterance, p.seed, delivery.as_ref()) {
            let key = self.voice_key(caption.as_deref(), &ref_wavs);
            if let Some((spec_seed, spec_text, ready)) = self.bind_spec(utt, request, &chunks[0], &key) {
                seed = spec_seed;
                first = 1;
                if let Some(result) = ready {
                    self.emit_chunk(request, 0, &spec_text, result, true, None);
                    self.mark_chunk_done(request, false);
                }
            }
        }

        for (i, chunk_text) in chunks.iter().enumerate() {
            if i < first {
                continue;
            }
            let chunk_text = match &delivery {
                Some(d) => d.annotated_text(chunk_text),
                None => chunk_text.clone(),
            };
            self.queue.push(TtsJob {
                request,
                chunk: i as u32,
                text: chunk_text,
                caption: caption.clone(),
                ref_wavs: ref_wavs.clone(),
                seed: Some(seed),
                spec: None,
                warmup: false,
                enqueued: Some(Instant::now()),
                sampling: Some(sampling.clone()),
            });
        }
    }

    // ---------- 投機的 TTS ----------

    fn spec_enabled(&self) -> bool {
        let cfg = self.cfg();
        get_bool(&cfg, "pipeline", "speculative_tts", false) && get_bool(&cfg, "pipeline", "auto_speak", true)
    }

    /// 安定した partial(同じ先頭チャンクが N 回連続)から先頭チャンクを先行合成する。
    fn maybe_speculate(&self, utterance: u64, text: &str) {
        if !self.spec_enabled() || text.trim().is_empty() {
            return;
        }
        let chunks = self.split(text);
        // 先頭チャンクの後ろに続きがある(=切れ目が確定している)場合だけ候補にする
        if chunks.len() < 2 {
            lock(&self.spec).track.remove(&utterance);
            return;
        }
        let candidate = chunks[0].clone();
        let needed = get_i64(&self.cfg(), "pipeline", "speculative_stable_partials", 2).max(1) as u32;
        let (caption, ref_wavs) = self.resolve_voice(None, None);
        let key = self.voice_key(caption.as_deref(), &ref_wavs);
        let (id, seed) = {
            let mut st = lock(&self.spec);
            let count = match st.track.get(&utterance) {
                Some((prev, c)) if *prev == candidate => c + 1,
                _ => 1,
            };
            st.track = HashMap::from([(utterance, (candidate.clone(), count))]); // 古い発話の追跡情報は捨てる
            if count < needed {
                return;
            }
            if let Some(cur_id) = st.by_utterance.get(&utterance) {
                if let Some(cur) = st.entries.get(cur_id) {
                    if cur.text == candidate && cur.voice_key == key && cur.status != SpecStatus::Discarded {
                        return; // 既に同じ候補で投機済み
                    }
                    if cur.request.is_some() {
                        return; // 既に確定に束縛済み
                    }
                }
            }
            for e in st.entries.values_mut() {
                if e.request.is_none() {
                    e.status = SpecStatus::Discarded;
                }
            }
            st.next_id += 1;
            let id = st.next_id;
            let seed = u64::from(rand::rng().random::<u32>() >> 1);
            st.entries.clear();
            st.by_utterance.clear();
            st.entries.insert(
                id,
                SpecEntry { text: candidate.clone(), seed, voice_key: key, status: SpecStatus::Queued, request: None, result: None },
            );
            st.by_utterance.insert(utterance, id);
            (id, seed)
        };
        self.sink.debug(format!("speculative TTS start: utt={utterance} text={candidate:?}"));
        self.queue.push(TtsJob {
            request: 0,
            chunk: 0,
            text: candidate,
            caption,
            ref_wavs,
            seed: Some(seed),
            spec: Some(id),
            warmup: false,
            enqueued: None,
            sampling: None,
        });
    }

    /// 確定文の先頭チャンクに束縛できる投機結果があれば `(seed, text, すでに出来ていた音声)` を返す。
    /// 一致しなければ None で投機結果は破棄(束縛できても、まだ合成中なら音声は None で、完了時に送られる)。
    fn bind_spec(&self, utterance: u64, request: u64, first_text: &str, key: &Value) -> Option<(u64, String, Option<TtsOutput>)> {
        let mut st = lock(&self.spec);
        let id = st.by_utterance.remove(&utterance)?;
        st.track.remove(&utterance);
        let spec = st.entries.get_mut(&id)?;
        let ok = spec.text == first_text
            && spec.voice_key == *key
            && matches!(spec.status, SpecStatus::Queued | SpecStatus::Running | SpecStatus::Done)
            && spec.request.is_none();
        if !ok {
            if spec.status != SpecStatus::Discarded {
                let (a, b) = (spec.text.clone(), first_text.to_string());
                spec.status = SpecStatus::Discarded;
                drop(st);
                self.sink.debug(format!("speculative TTS discarded: {a:?} != {b:?}"));
            }
            return None;
        }
        spec.request = Some(request);
        let (seed, text) = (spec.seed, spec.text.clone());
        let ready = if spec.status == SpecStatus::Done {
            spec.status = SpecStatus::Emitted;
            spec.result.take()
        } else {
            None
        };
        Some((seed, text, ready))
    }

    fn discard_specs(&self) {
        let mut st = lock(&self.spec);
        for e in st.entries.values_mut() {
            e.status = SpecStatus::Discarded;
        }
        st.entries.clear();
        st.by_utterance.clear();
        st.track.clear();
    }

    fn save_wav(&self, wav: &[u8]) -> Option<String> {
        if !self.opts.save_wavs {
            return None;
        }
        let dir = &self.opts.output_dir;
        let write = || -> std::io::Result<PathBuf> {
            std::fs::create_dir_all(dir)?;
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let n = self.wav_counter.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("tts_{secs}_{n:04}.wav"));
            std::fs::write(&path, wav)?;
            Ok(path)
        };
        match write() {
            Ok(p) => Some(p.to_string_lossy().into_owned()),
            Err(e) => {
                self.sink.warn(format!("failed to save wav: {e}"));
                None
            }
        }
    }

    /// 受付済みの全リクエストを取り消す。
    ///
    /// - キュー上の未合成チャンクを破棄
    /// - 合成中のチャンク(Irodori は途中中断できない)は完了後に結果を捨てる
    ///   (`emit_chunk` が cancelled / 削除済みリクエストを送らない)
    /// - 投機的 TTS も破棄
    pub(crate) fn cancel_speak(&self) {
        // ウォームアップは取り消し対象外(モデルロードを無駄にしない)
        self.queue.drain_non_warmup();
        self.discard_specs();
        {
            let mut pend = lock(&self.pending);
            let mut ids: Vec<u64> = pend.map.keys().copied().collect();
            ids.sort_unstable();
            for req in ids {
                if let Some(info) = pend.map.get_mut(&req) {
                    info.cancelled = true;
                }
                self.send_speak_done_locked(&mut pend, req);
            }
        }
        self.sink.info("発話をキャンセルしました(合成中のチャンクは完了後に破棄)");
    }

    /// `pending` ロック保持中に呼ぶこと。done==total か cancelled で speak_done を送る。
    fn send_speak_done_locked(&self, pend: &mut Pending, request: u64) {
        let Some(info) = pend.map.get(&request) else { return };
        if info.done >= info.total || info.cancelled {
            let info = pend.map.remove(&request).expect("checked above");
            self.sink.send(BackendMessage::SpeakDone { request, chunks: info.done, cancelled: info.cancelled, failed: info.failed });
        }
    }

    /// 1チャンク分の tts_chunk_start / tts_audio / tts_chunk_done を送る。
    ///
    /// キャンセル済み(または speak_done 送信済み)のリクエストには何も送らない。
    /// 判定と送信を `pending` ロック内で行い、cancel_speak との競合で speak_done 後に音声が届くことを防ぐ。
    fn emit_chunk(&self, request: u64, chunk: u32, text: &str, result: TtsOutput, speculative: bool, queue_wait_ms: Option<u64>) {
        {
            let pend = lock(&self.pending);
            match pend.map.get(&request) {
                Some(i) if !i.cancelled => {}
                _ => {
                    self.sink.debug(format!("drop audio of cancelled request {request} chunk {chunk}"));
                    return;
                }
            }
        }
        let path = self.save_wav(&result.wav);
        let wav_base64 = base64::engine::general_purpose::STANDARD.encode(&result.wav);
        let pend = lock(&self.pending);
        let Some(info) = pend.map.get(&request).filter(|i| !i.cancelled) else {
            self.sink.debug(format!("drop audio of cancelled request {request} chunk {chunk}"));
            return;
        };
        self.sink.send(BackendMessage::TtsChunkStart { request, chunk, text: text.to_string() });
        let (mut first_chunk_ms, mut e2e_ms) = (None, None);
        if chunk == 0 {
            let t = now();
            first_chunk_ms = Some(((t - info.accepted) * 1000.0) as u64);
            // 発話終了 → 先頭チャンク送出(GUI 側で受信→再生キュー投入分を加算して表示)
            e2e_ms = info.speech_end.map(|se| ((t - se) * 1000.0) as u64);
        }
        self.sink.send(BackendMessage::TtsAudio {
            request,
            chunk,
            wav_base64,
            sample_rate: result.sample_rate,
            duration_ms: result.duration_ms,
            gen_ms: result.gen_ms,
            path,
            seed: result.used_seed,
            first_chunk: chunk == 0,
            speculative,
            rtf: (result.duration_ms > 0).then(|| ((result.gen_ms as f64 / result.duration_ms as f64) * 1000.0).round() / 1000.0),
            queue_wait_ms,
            first_chunk_ms,
            e2e_ms,
            stages: result.stages,
        });
        self.sink.send(BackendMessage::TtsChunkDone { request, chunk, gen_ms: result.gen_ms });
    }

    fn mark_chunk_done(&self, request: u64, failed: bool) {
        let mut pend = lock(&self.pending);
        if let Some(info) = pend.map.get_mut(&request) {
            info.done += 1;
            if failed {
                info.failed = true;
            }
            self.send_speak_done_locked(&mut pend, request);
        }
    }

    /// 呼び出しは TTS ワーカースレッドからのみ。必要ならモデルをロードして返す。
    fn ensure_engine(&self) -> Result<Arc<dyn TtsEngine>> {
        let cfg = self.cfg();
        let model = get(&cfg, "tts", "model").as_str().unwrap_or("v4.1-small-mf").to_string();
        {
            let ready = {
                let st = lock(&self.state);
                st.tts_phase == READY && st.tts_loaded_model.as_deref() == Some(model.as_str())
            };
            if ready {
                if let Some(e) = lock(&self.engine).clone() {
                    return Ok(e);
                }
            }
        }
        if let Some(old) = lock(&self.engine).take() {
            self.sink.info(format!("旧モデルを解放: {}", old.model_id()));
        }
        self.set_tts(LOADING, Some(format!("loading {model}")), false);
        let progress = |m: &str| self.set_tts(LOADING, Some(m.to_string()), false);
        let created = if self.mock_tts() {
            std::thread::sleep(Duration::from_millis(300)); // ロードを模倣
            Ok(Arc::new(MockTts::new(
                &model,
                get_f64(&cfg, "tts", "mock_delay_ms", 50.0),
                get_f64(&cfg, "tts", "mock_rtf", 0.0),
            )) as Arc<dyn TtsEngine>)
        } else {
            self.platform.create_tts(&cfg, &progress)
        };
        match created {
            Ok(engine) => {
                *lock(&self.engine) = Some(Arc::clone(&engine));
                lock(&self.state).tts_loaded_model = Some(model);
                self.set_tts(READY, None, false);
                Ok(engine)
            }
            Err(e) => {
                // 失敗を LOADING のまま放置しない(次の発話で再試行される)
                self.set_tts(ERROR, Some(format!("TTS のロードに失敗: {e:#}")), false);
                Err(e)
            }
        }
    }

    fn synthesize(&self, job: &TtsJob) -> Result<TtsOutput> {
        let engine = self.ensure_engine()?;
        let cfg = self.cfg();
        let default_sampling = get(&cfg, "tts", "sampling").as_object().cloned().unwrap_or_default();
        let sampling = job.sampling.as_ref().unwrap_or(&default_sampling);
        check_sampling_overrides(sampling)?;
        engine.synthesize(&TtsRequest { text: &job.text, caption: job.caption.as_deref(), ref_wavs: &job.ref_wavs, seed: job.seed, sampling })
    }

    fn run_spec_job(&self, job: &TtsJob) {
        let Some(id) = job.spec else { return };
        {
            let mut st = lock(&self.spec);
            match st.entries.get_mut(&id) {
                Some(e) if e.status != SpecStatus::Discarded => e.status = SpecStatus::Running,
                _ => return,
            }
        }
        let result = match self.synthesize(job) {
            Ok(r) => Some(r),
            Err(e) => {
                self.sink.warn(format!("speculative synthesis failed: {e:#}"));
                None
            }
        };
        let (request, text) = {
            let mut st = lock(&self.spec);
            let Some(spec) = st.entries.get_mut(&id) else { return };
            if spec.status == SpecStatus::Discarded {
                return;
            }
            match spec.request {
                None => {
                    spec.status = if result.is_some() { SpecStatus::Done } else { SpecStatus::Failed };
                    spec.result = result;
                    return;
                }
                Some(request) => {
                    spec.status = SpecStatus::Emitted;
                    (request, spec.text.clone())
                }
            }
        };
        // 確定文に束縛済み → 先頭チャンクとして送る(後続チャンクは FIFO でこの後ろ)
        match result {
            None => {
                // 投機合成が失敗した場合は同じテキストを通常合成し直す(順序を保つためここで同期実行)
                self.run_job(&TtsJob {
                    request,
                    chunk: 0,
                    text,
                    caption: job.caption.clone(),
                    ref_wavs: job.ref_wavs.clone(),
                    seed: job.seed,
                    spec: None,
                    warmup: false,
                    enqueued: None,
                    sampling: None,
                });
            }
            Some(r) => {
                self.emit_chunk(request, 0, &text, r, true, None);
                self.mark_chunk_done(request, false);
            }
        }
    }

    fn run_job(&self, job: &TtsJob) {
        let cancelled = {
            let pend = lock(&self.pending);
            pend.map.get(&job.request).is_none_or(|i| i.cancelled)
        };
        if cancelled {
            return; // キャンセル済みリクエストの残チャンクはスキップ
        }
        let queue_wait_ms = job.enqueued.map(|t| t.elapsed().as_millis() as u64);
        match self.synthesize(job) {
            Ok(result) => {
                self.emit_chunk(job.request, job.chunk, &job.text, result, false, queue_wait_ms);
                self.mark_chunk_done(job.request, false);
            }
            Err(e) => {
                let text = format!("{e:#}");
                self.sink.warn(format!("synthesis failed: {text}"));
                if is_device_fatal(&text) {
                    // デバイス喪失は同一プロセス内では復帰しない。状態を ERROR にして GUI へ見せ、
                    // 以降の発話は speak() で即座に拒否する(死んだデバイスへ投げ続けない)。
                    let msg = format!("音声合成デバイスが停止しました。アプリを再起動してください: {text}");
                    self.set_tts(ERROR, Some(msg.clone()), true);
                    self.sink.error("tts", msg, false);
                } else {
                    self.sink.error("tts", format!("合成失敗: {text}"), true);
                }
                self.mark_chunk_done(job.request, true);
            }
        }
    }

    fn run_warmup(&self) {
        let t0 = Instant::now();
        let result = self.ensure_engine().and_then(|e| e.warmup().map(|()| e));
        match result {
            Ok(_) => self.sink.info(format!("ウォームアップ完了(ロード込み {} ms)", t0.elapsed().as_millis())),
            Err(e) => self.sink.warn(format!("ウォームアップ失敗: {e:#}")),
        }
    }

    fn tts_worker(&self) {
        while let Some(job) = self.queue.pop() {
            if self.stop.load(Ordering::SeqCst) {
                break;
            }
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if job.warmup {
                    // 投入後にモデルが変わったウォームアップは捨てる
                    let model = get(&self.cfg(), "tts", "model").as_str().map(str::to_string);
                    if *lock(&self.warmup_model) == model {
                        self.run_warmup();
                    }
                } else if job.spec.is_some() {
                    self.run_spec_job(&job);
                } else {
                    self.run_job(&job);
                }
            }));
            if r.is_err() {
                self.sink.error("tts", "TTS ワーカーで内部エラーが発生しました", true);
                if !job.warmup && job.spec.is_none() {
                    self.mark_chunk_done(job.request, true);
                }
            }
        }
    }

    // ---------- セッション(マイク+ASR) ----------

    pub(crate) fn start_session(self: &Arc<Self>) {
        if lock(&self.session).is_some() {
            self.sink.warn("セッションは既に実行中です");
            return;
        }
        // 停止直後の再開ガード(デバイスの短時間反復 open/close は USB オーディオ
        // ドライバクラッシュ(BugCheck 0xD1 を 2026-10 に2度発生)を引き起こした)。
        if let Some(stopped) = *lock(&self.last_session_stop) {
            let since = stopped.elapsed().as_secs_f64();
            if since < SESSION_RESTART_COOLDOWN_S {
                self.sink.error(
                    "asr",
                    format!(
                        "停止直後の再開は{SESSION_RESTART_COOLDOWN_S}秒以内は受け付けません(残り{:.1}秒。デバイス保護のため)",
                        SESSION_RESTART_COOLDOWN_S - since
                    ),
                    true,
                );
                return;
            }
        }
        let seq = self.utterance_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let cfg = self.cfg();
        let host: Arc<dyn SessionHost> = Arc::clone(self) as Arc<dyn SessionHost>;
        let (session, preloaded): (Arc<dyn SessionHandle>, bool) = if self.opts.mock {
            (crate::mock::MockSession::start(host), true)
        } else {
            let key = self.asr_config_key();
            // プリロード済み ASR があれば注入(マイク開始→即文字起こし可能)。
            // 現設定で作ったものに限る(切替後のプリロード失敗/ロード中に旧エンジンを使わない)
            let preloaded = {
                let a = lock(&self.asr);
                if a.engine_key.as_ref() == Some(&key) { a.engine.clone() } else { None }
            };
            let was_preloaded = preloaded.is_some();
            let asr = match preloaded {
                Some(e) => AsrSource::Preloaded(e),
                None => {
                    let me = Arc::clone(self);
                    AsrSource::Load(Box::new(move || {
                        let cfg = me.cfg();
                        let progress = |m: &str| me.set_asr(LOADING, Some(m.to_string()), None);
                        me.platform.create_asr(&cfg, &progress)
                    }))
                }
            };
            let interval = get_i64(&cfg, "asr", "partial_interval_ms", 0).max(0) as u64;
            let wavs = self.opts.input_wavs.clone();
            let source_factory = {
                let me = Arc::clone(self);
                let cfg = cfg.clone();
                Box::new(move |on_block: OnBlock| {
                    let on_eof: Option<Box<dyn FnOnce() + Send>> = if me.opts.input_wavs.is_empty() {
                        None
                    } else {
                        let me2 = Arc::clone(&me);
                        Some(Box::new(move || {
                            // 投入済み音声の ASR がすべて終わってから通知する(ベンチの終了判定用)
                            let s = lock(&me2.session).clone();
                            if let Some(s) = s {
                                s.wait_asr_idle(Duration::from_secs(120));
                            }
                            me2.sink.info("input_eof");
                        }))
                    };
                    me.platform.open_source(&cfg, &wavs, on_block, on_eof)
                })
            };
            let vad_factory = {
                let me = Arc::clone(self);
                let cfg = cfg.clone();
                Box::new(move || me.platform.create_vad(&cfg))
            };
            let s = LiveSession::start(host, SessionConfig { partial_interval_ms: interval, utterance_start: seq * 1_000_000 }, asr, source_factory, vad_factory);
            (Arc::new(s), was_preloaded)
        };
        *lock(&self.session) = Some(session);
        lock(&self.state).mic_running = true;
        let label = {
            let engine = get(&cfg, "asr", "engine").as_str().unwrap_or("kotoba");
            engine.to_string()
        };
        if preloaded {
            self.set_asr(READY, None, None);
        } else {
            self.set_asr(LOADING, Some(format!("loading {label}")), None);
        }
        self.sink.info("マイクセッション開始");
    }

    pub(crate) fn stop_session(&self) {
        let Some(session) = lock(&self.session).take() else { return };
        session.stop();
        self.cancel_performance();
        *lock(&self.last_session_stop) = Some(Instant::now());
        let (loaded, ()) = {
            let mut st = lock(&self.state);
            st.mic_running = false;
            (st.asr_loaded_model.is_some(), ())
        };
        self.set_asr(if loaded { READY } else { IDLE }, None, None);
        self.sink.info("マイクセッション停止");
    }

    fn cancel_performance(&self) {
        for (_, slot) in lock(&self.perf_pending).drain() {
            lock(&slot.st).1 = true;
            slot.cv.notify_all();
        }
    }

    // ---------- 発話表現(並行解析) ----------

    fn performance_enabled(&self) -> bool {
        get_bool(&self.cfg(), "pipeline", "performance_enabled", true) && !self.spec_enabled()
    }

    /// VAD 確定時に呼ぶ。ASR と並列で解析し、VAD スレッドでは計算しない。
    fn submit_performance(&self, utterance: u64, audio: &[f32]) {
        if !self.performance_enabled() {
            return;
        }
        let slot = Arc::new(PerfSlot::default());
        {
            let mut pend = lock(&self.perf_pending);
            // 混雑時は古い未処理結果を破棄し、マイクや ASR を待たせない。
            pend.retain(|&u, s| {
                let keep = u + 2 >= utterance;
                if !keep {
                    lock(&s.st).1 = true;
                    s.cv.notify_all();
                }
                keep
            });
            pend.insert(utterance, Arc::clone(&slot));
        }
        let audio = audio.to_vec();
        let job: PerfJob = Box::new(move || {
            if lock(&slot.st).1 {
                return; // 取り消し済み
            }
            let obs = observe(&audio, "", 16000);
            let mut st = lock(&slot.st);
            st.0 = Some(obs);
            slot.cv.notify_all();
        });
        if let Some(tx) = lock(&self.perf_tx).as_ref() {
            let _ = tx.send(job);
        }
    }

    /// ASR 確定後、並行解析の完了を待つ(上限 `performance_wait_ms`)。間に合わなければ明瞭読み上げ。
    fn delivery_for(&self, utterance: u64, text: &str) -> (Option<Delivery>, Option<AcousticObservation>) {
        if !self.performance_enabled() {
            return (None, None);
        }
        let Some(slot) = lock(&self.perf_pending).remove(&utterance) else {
            return (None, None);
        };
        let wait = Duration::from_millis(get_i64(&self.cfg(), "pipeline", "performance_wait_ms", 150).max(0) as u64);
        let obs = {
            let st = lock(&slot.st);
            let (mut st, _) = slot.cv.wait_timeout_while(st, wait, |s| s.0.is_none() && !s.1).unwrap_or_else(|e| e.into_inner());
            if st.0.is_none() {
                st.1 = true; // 間に合わなかった解析は捨てる
                return (None, None);
            }
            st.0.take()
        };
        let Some(obs) = obs else { return (None, None) };
        let rate = (obs.active_ms >= 400).then(|| count_mora(text) / (obs.active_ms as f64 / 1000.0));
        let obs = AcousticObservation { mora_per_s: rate, ..obs };
        let baseline = {
            let r = lock(&self.recent_rates);
            (r.len() >= 3).then(|| {
                let mut v: Vec<f64> = r.iter().copied().collect();
                v.sort_by(f64::total_cmp);
                let mid = v.len() / 2;
                if v.len() % 2 == 1 { v[mid] } else { (v[mid - 1] + v[mid]) / 2.0 }
            })
        };
        if get(&self.cfg(), "pipeline", "emotion_engine").as_str() == Some("emotion2vec") && !self.emotion_missing_reported.swap(true, Ordering::SeqCst) {
            self.sink.error("performance", "emotion2vec は Rust 版では未対応です(pipeline.emotion_engine を none にしてください)", true);
        }
        let delivery = plan_delivery(&obs, None, baseline);
        if let Some(rate) = rate.filter(|_| obs.active_ms >= 800) {
            let mut r = lock(&self.recent_rates);
            if r.len() == 8 {
                r.pop_front();
            }
            r.push_back(rate);
        }
        (Some(delivery), Some(obs))
    }

    // ---------- 終了 ----------

    pub(crate) fn teardown(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let session = lock(&self.session).take();
        if let Some(s) = session {
            s.stop();
        }
        self.cancel_performance();
        *lock(&self.perf_tx) = None;
        self.queue.close();
        // 合成中の 1 チャンクは中断できないため、ワーカーの完了は呼び出し側(Backend::join)が待つ。
        // 未完了リクエスト(未合成チャンクは破棄)に speak_done を返してプロトコルを閉じる
        {
            let mut pend = lock(&self.pending);
            let mut ids: Vec<u64> = pend.map.keys().copied().collect();
            ids.sort_unstable();
            for req in ids {
                if let Some(info) = pend.map.get_mut(&req) {
                    info.cancelled = true;
                }
                self.send_speak_done_locked(&mut pend, req);
            }
        }
        let asr = {
            let mut a = lock(&self.asr);
            a.engine_key = None;
            a.preload_key = None;
            a.engine.take()
        };
        if let Some(e) = asr {
            e.unload();
        }
        *lock(&self.engine) = None;
        self.sink.info("backend stopped");
    }
}

/// `speak` の引数(GUI からの手動発話と、ASR 確定からの自動発話で共通)
pub(crate) struct SpeakParams {
    pub text: String,
    pub caption: Option<String>,
    pub ref_wavs: Option<Vec<String>>,
    pub seed: Option<i64>,
    pub tag: Option<String>,
    pub delivery: Option<Delivery>,
    pub origin: String,
    pub utterance: Option<u64>,
    pub speech_end: Option<f64>,
}

impl SpeakParams {
    pub fn manual() -> Self {
        Self { text: String::new(), caption: None, ref_wavs: None, seed: None, tag: None, delivery: None, origin: "manual".into(), utterance: None, speech_end: None }
    }
}

// ---------------------------------------------------------------- セッション → アプリ

impl SessionHost for Inner {
    fn set_asr_loading(&self, message: &str) {
        self.set_asr(LOADING, Some(message.to_string()), None);
    }

    fn on_asr_model_ready(&self, model_id: &str) {
        self.set_asr(READY, None, Some(model_id.to_string()));
    }

    fn on_asr_error(&self, message: &str) {
        self.set_asr(ERROR, Some(message.to_string()), None);
        self.sink.error("asr", message, true);
    }

    fn on_mic_level(&self, rms: f32, db: f32) {
        self.sink.send(BackendMessage::MicLevel { rms, db });
    }

    fn on_utterance_audio(&self, utterance: u64, audio: &[f32]) {
        self.submit_performance(utterance, audio);
    }

    fn on_asr_partial(&self, utterance: u64, text: &str, asr_ms: Option<u64>) {
        self.sink.send(BackendMessage::AsrPartial { utterance, text: text.to_string(), asr_ms });
        self.maybe_speculate(utterance, text);
    }

    fn on_asr_final(&self, utterance: u64, text: &str, timing: Timing) {
        let (delivery, observation) = if text.trim().is_empty() {
            if let Some(slot) = lock(&self.perf_pending).remove(&utterance) {
                lock(&slot.st).1 = true;
            }
            (None, None)
        } else {
            self.delivery_for(utterance, text)
        };
        self.sink.send(BackendMessage::AsrFinal {
            utterance,
            text: text.to_string(),
            // 計測用(エンジンの単調時計基準 ms)
            speech_end_ms: timing.speech_end.map(ms),
            vad_wait_ms: match (timing.speech_end, timing.vad_end) {
                (Some(se), Some(ve)) if se != 0.0 && ve != 0.0 => Some(((ve - se) * 1000.0) as u64),
                _ => None,
            },
            asr_ms: timing.asr_ms,
            audio_ms: timing.audio_ms,
            delivery: delivery.as_ref().map(Delivery::summary),
            pause_ms: observation.map(|o| o.pause_ms),
        });
        if get_bool(&self.cfg(), "pipeline", "auto_speak", true) && !text.trim().is_empty() {
            self.speak(SpeakParams {
                text: text.to_string(),
                origin: "auto".into(),
                utterance: Some(utterance),
                speech_end: timing.speech_end,
                delivery,
                ..SpeakParams::manual()
            });
        } else {
            self.discard_specs();
        }
    }
}

#[cfg(test)]
mod tests;
