//! Gemini Live API(`gemini-3.5-transcribe-live`)によるクラウド ASR の Rust 実装。
//!
//! 元は Python 版 `asr_gemini.py`(google-genai SDK。git 履歴の backend/ にある)。
//! SDK が WebSocket(BidiGenerateContent)で実際に送受信する JSON を再現している。
//!
//! 統合形態: ローカル VAD の発話開始で発話専用 Live セッションを開き、activityStart →
//! 録音中から 100ms 単位で音声 → 終了で activityEnd を送って確定を待つ。
//! サーバ側自動 VAD は無効化する(有効のままだと、発話後の無音でサーバが先に確定を返し、
//! その後の確定待ちが毎回タイムアウトする)。
//!
//! スレッド構成: 発話ごとに 1 スレッド。送信(キューの排出)と受信(短い読み取りタイムアウト)を
//! 同じスレッドで交互に回す。VAD 側の feed/end/abort は待機しない(キューに積むだけ)。
//!
//! API キーはログ・エラー文に出さない(認証は HTTP ヘッダ `x-goog-api-key` のみ)。

use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::{TcpStream, ToSocketAddrs};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use base64::Engine as _;
use serde_json::{Value, json};
use tungstenite::client::IntoClientRequest;
use tungstenite::http::HeaderValue;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Connector, Message, WebSocket};

pub const DEFAULT_MODEL: &str = "gemini-3.5-transcribe-live";
/// SDK の既定のベース URL(`_websocket_base_url`)。
const DEFAULT_ENDPOINT: &str = "wss://generativelanguage.googleapis.com";
/// SDK の api_version 既定値。
const API_VERSION: &str = "v1beta";
const AUDIO_MIME: &str = "audio/pcm;rate=16000";
/// 100ms 分(16kHz × 2 byte)。これ以上溜まったら即送る。
const CHUNK_BYTES: usize = 3200;
const CHUNK_INTERVAL: Duration = Duration::from_millis(100);
/// 受信が空のときの待ち。送信キューの排出・中断の応答性を決める。
const READ_POLL: Duration = Duration::from_millis(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// activityEnd 後、サーバが黙ったままこれだけ経ったら、受信済みのテキストで確定する。
/// 実 API は確定・完了シグナルを返さないことがあり、`timeout_s` まで待つと後続の発話の確定も詰まる。
const SETTLE_AFTER_END: Duration = Duration::from_millis(2500);

/// Python と同じ文言(GUI にそのまま出る)。
const NO_KEY_MESSAGE: &str = "Gemini API キーがありません。GUI の「キー」欄に AI Studio で発行したキーを\
入力してください(data/backend.json の asr.gemini_api_key / 環境変数 GEMINI_API_KEY でも可)";

#[derive(Debug, Clone)]
pub struct GeminiOptions {
    pub model: String,
    pub api_key: Option<String>,
    /// "ja" のような短いコードは BCP-47("ja-JP")に揃える。
    pub language: String,
    /// "VERBATIM" | "SMART"(それ以外は VERBATIM)。
    pub mode: String,
    pub timeout_s: f64,
    /// テスト用。`ws://127.0.0.1:port` のようなベース URL を差し替える。
    pub endpoint: Option<String>,
}

impl Default for GeminiOptions {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.into(),
            api_key: None,
            language: "ja-JP".into(),
            mode: "VERBATIM".into(),
            timeout_s: 20.0,
            endpoint: None,
        }
    }
}

/// config のキー > GEMINI_API_KEY > GOOGLE_API_KEY の順に解決する(空文字は無いものとして扱う)。
pub fn resolve_api_key(configured: Option<&str>) -> Option<String> {
    resolve_api_key_with(configured, |k| std::env::var(k).ok())
}

fn resolve_api_key_with(
    configured: Option<&str>,
    env: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let non_empty = |s: Option<String>| s.filter(|s| !s.is_empty());
    non_empty(configured.map(str::to_owned))
        .or_else(|| non_empty(env("GEMINI_API_KEY")))
        .or_else(|| non_empty(env("GOOGLE_API_KEY")))
}

fn normalize_language(language: &str) -> String {
    if language.contains('-') || language != "ja" {
        language.to_owned()
    } else {
        "ja-JP".to_owned()
    }
}

/// SDK(`live.connect`)が最初に送る setup メッセージ。
/// 入れ子の `inputAudioTranscription` / `realtimeInputConfig` の中身が snake_case なのは SDK の
/// 実出力どおり(pydantic のダンプがそのまま載る)。
fn setup_message(model: &str, language: &str, mode: &str) -> Value {
    json!({
        "setup": {
            "model": format!("models/{model}"),
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": {"language_codes": [language], "mode": mode},
            "realtimeInputConfig": {"automatic_activity_detection": {"disabled": true}},
        }
    })
}

fn audio_message(pcm: &[u8]) -> Value {
    json!({"realtime_input": {"audio": {
        "data": base64::engine::general_purpose::STANDARD.encode(pcm),
        "mime_type": AUDIO_MIME,
    }}})
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// f32(16kHz)→ PCM16 LE。numpy の `astype("<i2")` と同じく 0 方向へ切り捨てる。
fn f32_to_pcm16(frame: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.len() * 2);
    for &s in frame {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub type PartialCallback = Arc<dyn Fn(u64, &str) + Send + Sync>;

pub struct GeminiLive {
    model: String,
    api_key: String,
    language: String,
    mode: String,
    timeout: Duration,
    endpoint: String,
    streams: Mutex<HashMap<u64, Arc<Utterance>>>,
}

impl std::fmt::Debug for GeminiLive {
    // api_key を出さない
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiLive")
            .field("model", &self.model)
            .field("language", &self.language)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl GeminiLive {
    pub fn load(opts: GeminiOptions, progress: &dyn Fn(&str)) -> Result<Self> {
        let Some(api_key) = resolve_api_key(opts.api_key.as_deref()) else {
            bail!("{NO_KEY_MESSAGE}");
        };
        let model = if opts.model.is_empty() { DEFAULT_MODEL.to_owned() } else { opts.model };
        let mode = if opts.mode.to_uppercase() == "SMART" { "SMART" } else { "VERBATIM" }.to_owned();
        progress(&format!("Gemini 接続準備: {model} ({mode})"));
        let language = normalize_language(&opts.language);
        let endpoint = opts
            .endpoint
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned())
            .trim_end_matches('/')
            .to_owned();
        progress(&format!("ASR準備完了: Gemini ({mode})"));
        Ok(Self {
            model,
            api_key,
            language,
            mode,
            timeout: Duration::from_secs_f64(opts.timeout_s.max(0.0)),
            endpoint,
            streams: Mutex::new(HashMap::new()),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model
    }

    /// 発話専用の Live セッションを開始する(接続は別スレッド。待機しない)。
    pub fn begin_stream(&self, utterance: u64, on_partial: PartialCallback) -> Result<()> {
        let stream = {
            let mut streams = lock(&self.streams);
            if streams.contains_key(&utterance) {
                bail!("duplicate Gemini utterance {utterance}");
            }
            let stream = Utterance::new(utterance, on_partial);
            streams.insert(utterance, stream.clone());
            stream
        };
        let ctx = SessionCtx {
            url: format!(
                "{}//ws/google.ai.generativelanguage.{API_VERSION}.GenerativeService.BidiGenerateContent",
                self.endpoint
            ),
            api_key: self.api_key.clone(),
            setup: setup_message(&self.model, &self.language, &self.mode),
        };
        let worker = stream.clone();
        let handle = thread::Builder::new()
            .name(format!("gemini-live-{utterance}"))
            .spawn(move || worker.thread_main(ctx))
            .map_err(|e| {
                lock(&self.streams).remove(&utterance);
                anyhow!("Gemini Live スレッドを起動できません: {e}")
            })?;
        *lock(&stream.thread) = Some(handle);
        Ok(())
    }

    /// 16kHz f32 のフレームを積む(待機しない。未知の発話は無視)。
    pub fn feed_stream(&self, utterance: u64, frame: &[f32]) {
        if let Some(s) = self.get(utterance) {
            s.feed(f32_to_pcm16(frame));
        }
    }

    /// 発話終了(activityEnd を送らせる)。待機しない。
    pub fn end_stream(&self, utterance: u64) {
        if let Some(s) = self.get(utterance) {
            s.end();
        }
    }

    pub fn abort_stream(&self, utterance: u64) {
        let s = lock(&self.streams).remove(&utterance);
        if let Some(s) = s {
            s.abort();
        }
    }

    /// 確定テキストを待って返す(最大 `timeout_s`)。
    ///
    /// - ストリーム未開始: エラー
    /// - 完了シグナルが来ないまま時間切れ: 受信済みの確定テキストがあればそれを返し、無ければ
    ///   `Gemini Live transcription timed out`
    /// - 接続失敗・サーバエラー・不正 JSON: `Gemini Live transcription failed: …`
    ///
    /// `audio` は使わない(Python 互換の引数。音声は feed_stream で送信済み)。
    pub fn finish_stream(&self, utterance: u64, _audio: &[f32]) -> Result<String> {
        let Some(stream) = self.get(utterance) else {
            bail!("Gemini stream unavailable for utterance {utterance}");
        };
        stream.end();
        let result = stream.result(self.timeout);
        let mut streams = lock(&self.streams);
        if streams.get(&utterance).is_some_and(|s| Arc::ptr_eq(s, &stream)) {
            streams.remove(&utterance);
        }
        result
    }

    pub fn abort_all_streams(&self) {
        let streams: Vec<_> = lock(&self.streams).drain().map(|(_, s)| s).collect();
        for s in streams {
            s.abort();
        }
    }

    /// 発話全体を一括送信して確定を得る(ストリームを使わない経路)。
    pub fn transcribe_utterance(&self, audio: &[f32]) -> Result<String> {
        // 一括用の発話 ID は衝突しない上位ビットを使う
        let id = u64::MAX - ONESHOT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.begin_stream(id, Arc::new(|_, _| {}))?;
        self.feed_stream(id, audio);
        self.finish_stream(id, audio)
    }

    fn get(&self, utterance: u64) -> Option<Arc<Utterance>> {
        lock(&self.streams).get(&utterance).cloned()
    }
}

static ONESHOT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Drop for GeminiLive {
    fn drop(&mut self) {
        self.abort_all_streams();
    }
}

// ---------------- 1 発話の接続 ----------------

struct SessionCtx {
    url: String,
    api_key: String,
    setup: Value,
}

#[derive(Default)]
struct Queue {
    items: VecDeque<Option<Vec<u8>>>,
    ended: bool,
}

#[derive(Default)]
struct Outcome {
    done: bool,
    text: String,
    /// 最後の途中経過。確定テキストが来なかったときの代わりに使う
    interim: String,
    error: Option<String>,
}

impl Outcome {
    /// 確定テキスト、無ければ最後の途中経過
    fn best_text(&self) -> String {
        let text = self.text.trim();
        if text.is_empty() { self.interim.trim() } else { text }.to_owned()
    }

    /// 完了シグナル無しで確定してよいか: 確定テキストが途中経過に追いついている
    /// (短いうちは、サーバがまだ確定テキストを送っている途中とみなす)。確定テキストが無ければ途中経過で確定する。
    fn settled(&self) -> bool {
        let (text, interim) = (self.text.trim(), self.interim.trim());
        if text.is_empty() {
            return !interim.is_empty();
        }
        text.chars().filter(|c| !c.is_whitespace()).count() >= interim.chars().filter(|c| !c.is_whitespace()).count()
    }
}

struct Utterance {
    id: u64,
    on_partial: PartialCallback,
    queue: Mutex<Queue>,
    aborted: AtomicBool,
    outcome: Mutex<Outcome>,
    done_cv: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Utterance {
    fn new(id: u64, on_partial: PartialCallback) -> Arc<Self> {
        Arc::new(Self {
            id,
            on_partial,
            queue: Mutex::new(Queue::default()),
            aborted: AtomicBool::new(false),
            outcome: Mutex::new(Outcome::default()),
            done_cv: Condvar::new(),
            thread: Mutex::new(None),
        })
    }

    fn feed(&self, raw: Vec<u8>) {
        let mut q = lock(&self.queue);
        if !q.ended && !self.aborted.load(Ordering::SeqCst) {
            q.items.push_back(Some(raw));
        }
    }

    fn end(&self) {
        let mut q = lock(&self.queue);
        if !q.ended {
            q.ended = true;
            q.items.push_back(None);
        }
    }

    fn abort(&self) {
        {
            let mut q = lock(&self.queue);
            self.aborted.store(true, Ordering::SeqCst);
            q.items.clear();
            q.items.push_back(None);
        }
        self.join(Duration::from_secs(5));
    }

    /// スレッド終了を待つ(自分自身からの呼び出しでは待たない。超過したら切り離す)。
    fn join(&self, timeout: Duration) {
        let Some(handle) = lock(&self.thread).take() else { return };
        if handle.thread().id() == thread::current().id() {
            return;
        }
        let until = Instant::now() + timeout;
        while !handle.is_finished() && Instant::now() < until {
            thread::sleep(Duration::from_millis(5));
        }
        if handle.is_finished() {
            let _ = handle.join();
        }
    }

    fn result(&self, timeout: Duration) -> Result<String> {
        let finished = {
            let guard = lock(&self.outcome);
            let (guard, _) = self
                .done_cv
                .wait_timeout_while(guard, timeout, |o| !o.done)
                .unwrap_or_else(|e| e.into_inner());
            guard.done
        };
        if !finished {
            self.abort();
            let text = lock(&self.outcome).best_text();
            if !text.is_empty() {
                // 確定シグナルは来なかったが、受信済みのテキストがある。捨てずに使う
                return Ok(text);
            }
            bail!("Gemini Live transcription timed out");
        }
        self.join(Duration::from_secs(1));
        let o = lock(&self.outcome);
        if let Some(e) = &o.error {
            bail!("Gemini Live transcription failed: {e}");
        }
        Ok(o.best_text())
    }

    fn finish(&self, error: Option<String>) {
        let mut o = lock(&self.outcome);
        if error.is_some() {
            o.error = error;
        }
        o.done = true;
        self.done_cv.notify_all();
    }

    fn thread_main(&self, ctx: SessionCtx) {
        let err = match self.run(&ctx) {
            Ok(()) => None,
            Err(e) => Some(e.to_string()),
        };
        self.finish(err);
    }

    fn run(&self, ctx: &SessionCtx) -> Result<()> {
        if self.aborted.load(Ordering::SeqCst) {
            return Ok(());
        }
        let (mut ws, sock) = connect(ctx)?;
        // setup → 最初の応答(setupComplete)。Python は中身を見ずに読み捨てる(閉じられていれば失敗)
        sock.set_nonblocking(true)?;
        send_json(&mut ws, &ctx.setup)?;
        let setup_deadline = Instant::now() + CONNECT_TIMEOUT;
        loop {
            if self.aborted.load(Ordering::SeqCst) {
                let _ = ws.close(None);
                let _ = ws.flush();
                return Ok(());
            }
            match ws.read() {
                Ok(Message::Close(frame)) => bail!("{}", close_reason(frame.as_ref())),
                Ok(m) if m.is_text() || m.is_binary() => {
                    self.handle_server(&parse_message(&m)?)?;
                    break;
                }
                Ok(_) => {}
                Err(e) if would_block(&e) => {
                    if Instant::now() >= setup_deadline {
                        bail!("setup の応答がありません");
                    }
                    idle(&mut ws)?;
                }
                Err(e) => return Err(ws_err(e)),
            }
        }

        send_json(&mut ws, &json!({"realtime_input": {"activityStart": {}}}))?;
        let mut pending: Vec<u8> = Vec::new();
        let mut last_send = Instant::now();
        let mut sender_done = false;
        // activityEnd 送信後、最後にサーバから何か届いた(または activityEnd を送った)時刻
        let mut quiet_since = Instant::now();

        let result = (|| -> Result<()> {
            loop {
                if self.aborted.load(Ordering::SeqCst) {
                    return Ok(());
                }
                // 送信側: キューを排出して 100ms 単位でまとめて送る
                if !sender_done {
                    let items: Vec<_> = lock(&self.queue).items.drain(..).collect();
                    for item in items {
                        match item {
                            Some(raw) => {
                                pending.extend_from_slice(&raw);
                                if pending.len() >= CHUNK_BYTES {
                                    send_json(&mut ws, &audio_message(&pending))?;
                                    pending.clear();
                                    last_send = Instant::now();
                                }
                            }
                            None => {
                                if self.aborted.load(Ordering::SeqCst) {
                                    return Ok(());
                                }
                                if !pending.is_empty() {
                                    send_json(&mut ws, &audio_message(&pending))?;
                                    pending.clear();
                                }
                                send_json(&mut ws, &json!({"realtime_input": {"activityEnd": {}}}))?;
                                sender_done = true;
                                quiet_since = Instant::now();
                                break;
                            }
                        }
                    }
                    if !sender_done && !pending.is_empty() && last_send.elapsed() >= CHUNK_INTERVAL {
                        send_json(&mut ws, &audio_message(&pending))?;
                        pending.clear();
                        last_send = Instant::now();
                    }
                }
                // 受信側
                match ws.read() {
                    Ok(Message::Close(frame)) => bail!("{}", close_reason(frame.as_ref())),
                    Ok(m) if m.is_text() || m.is_binary() => {
                        quiet_since = Instant::now();
                        let complete = self.handle_server(&parse_message(&m)?)?;
                        if complete && sender_done {
                            return Ok(());
                        }
                    }
                    Ok(_) => {}
                    Err(e) if would_block(&e) => {
                        if sender_done && quiet_since.elapsed() >= SETTLE_AFTER_END && lock(&self.outcome).settled() {
                            return Ok(()); // 完了シグナル待ちで詰まらない
                        }
                        idle(&mut ws)?;
                    }
                    Err(e) => return Err(ws_err(e)),
                }
            }
        })();
        // 正常/異常を問わず切断する(待たない)
        let _ = ws.close(None);
        let _ = ws.flush();
        result
    }

    /// サーバメッセージを処理する。完了(turnComplete / generationComplete)なら true。
    fn handle_server(&self, msg: &Value) -> Result<bool> {
        if let Some(err) = msg.get("error") {
            let text = err.get("message").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| err.to_string());
            bail!("API error: {text}");
        }
        let Some(sc) = msg.get("serverContent") else { return Ok(false) };
        let interim = sc.pointer("/interimInputTranscription/text").and_then(Value::as_str);
        if let Some(text) = interim.filter(|t| !t.is_empty()) {
            lock(&self.outcome).interim = text.to_owned();
            let cb = self.on_partial.clone();
            let id = self.id;
            if catch_unwind(AssertUnwindSafe(|| cb(id, text))).is_err() {
                eprintln!("Gemini interim callback failed");
            }
        }
        if let Some(text) = sc.pointer("/inputTranscription/text").and_then(Value::as_str) {
            let fragment = text.trim();
            if !fragment.is_empty() {
                let mut o = lock(&self.outcome);
                if fragment != o.text {
                    // 累積テキストの延長なら置換、そうでなければ連結(空なら置換と同じ)
                    o.text = if o.text.is_empty() || fragment.starts_with(&o.text) {
                        fragment.to_owned()
                    } else {
                        format!("{} {}", o.text, fragment).trim().to_owned()
                    };
                }
            }
        }
        let flag = |k: &str| sc.get(k).and_then(Value::as_bool).unwrap_or(false);
        Ok(flag("turnComplete") || flag("generationComplete"))
    }
}

fn parse_message(m: &Message) -> Result<Value> {
    let raw: &[u8] = match m {
        Message::Text(t) => t.as_bytes(),
        Message::Binary(b) => b,
        _ => return Ok(json!({})),
    };
    if raw.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_slice(raw).map_err(|_| {
        let s = String::from_utf8_lossy(raw);
        let head: String = s.chars().take(200).collect();
        anyhow!("Failed to parse response: {head:?}")
    })
}

fn close_reason(frame: Option<&tungstenite::protocol::CloseFrame>) -> String {
    match frame {
        Some(f) => format!("connection closed by server: {} {}", u16::from(f.code), f.reason),
        None => "connection closed by server: 1006 Abnormal closure.".to_owned(),
    }
}

fn ws_err(e: tungstenite::Error) -> anyhow::Error {
    match e {
        tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed => {
            anyhow!("connection closed by server: 1006 Abnormal closure.")
        }
        other => anyhow!("{other}"),
    }
}

type Ws = WebSocket<MaybeTlsStream<TcpStream>>;

fn would_block(e: &tungstenite::Error) -> bool {
    matches!(e, tungstenite::Error::Io(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut))
}

/// 非ブロッキングソケットなので、WouldBlock は「キューに積んだ。後で flush される」の意味。
fn send_json(ws: &mut Ws, v: &Value) -> Result<()> {
    match ws.send(Message::text(v.to_string())) {
        Ok(()) => Ok(()),
        Err(e) if would_block(&e) => Ok(()),
        Err(e) => Err(ws_err(e)),
    }
}

/// 受信待ちの間に書き残しを流し、少し眠る(READ_POLL が送信キュー排出・中断の応答性になる)。
fn idle(ws: &mut Ws) -> Result<()> {
    match ws.flush() {
        Ok(()) => {}
        Err(e) if would_block(&e) => {}
        Err(e) => return Err(ws_err(e)),
    }
    thread::sleep(READ_POLL);
    Ok(())
}

fn connect(ctx: &SessionCtx) -> Result<(Ws, TcpStream)> {
    let mut request = ctx.url.as_str().into_client_request().map_err(|e| anyhow!("{e}"))?;
    {
        let h = request.headers_mut();
        h.insert("Content-Type", HeaderValue::from_static("application/json"));
        let mut key = HeaderValue::from_str(&ctx.api_key).map_err(|_| anyhow!("API キーに使えない文字が含まれています"))?;
        key.set_sensitive(true);
        h.insert("x-goog-api-key", key);
        let ua = HeaderValue::from_static(concat!("sttts-gemini/", env!("CARGO_PKG_VERSION")));
        h.insert("user-agent", ua);
    }
    let uri = request.uri().clone();
    let host = uri.host().ok_or_else(|| anyhow!("endpoint にホストがありません"))?.to_owned();
    let tls = uri.scheme_str() == Some("wss");
    let port = uri.port_u16().unwrap_or(if tls { 443 } else { 80 });

    let addr = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| anyhow!("{host} を解決できません: {e}"))?
        .next()
        .ok_or_else(|| anyhow!("{host} を解決できません"))?;
    let stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .map_err(|e| anyhow!("{host}:{port} に接続できません: {e}"))?;
    let _ = stream.set_nodelay(true);
    // ハンドシェイク中はブロッキング(止まったサーバで固まらないよう上限を付ける)
    stream.set_read_timeout(Some(CONNECT_TIMEOUT))?;
    stream.set_write_timeout(Some(CONNECT_TIMEOUT))?;
    let sock = stream.try_clone()?;

    let connector = if tls { Some(Connector::Rustls(rustls_config()?)) } else { None };
    let (ws, _resp) = tungstenite::client_tls_with_config(request, stream, None, connector)
        .map_err(|e| anyhow!("WebSocket ハンドシェイクに失敗: {e}"))?;
    Ok((ws, sock))
}

fn rustls_config() -> Result<Arc<rustls::ClientConfig>> {
    let roots = rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| anyhow!("TLS 設定に失敗: {e}"))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(cfg))
}

#[cfg(test)]
mod tests;
