//! アプリ層のテスト(実モデル・実デバイス不要)。投機的 TTS・キャンセル・計測・事前ロード・クールダウン。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use serde_json::{Value, json};

use super::*;
use crate::asr::MockAsr;
use crate::tts::{MockTts, TtsEngine, TtsOutput, TtsRequest};

fn wait(timeout: Duration, mut pred: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + timeout;
    while Instant::now() < end {
        if pred() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    pred()
}

fn wait3(pred: impl FnMut() -> bool) -> bool {
    wait(Duration::from_secs(3), pred)
}

#[derive(Default)]
struct Rec {
    msgs: Mutex<Vec<Value>>,
}

impl Rec {
    fn of_type(&self, t: &str) -> Vec<Value> {
        self.msgs.lock().unwrap().iter().filter(|m| m["type"] == t).cloned().collect()
    }
    fn texts_of(&self, t: &str, field: &str) -> Vec<String> {
        self.of_type(t).iter().map(|m| m[field].as_str().unwrap_or_default().to_string()).collect()
    }
}

type AsrBuilder = Box<dyn Fn(&Value) -> Result<Arc<dyn AsrEngine>> + Send + Sync>;

struct FakePlatform {
    asr: Mutex<Option<AsrBuilder>>,
}

impl Platform for FakePlatform {
    fn create_tts(&self, _cfg: &Value, _p: TtsProgress) -> Result<Arc<dyn TtsEngine>> {
        bail!("no real tts in tests")
    }
    fn create_asr(&self, cfg: &Value, _p: Progress) -> Result<Arc<dyn AsrEngine>> {
        match self.asr.lock().unwrap().as_ref() {
            Some(f) => f(cfg),
            None => bail!("no asr factory"),
        }
    }
    fn open_source(&self, _c: &Value, _w: &[PathBuf], _b: OnBlock, _e: Option<Box<dyn FnOnce() + Send>>) -> Result<Box<dyn AudioSource>> {
        bail!("no audio in tests")
    }
    fn create_vad(&self, _c: &Value) -> Result<Box<dyn Vad>> {
        bail!("no vad in tests")
    }
    fn list_devices(&self) -> (Vec<AudioDeviceInfo>, Vec<AudioDeviceInfo>) {
        (vec![], vec![])
    }
}

struct Harness {
    app: Arc<Inner>,
    rec: Arc<Rec>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Harness {
    fn new(mock: bool) -> Self {
        let rec = Arc::new(Rec::default());
        let r2 = Arc::clone(&rec);
        let sink = Sink::new(move |m| r2.msgs.lock().unwrap().push(serde_json::to_value(&m).unwrap()));
        let opts = BackendOptions { mock, save_wavs: false, load_user_file: false, ..Default::default() };
        let app = Inner::new(opts, Arc::new(FakePlatform { asr: Mutex::new(None) }), sink);
        Self { app, rec, worker: None }
    }

    /// 与えたエンジンで READY 状態にして TTS ワーカーを起動する
    fn with_engine(mut self, engine: Arc<dyn TtsEngine>) -> Self {
        let model = get(&self.app.cfg(), "tts", "model").as_str().unwrap().to_string();
        *lock(&self.app.engine) = Some(engine);
        {
            let mut st = lock(&self.app.state);
            st.tts_loaded_model = Some(model);
            st.tts_phase = READY.into();
        }
        let app = Arc::clone(&self.app);
        self.worker = Some(std::thread::spawn(move || app.tts_worker()));
        self
    }

    fn set(&self, section: &str, key: &str, value: Value) {
        let mut cfg = lock(&self.app.config);
        cfg[section][key] = value;
    }

    fn speak(&self, text: &str) {
        self.app.speak(SpeakParams { text: text.into(), ..SpeakParams::manual() });
    }

    fn drain_jobs(&self) -> Vec<TtsJob> {
        self.app.queue.take_all()
    }

    fn done(&self) -> Vec<Value> {
        self.rec.of_type("speak_done")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.app.stop.store(true, Ordering::SeqCst);
        self.app.queue.close();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// 合成をゲートで止められる TTS(合成したテキストを記録する)
struct GateTts {
    inner: MockTts,
    texts: Mutex<Vec<String>>,
    gate: (Mutex<bool>, Condvar),
    started: (Mutex<bool>, Condvar),
}

impl GateTts {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: MockTts::new("gate", 50.0, 0.0),
            texts: Mutex::default(),
            gate: (Mutex::new(true), Condvar::new()),
            started: (Mutex::new(false), Condvar::new()),
        })
    }
    fn close_gate(&self) {
        *self.gate.0.lock().unwrap() = false;
    }
    fn open_gate(&self) {
        *self.gate.0.lock().unwrap() = true;
        self.gate.1.notify_all();
    }
    fn wait_started(&self) -> bool {
        let g = self.started.0.lock().unwrap();
        *self.started.1.wait_timeout_while(g, Duration::from_secs(2), |s| !*s).unwrap().0
    }
    fn texts(&self) -> Vec<String> {
        self.texts.lock().unwrap().clone()
    }
}

impl TtsEngine for GateTts {
    fn model_id(&self) -> &str {
        "gate"
    }
    fn synthesize(&self, req: &TtsRequest) -> Result<TtsOutput> {
        self.texts.lock().unwrap().push(req.text.to_string());
        *self.started.0.lock().unwrap() = true;
        self.started.1.notify_all();
        let g = self.gate.0.lock().unwrap();
        let _ = self.gate.1.wait_timeout_while(g, Duration::from_secs(5), |o| !*o).unwrap();
        self.inner.synthesize(req)
    }
}

// ---------------------------------------------------------------- speak(seed / チャンク設定 / デバイス喪失)

fn seeds(jobs: &[TtsJob]) -> Vec<Option<u64>> {
    jobs.iter().map(|j| j.seed).collect()
}

#[test]
fn random_seed_is_shared_by_all_chunks_of_a_request() {
    let h = Harness::new(true);
    h.speak("こんにちは、今日はいい天気ですね。明日も晴れるといいですね。");
    let jobs = h.drain_jobs();
    assert!(jobs.len() >= 2);
    let mut s = seeds(&jobs);
    s.dedup();
    assert_eq!(s.len(), 1);
    assert!(s[0].is_some());
}

#[test]
fn explicit_seed_is_kept() {
    let h = Harness::new(true);
    h.app.speak(SpeakParams { text: "一文目です。二文目もあります。".into(), seed: Some(1234), ..SpeakParams::manual() });
    assert!(h.drain_jobs().iter().all(|j| j.seed == Some(1234)));
}

#[test]
fn each_request_gets_its_own_random_seed() {
    let h = Harness::new(true);
    let mut all = std::collections::HashSet::new();
    for _ in 0..5 {
        h.speak("テストです。");
        all.extend(seeds(&h.drain_jobs()));
    }
    assert!(all.len() >= 4); // 31bit 乱数なので衝突はまず起きない
}

#[test]
fn speak_uses_pipeline_chunk_config() {
    let h = Harness::new(true);
    h.set("pipeline", "first_chunk_mora_max", json!(0));
    h.speak("こんにちは、今日はいい天気ですね。");
    assert_eq!(h.drain_jobs().iter().map(|j| j.text.clone()).collect::<Vec<_>>(), ["こんにちは、今日はいい天気ですね。"]);
    h.set("pipeline", "first_chunk_mora_max", json!(12));
    h.speak("こんにちは、今日はいい天気ですね。");
    assert_eq!(h.drain_jobs().iter().map(|j| j.text.clone()).collect::<Vec<_>>(), ["こんにちは、", "今日はいい天気ですね。"]);
}

struct FailTts(&'static str);

impl TtsEngine for FailTts {
    fn model_id(&self) -> &str {
        "fail"
    }
    fn synthesize(&self, _r: &TtsRequest) -> Result<TtsOutput> {
        bail!("{}", self.0)
    }
}

#[test]
fn device_lost_marks_tts_error_and_rejects_further_speaks() {
    let h = Harness::new(true);
    *lock(&h.app.engine) = Some(Arc::new(FailTts("wgpu: Parent device is lost (DEVICE_LOST)")));
    {
        let mut st = lock(&h.app.state);
        st.tts_phase = READY.into();
        st.tts_loaded_model = Some("v4.1-small-mf".into());
    }
    h.speak("テストです。");
    for job in h.drain_jobs() {
        h.app.run_job(&job);
    }
    assert_eq!(lock(&h.app.state).tts_phase, ERROR);
    let errors = h.rec.of_type("error");
    assert_eq!(errors.last().unwrap()["recoverable"], false);
    assert_eq!(h.rec.of_type("state").last().unwrap()["tts"]["phase"], "error");

    // 以降の発話は受け付けず(speak_accepted を出さず)、キューにも積まない
    let before = h.rec.of_type("speak_accepted").len();
    h.speak("もう一度です。");
    assert_eq!(h.rec.of_type("speak_accepted").len(), before);
    assert!(h.drain_jobs().is_empty());
}

#[test]
fn ordinary_synthesis_failure_stays_recoverable() {
    let h = Harness::new(true);
    *lock(&h.app.engine) = Some(Arc::new(FailTts("something odd")));
    {
        let mut st = lock(&h.app.state);
        st.tts_phase = READY.into();
        st.tts_loaded_model = Some("v4.1-small-mf".into());
    }
    h.speak("テストです。");
    for job in h.drain_jobs() {
        h.app.run_job(&job);
    }
    assert_ne!(lock(&h.app.state).tts_phase, ERROR);
    assert_eq!(h.rec.of_type("error").last().unwrap()["recoverable"], true);
    assert_eq!(h.done()[0]["failed"], true);
}

#[test]
fn unknown_sampling_key_reports_error_instead_of_silent_drop() {
    let h = Harness::new(true).with_engine(GateTts::new());
    h.set("tts", "sampling", json!({"seed": 1}));
    h.speak("テストです。");
    assert!(wait3(|| !h.done().is_empty()));
    let errors = h.rec.of_type("error");
    assert!(errors.iter().any(|e| e["message"].as_str().unwrap().contains("seed")), "{errors:?}");
}

// ---------------------------------------------------------------- 投機的 TTS

const PARTIAL: &str = "こんにちは、今日はいい天気";
const FINAL: &str = "こんにちは、今日はいい天気ですね。";

fn spec_harness() -> (Harness, Arc<GateTts>) {
    let eng = GateTts::new();
    let h = Harness::new(true).with_engine(eng.clone());
    h.set("pipeline", "speculative_tts", json!(true));
    h.set("pipeline", "speculative_stable_partials", json!(2));
    (h, eng)
}

fn partial(h: &Harness, u: u64, t: &str) {
    SessionHost::on_asr_partial(&*h.app, u, t, None);
}

fn final_(h: &Harness, u: u64, t: &str) {
    SessionHost::on_asr_final(&*h.app, u, t, Timing::default());
}

fn audio_pairs(h: &Harness) -> Vec<(u64, bool)> {
    h.rec.of_type("tts_audio").iter().map(|a| (a["chunk"].as_u64().unwrap(), a["speculative"].as_bool().unwrap())).collect()
}

#[test]
fn speculative_disabled_by_default() {
    let h = Harness::new(true);
    assert_eq!(get(&h.app.cfg(), "pipeline", "speculative_tts"), &json!(false));
    for _ in 0..3 {
        partial(&h, 1, PARTIAL);
    }
    assert!(h.app.queue.is_empty());
}

#[test]
fn match_reuses_speculative_first_chunk() {
    let (h, eng) = spec_harness();
    partial(&h, 1, PARTIAL);
    partial(&h, 1, PARTIAL);
    assert!(wait3(|| eng.texts() == ["こんにちは、"]));
    std::thread::sleep(Duration::from_millis(150));
    assert!(h.rec.of_type("tts_audio").is_empty()); // 確定前は絶対に送らない
    final_(&h, 1, FINAL);
    assert!(wait3(|| h.done().len() == 1));
    assert_eq!(audio_pairs(&h), [(0, true), (1, false)]);
    assert_eq!(h.rec.texts_of("tts_chunk_start", "text"), ["こんにちは、", "今日はいい天気ですね。"]);
    assert_eq!(eng.texts(), ["こんにちは、", "今日はいい天気ですね。"]); // 先頭は1回だけ合成
    let audio = h.rec.of_type("tts_audio");
    assert_eq!(audio[0]["seed"], audio[1]["seed"]); // 投機チャンクとも seed を共有
    assert_eq!(h.done()[0]["chunks"], 2);
    assert_eq!(h.done()[0]["cancelled"], false);
}

#[test]
fn mismatch_discards_speculative_audio() {
    let (h, eng) = spec_harness();
    partial(&h, 1, PARTIAL);
    partial(&h, 1, PARTIAL);
    assert!(wait3(|| eng.texts() == ["こんにちは、"]));
    final_(&h, 1, "こんばんは、今日はいい天気ですね。");
    assert!(wait3(|| h.done().len() == 1));
    assert_eq!(h.rec.texts_of("tts_chunk_start", "text"), ["こんばんは、", "今日はいい天気ですね。"]);
    assert!(audio_pairs(&h).iter().all(|a| !a.1));
    assert_eq!(h.rec.of_type("tts_audio").len(), 2);
}

#[test]
fn bind_while_speculation_in_flight_keeps_order() {
    let (h, eng) = spec_harness();
    eng.close_gate();
    partial(&h, 1, PARTIAL);
    partial(&h, 1, PARTIAL);
    assert!(eng.wait_started()); // 投機合成が走っている最中に確定
    final_(&h, 1, FINAL);
    std::thread::sleep(Duration::from_millis(100));
    assert!(h.rec.of_type("tts_audio").is_empty());
    eng.open_gate();
    assert!(wait3(|| h.done().len() == 1));
    assert_eq!(audio_pairs(&h), [(0, true), (1, false)]);
    assert_eq!(eng.texts(), ["こんにちは、", "今日はいい天気ですね。"]);
}

#[test]
fn mismatch_while_in_flight_never_plays_wrong_audio() {
    let (h, eng) = spec_harness();
    eng.close_gate();
    partial(&h, 1, PARTIAL);
    partial(&h, 1, PARTIAL);
    assert!(eng.wait_started());
    final_(&h, 1, "こんばんは、今日はいい天気ですね。");
    eng.open_gate();
    assert!(wait3(|| h.done().len() == 1));
    let starts = h.rec.texts_of("tts_chunk_start", "text");
    assert!(!starts.contains(&"こんにちは、".to_string()));
    assert_eq!(starts, ["こんばんは、", "今日はいい天気ですね。"]);
}

#[test]
fn unstable_partials_do_not_speculate() {
    let (h, eng) = spec_harness();
    partial(&h, 1, PARTIAL);
    partial(&h, 1, "こんばんは、今日はいい天気");
    partial(&h, 1, PARTIAL);
    std::thread::sleep(Duration::from_millis(200));
    assert!(eng.texts().is_empty());
}

#[test]
fn partial_without_boundary_does_not_speculate() {
    let (h, eng) = spec_harness();
    for _ in 0..3 {
        partial(&h, 1, "こんにちは"); // 先頭チャンクの後ろが未確定
    }
    std::thread::sleep(Duration::from_millis(200));
    assert!(eng.texts().is_empty());
}

#[test]
fn voice_change_between_spec_and_final_discards() {
    let (h, eng) = spec_harness();
    partial(&h, 1, PARTIAL);
    partial(&h, 1, PARTIAL);
    assert!(wait3(|| eng.texts() == ["こんにちは、"]));
    h.set("voice", "caption", json!("落ち着いた女性の声"));
    final_(&h, 1, FINAL);
    assert!(wait3(|| h.done().len() == 1));
    assert!(audio_pairs(&h).iter().all(|a| !a.1));
    assert_eq!(eng.texts(), ["こんにちは、", "こんにちは、", "今日はいい天気ですね。"]);
}

// ---------------------------------------------------------------- 逐次読み上げ(話し続けている間の TTS)

const S1: &str = "一文目を話しています。";
const S2: &str = "二文目も続けて話しています。";

fn requests_of(h: &Harness, t: &str) -> Vec<u64> {
    h.rec.of_type(t).iter().map(|m| m["request"].as_u64().unwrap()).collect()
}

#[test]
fn continuous_speech_starts_speaking_before_final() {
    let (h, eng) = gate_harness();
    partial(&h, 1, &format!("{S1}二文"));
    partial(&h, 1, &format!("{S1}二文目も"));
    // 確定(=話し終わり)を待たずに、安定した 1 文目の合成・送出が始まる
    assert!(wait3(|| !h.rec.of_type("tts_audio").is_empty()));
    assert!(h.done().is_empty()); // 発話の途中ではリクエストを閉じない
    partial(&h, 1, &format!("{S1}{S2}三"));
    partial(&h, 1, &format!("{S1}{S2}三文目"));
    final_(&h, 1, &format!("{S1}{S2}三文目で終わります。"));
    assert!(wait3(|| h.done().len() == 1));
    // 1 発話 = 1 リクエスト。既読分は二度読まない
    assert_eq!(requests_of(&h, "speak_accepted"), [1]);
    assert_eq!(h.rec.texts_of("tts_chunk_start", "text").concat(), format!("{S1}{S2}三文目で終わります。"));
    assert_eq!(eng.texts().concat(), format!("{S1}{S2}三文目で終わります。"));
    let audio = h.rec.of_type("tts_audio");
    let chunks: Vec<u64> = audio.iter().map(|a| a["chunk"].as_u64().unwrap()).collect();
    assert_eq!(chunks, (0..audio.len() as u64).collect::<Vec<_>>());
    assert!(audio.iter().all(|a| a["seed"] == audio[0]["seed"])); // 声質が途中で変わらない
    assert_eq!(h.done()[0]["chunks"], audio.len());
    assert_eq!(h.done()[0]["cancelled"], false);
}

#[test]
fn single_unconfirmed_partial_is_not_spoken() {
    let (h, eng) = gate_harness();
    partial(&h, 1, &format!("{S1}二文"));
    partial(&h, 1, "一文目を離しています。二文目"); // 前回と食い違う
    std::thread::sleep(Duration::from_millis(200));
    assert!(eng.texts().is_empty());
    assert!(h.rec.of_type("speak_accepted").is_empty());
}

#[test]
fn sentence_end_at_tail_waits_for_following_speech() {
    let (h, eng) = gate_harness();
    for _ in 0..3 {
        partial(&h, 1, S1); // 末尾の「。」は後続の音声で変わりうる
    }
    std::thread::sleep(Duration::from_millis(200));
    assert!(eng.texts().is_empty());
    final_(&h, 1, S1);
    assert!(wait3(|| h.done().len() == 1));
    assert_eq!(eng.texts().concat(), S1);
    assert_eq!(requests_of(&h, "speak_accepted"), [1]);
}

#[test]
fn final_rewriting_spoken_prefix_does_not_repeat_it() {
    let (h, _eng) = gate_harness();
    partial(&h, 1, &format!("{S1}二"));
    partial(&h, 1, &format!("{S1}二文"));
    assert!(wait3(|| !h.rec.of_type("tts_audio").is_empty()));
    final_(&h, 1, &format!("一文目を離しています。{S2}"));
    assert!(wait3(|| h.done().len() == 1));
    assert_eq!(h.rec.texts_of("tts_chunk_start", "text").concat(), format!("{S1}{S2}"));
}

#[test]
fn auto_speak_off_does_not_speak_while_talking() {
    let (h, eng) = gate_harness();
    h.set("pipeline", "auto_speak", json!(false));
    partial(&h, 1, &format!("{S1}二"));
    partial(&h, 1, &format!("{S1}二文"));
    final_(&h, 1, &format!("{S1}{S2}"));
    std::thread::sleep(Duration::from_millis(200));
    assert!(eng.texts().is_empty());
    assert!(h.rec.of_type("speak_accepted").is_empty());
}

#[test]
fn session_end_without_final_closes_open_request() {
    let (h, _eng) = gate_harness();
    partial(&h, 1, &format!("{S1}二"));
    partial(&h, 1, &format!("{S1}二文"));
    assert!(wait3(|| !h.rec.of_type("tts_audio").is_empty()));
    h.app.close_all_incremental();
    assert!(wait3(|| h.done().len() == 1));
    assert_eq!(h.done()[0]["cancelled"], false);
    partial(&h, 2, &format!("{S2}三")); // 次の発話は新しいリクエスト
    partial(&h, 2, &format!("{S2}三文"));
    final_(&h, 2, &format!("{S2}三文目。"));
    assert!(wait3(|| h.done().len() == 2));
    assert_eq!(requests_of(&h, "speak_accepted"), [1, 2]);
}

#[test]
fn cancel_while_talking_then_rest_is_spoken_as_new_request() {
    let (h, eng) = gate_harness();
    eng.close_gate();
    partial(&h, 1, &format!("{S1}二"));
    partial(&h, 1, &format!("{S1}二文"));
    assert!(eng.wait_started());
    h.app.cancel_speak();
    eng.open_gate();
    assert!(wait3(|| h.done().len() == 1));
    final_(&h, 1, &format!("{S1}{S2}"));
    assert!(wait3(|| h.done().len() == 2));
    assert_eq!(requests_of(&h, "speak_accepted"), [1, 2]);
    assert_eq!(h.rec.texts_of("tts_chunk_start", "text").concat(), S2); // 取り消した分は読み直さない
}

#[test]
fn next_utterance_final_closes_previous_open_request() {
    let (h, _eng) = gate_harness();
    partial(&h, 1, &format!("{S1}二"));
    partial(&h, 1, &format!("{S1}二文"));
    final_(&h, 2, S2); // 発話 1 の確定が来なかった
    assert!(wait3(|| h.done().len() == 2));
}

// ---------------------------------------------------------------- キャンセル

fn gate_harness() -> (Harness, Arc<GateTts>) {
    let eng = GateTts::new();
    (Harness::new(true).with_engine(eng.clone()), eng)
}

#[test]
fn cancel_drops_chunk_being_synthesized() {
    let (h, eng) = gate_harness();
    eng.close_gate();
    h.speak("これは一つだけのチャンクです。");
    assert!(eng.wait_started()); // 合成中(Irodori は中断できない)
    h.app.cancel_speak();
    eng.open_gate();
    std::thread::sleep(Duration::from_millis(300));
    assert!(h.rec.of_type("tts_audio").is_empty()); // 合成中だったチャンクも送らない
    let done = h.done();
    assert_eq!((done.len(), done[0]["cancelled"].as_bool()), (1, Some(true)));
}

#[test]
fn cancel_drops_queued_and_inflight_chunks_of_multi_chunk_request() {
    let (h, eng) = gate_harness();
    eng.close_gate();
    h.speak("こんにちは、今日はいい天気ですね。明日も晴れるといいですね。");
    assert!(eng.wait_started());
    h.app.cancel_speak();
    eng.open_gate();
    std::thread::sleep(Duration::from_millis(300));
    assert!(h.rec.of_type("tts_audio").is_empty());
    assert_eq!(eng.texts().len(), 1); // 後続チャンクは合成もしない
    assert_eq!(h.done().iter().map(|d| d["cancelled"].as_bool().unwrap()).collect::<Vec<_>>(), [true]);
}

#[test]
fn speak_after_cancel_plays_normally() {
    let (h, eng) = gate_harness();
    eng.close_gate();
    h.speak("一つ目です。");
    assert!(eng.wait_started());
    h.app.cancel_speak();
    eng.open_gate();
    h.speak("二つ目です。");
    assert!(wait3(|| h.done().len() == 2));
    assert_eq!(h.rec.of_type("tts_audio").iter().map(|a| a["request"].as_u64().unwrap()).collect::<Vec<_>>(), [2]);
    assert_eq!(h.done()[1], json!({"type": "speak_done", "request": 2, "chunks": 1, "cancelled": false, "failed": false}));
}

#[test]
fn cancel_discards_speculation() {
    let (h, eng) = gate_harness();
    h.set("pipeline", "speculative_tts", json!(true));
    eng.close_gate();
    partial(&h, 1, "こんにちは、今日はいい天気");
    partial(&h, 1, "こんにちは、今日はいい天気");
    assert!(eng.wait_started());
    h.app.cancel_speak();
    eng.open_gate();
    final_(&h, 1, FINAL);
    assert!(wait3(|| h.done().len() == 1));
    assert!(audio_pairs(&h).iter().all(|a| !a.1));
}

#[test]
fn session_restart_cooldown_blocks_and_allows() {
    let h = Harness::new(true);
    h.app.start_session();
    assert!(lock(&h.app.session).is_some());
    h.app.stop_session();
    assert!(lock(&h.app.session).is_none());

    // クールダウン内の再開は拒否(Error 通知)
    h.app.start_session();
    assert!(lock(&h.app.session).is_none());
    let errs = h.rec.of_type("error");
    assert_eq!(errs.last().unwrap()["scope"], "asr");

    // クールダウン経過後は開始できる
    std::thread::sleep(Duration::from_secs_f64(SESSION_RESTART_COOLDOWN_S + 0.1));
    h.app.start_session();
    assert!(lock(&h.app.session).is_some());
    h.app.stop_session();
}

// ---------------------------------------------------------------- 計測 / 表現

#[test]
fn auto_speak_carries_speech_end_to_e2e() {
    let (h, _eng) = gate_harness();
    let speech_end = now() - 0.5; // 0.5 秒前に話し終わった
    SessionHost::on_asr_final(
        &*h.app,
        1,
        "こんにちは、今日はいい天気ですね。",
        Timing { speech_end: Some(speech_end), vad_end: Some(speech_end + 0.28), asr_ms: Some(120), audio_ms: Some(2000) },
    );
    assert!(wait(Duration::from_secs(5), || h.done().len() == 1));
    let fin = &h.rec.of_type("asr_final")[0];
    assert_eq!((fin["asr_ms"].as_u64(), fin["audio_ms"].as_u64()), (Some(120), Some(2000)));
    assert!((270..=290).contains(&fin["vad_wait_ms"].as_u64().unwrap()));
    assert!((fin["speech_end_ms"].as_f64().unwrap() - speech_end * 1000.0).abs() < 1.0);
    let acc = &h.rec.of_type("speak_accepted")[0];
    assert_eq!(acc["utterance"], 1);
    assert_eq!(acc["speech_end_ms"], fin["speech_end_ms"]);
    let audio = h.rec.of_type("tts_audio");
    let (first, rest) = (&audio[0], &audio[1..]);
    assert_eq!(first["first_chunk"], true);
    assert!(first["e2e_ms"].as_u64().unwrap() >= 500);
    assert!(first["first_chunk_ms"].as_u64().unwrap() <= first["e2e_ms"].as_u64().unwrap());
    assert!(rest.iter().all(|a| a["e2e_ms"].is_null() && a["first_chunk"] == false));
    assert!(audio.iter().all(|a| !a["rtf"].is_null() && !a["queue_wait_ms"].is_null() || a["speculative"] == true));
}

fn perf_slot(obs: AcousticObservation) -> Arc<PerfSlot> {
    let slot = Arc::new(PerfSlot::default());
    lock(&slot.st).0 = Some(obs);
    slot
}

#[test]
fn auto_speak_sends_separate_delivery_to_irodori() {
    let h = Harness::new(true);
    h.set("voice", "caption", json!("落ち着いた声"));
    h.set("voice", "ref_wavs", json!(["target.wav"]));
    h.set("pipeline", "performance_wait_ms", json!(0));
    // 解析済みの結果(活動音声で、間は無い)を仕込む。感情は Rust 版では未対応なので話速のみ。
    let obs = AcousticObservation { audio_ms: 1200, active_ms: 1100, pause_ms: 450, rms: 0.08, mora_per_s: None };
    lock(&h.app.perf_pending).insert(7, perf_slot(obs));
    final_(&h, 7, "こんにちは。");
    let fin = h.rec.of_type("asr_final").pop().unwrap();
    let acc = h.rec.of_type("speak_accepted").pop().unwrap();
    let job = h.drain_jobs().remove(0);
    assert_eq!(fin["text"], "こんにちは。");
    assert_eq!(fin["delivery"]["style"], "間を取りながら");
    assert_eq!(acc["delivery"]["style"], "間を取りながら");
    assert_eq!(job.text, "こんにちは。");
    assert_eq!(job.caption.as_deref(), Some("落ち着いた声。話し方は間を取りながら。"));
    assert_eq!(job.ref_wavs, ["target.wav"]);
}

fn configure_voice(h: &Harness, voice: sttts_protocol::VoiceConfig) {
    h.app.dispatch(GuiMessage::configure_voice(voice));
}

/// GUI の「声」欄(話し方・seed・声)の変更が、次の自動発話にそのまま効く
#[test]
fn auto_speak_follows_voice_configure() {
    let h = Harness::new(true);
    h.set("pipeline", "performance_enabled", json!(false));
    configure_voice(
        &h,
        sttts_protocol::VoiceConfig {
            caption: Some("落ち着いた声".into()),
            ref_wavs: Some(vec!["target.wav".into()]),
            no_ref: Some(false),
            seed: Some(42),
        },
    );
    final_(&h, 1, "こんにちは。");
    let job = h.drain_jobs().remove(0);
    assert_eq!((job.caption.as_deref(), job.ref_wavs.as_slice(), job.seed), (Some("落ち着いた声"), &["target.wav".to_string()][..], Some(42)));

    // 既定の声・話し方なし・ランダムへ戻すと、前の値を引きずらない
    configure_voice(
        &h,
        sttts_protocol::VoiceConfig { caption: Some(String::new()), ref_wavs: Some(vec![]), no_ref: Some(true), seed: None },
    );
    final_(&h, 2, "こんにちは。");
    let job = h.drain_jobs().remove(0);
    assert_eq!((job.caption, job.ref_wavs.len()), (None, 0));
    assert_ne!(job.seed, Some(42));
}

/// 「テンポと間を再現」の話速は、合成パラメータの duration_scale を置き換えず掛け合わせる
#[test]
fn delivery_tempo_multiplies_user_duration_scale() {
    let mut sampling = json!({"duration_scale": 1.2, "trim_tail": false}).as_object().cloned().unwrap();
    apply_delivery_scale(&mut sampling, 0.9);
    assert!((sampling["duration_scale"].as_f64().unwrap() - 1.08).abs() < 1e-9);
    assert_eq!(sampling["trim_tail"], false);
    let mut sampling = Map::new(); // 未指定なら Irodori の既定(1.0)が基準
    apply_delivery_scale(&mut sampling, 1.1);
    assert!((sampling["duration_scale"].as_f64().unwrap() - 1.1).abs() < 1e-9);
}

#[test]
fn speaking_while_talking_uses_fixed_seed() {
    let (h, _eng) = gate_harness();
    h.set("voice", "seed", json!(7));
    partial(&h, 1, &format!("{S1}二文"));
    partial(&h, 1, &format!("{S1}二文目も"));
    final_(&h, 1, &format!("{S1}{S2}"));
    assert!(wait3(|| h.done().len() == 1));
    let audio = h.rec.of_type("tts_audio");
    assert!(!audio.is_empty() && audio.iter().all(|a| a["seed"] == 7), "{audio:?}");
}

#[test]
fn speculation_uses_fixed_seed_and_is_discarded_when_it_changes() {
    let (h, eng) = spec_harness();
    h.set("voice", "seed", json!(7));
    partial(&h, 1, PARTIAL);
    partial(&h, 1, PARTIAL);
    assert!(wait3(|| eng.texts() == ["こんにちは、"]));
    h.set("voice", "seed", json!(8));
    final_(&h, 1, FINAL);
    assert!(wait3(|| h.done().len() == 1));
    assert!(audio_pairs(&h).iter().all(|a| !a.1)); // seed 7 の投機結果は流用しない
    assert!(h.rec.of_type("tts_audio").iter().all(|a| a["seed"] == 8));
}

#[test]
fn expression_deadline_does_not_delay_speech() {
    let h = Harness::new(true);
    h.set("pipeline", "performance_wait_ms", json!(0));
    lock(&h.app.perf_pending).insert(8, Arc::new(PerfSlot::default()));
    final_(&h, 8, "大丈夫です。");
    assert!(h.rec.of_type("asr_final").last().unwrap()["delivery"].is_null());
    assert_eq!(h.drain_jobs().remove(0).text, "大丈夫です。");
}

#[test]
fn utterance_audio_is_analysed_in_background() {
    let h = Harness::new(true);
    h.set("pipeline", "performance_wait_ms", json!(2000));
    let mut audio = vec![0.08f32; 16000];
    audio.extend(vec![0.0; 8000]);
    audio.extend(vec![0.08f32; 16000]);
    SessionHost::on_utterance_audio(&*h.app, 3, &audio);
    final_(&h, 3, "背景解析のテストです。");
    let fin = h.rec.of_type("asr_final").pop().unwrap();
    assert!(fin["pause_ms"].as_u64().unwrap() >= 400, "{fin}");
}

// ---------------------------------------------------------------- ASR 事前ロード

/// 設定ごとに作ったエンジンを記録する偽ファクトリ
struct FakeAsr {
    cfg: Value,
    unloaded: Mutex<bool>,
}

impl AsrEngine for FakeAsr {
    fn model_id(&self) -> String {
        format!("fake-{}-{}", self.cfg["engine"], self.cfg["gemini_api_key"])
    }
    fn transcribe_utterance(&self, _a: &[f32]) -> Result<String> {
        Ok(String::new())
    }
    fn transcribe_partial(&self, _a: &[f32]) -> Result<String> {
        Ok(String::new())
    }
    fn unload(&self) {
        *self.unloaded.lock().unwrap() = true;
    }
}

type Gate = Arc<(Mutex<bool>, Condvar)>;

#[derive(Default)]
struct FakeFactory {
    created: Mutex<Vec<Arc<FakeAsr>>>,
    gates: Mutex<std::collections::HashMap<String, Gate>>,
    fail_keys: Mutex<std::collections::HashSet<String>>,
    count: AtomicUsize,
}

impl FakeFactory {
    fn build(&self, cfg: &Value) -> Result<Arc<dyn AsrEngine>> {
        let asr = &cfg["asr"];
        let key = asr["gemini_api_key"].as_str().unwrap_or("").to_string();
        let gate = self.gates.lock().unwrap().get(&key).cloned();
        let engine = Arc::new(FakeAsr { cfg: asr.clone(), unloaded: Mutex::new(false) });
        self.created.lock().unwrap().push(Arc::clone(&engine));
        self.count.fetch_add(1, Ordering::SeqCst);
        if let Some(g) = gate {
            let guard = g.0.lock().unwrap();
            let _ = g.1.wait_timeout_while(guard, Duration::from_secs(5), |o| !*o).unwrap();
        }
        if self.fail_keys.lock().unwrap().contains(&key) {
            bail!("Gemini API キーがありません");
        }
        Ok(engine)
    }
}

fn real_app() -> (Harness, Arc<FakeFactory>) {
    let h = Harness::new(false);
    let factory = Arc::new(FakeFactory::default());
    let f2 = Arc::clone(&factory);
    // Harness の Platform は作成済みなので、Inner を作り直して差し替える
    let rec = Arc::clone(&h.rec);
    let sink = Sink::new(move |m| rec.msgs.lock().unwrap().push(serde_json::to_value(&m).unwrap()));
    let platform = FakePlatform { asr: Mutex::new(Some(Box::new(move |cfg| f2.build(cfg)))) };
    let opts = BackendOptions { mock: false, save_wavs: false, load_user_file: false, ..Default::default() };
    let app = Inner::new(opts, Arc::new(platform), sink);
    (Harness { app, rec: h.rec.clone(), worker: None }, factory)
}

fn configure_asr(h: &Harness, patch: Value) {
    h.app.configure(&json!({"asr": patch}));
}

fn injectable(h: &Harness) -> Option<Arc<dyn AsrEngine>> {
    let key = h.app.asr_config_key();
    let a = lock(&h.app.asr);
    if a.engine_key.as_ref() == Some(&key) { a.engine.clone() } else { None }
}

fn key_of(e: &Arc<dyn AsrEngine>) -> String {
    e.model_id()
}

#[test]
fn api_key_change_rebuilds_gemini_engine() {
    let (h, factory) = real_app();
    configure_asr(&h, json!({"engine": "gemini", "gemini_api_key": "key-a"}));
    assert!(wait3(|| injectable(&h).is_some()));
    let first = injectable(&h).unwrap();
    assert!(key_of(&first).contains("key-a"));

    configure_asr(&h, json!({"gemini_api_key": "key-b"}));
    assert!(wait3(|| injectable(&h).is_some_and(|e| key_of(&e).contains("key-b"))));
    assert!(wait3(|| *factory.created.lock().unwrap()[0].unloaded.lock().unwrap()));

    // 同じキーの再送では作り直さない
    let count = factory.count.load(Ordering::SeqCst);
    configure_asr(&h, json!({"gemini_api_key": "key-b"}));
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(factory.count.load(Ordering::SeqCst), count);
}

#[test]
fn failed_switch_does_not_reuse_previous_engine() {
    let (h, factory) = real_app();
    configure_asr(&h, json!({"engine": "nemotron"}));
    assert!(wait3(|| injectable(&h).is_some()));

    // キー未設定の Gemini へ切替 → プリロード失敗。旧(nemotron)は流用されない
    factory.fail_keys.lock().unwrap().insert(String::new());
    configure_asr(&h, json!({"engine": "gemini", "gemini_api_key": ""}));
    assert!(wait3(|| lock(&h.app.state).asr_phase == ERROR));
    assert!(injectable(&h).is_none());

    // キーを入れると再試行され、使えるようになる
    configure_asr(&h, json!({"gemini_api_key": "key-a"}));
    assert!(wait3(|| injectable(&h).is_some()));
    assert!(key_of(&injectable(&h).unwrap()).contains("gemini"));
}

#[test]
fn stale_preload_result_is_discarded() {
    let (h, factory) = real_app();
    let slow = Arc::new((Mutex::new(false), Condvar::new()));
    factory.gates.lock().unwrap().insert("key-slow".into(), Arc::clone(&slow));
    configure_asr(&h, json!({"engine": "gemini", "gemini_api_key": "key-slow"}));
    assert!(wait3(|| factory.count.load(Ordering::SeqCst) == 1));

    // ロード中にキーが変わる。旧設定(key-slow)のエンジンは流用されない
    configure_asr(&h, json!({"gemini_api_key": "key-fast"}));
    if let Some(cur) = injectable(&h) {
        assert!(key_of(&cur).contains("key-fast"));
    }

    // 先発が完了しても採用されず解放され、後発(現設定)が採用される。
    *slow.0.lock().unwrap() = true;
    slow.1.notify_all();
    assert!(wait3(|| injectable(&h).is_some()));
    assert!(key_of(&injectable(&h).unwrap()).contains("key-fast"));
    assert!(wait3(|| *factory.created.lock().unwrap()[0].unloaded.lock().unwrap()));
}

#[test]
fn configure_via_gui_message_reaches_config() {
    let h = Harness::new(true);
    let cfg = sttts_protocol::PipelineConfig { auto_speak: Some(false), ..Default::default() };
    h.app.dispatch(GuiMessage::configure_pipeline(cfg));
    assert_eq!(get(&h.app.cfg(), "pipeline", "auto_speak"), &json!(false));
    assert_eq!(get(&h.app.cfg(), "pipeline", "chunk_min_chars"), &json!(16)); // 他キーは保持
    // auto_speak 無効なら確定しても発話しない
    final_(&h, 1, "こんにちは。");
    assert!(h.rec.of_type("speak_accepted").is_empty());
}

/// GUI は tts.sampling を丸ごと送る。消した項目(= Irodori の既定に戻す)が残らないこと
#[test]
fn configure_replaces_tts_sampling_as_a_whole() {
    let h = Harness::new(true);
    let send = |sampling: Value| {
        let tts = sttts_protocol::TtsConfig { sampling: sampling.as_object().cloned(), ..Default::default() };
        h.app.dispatch(GuiMessage::configure_tts(tts));
    };
    send(json!({"duration_scale": 1.2, "trim_tail": false}));
    assert_eq!(get(&h.app.cfg(), "tts", "sampling"), &json!({"duration_scale": 1.2, "trim_tail": false}));
    send(json!({"trim_tail": false}));
    assert_eq!(get(&h.app.cfg(), "tts", "sampling"), &json!({"trim_tail": false}));
    assert_eq!(get(&h.app.cfg(), "tts", "model"), &json!("v4.1-small-mf")); // 他キーは保持
}

#[test]
fn ping_pong_and_hello() {
    let h = Harness::new(true);
    h.app.announce();
    h.app.dispatch(GuiMessage::Ping { nonce: 42 });
    let hello = h.rec.of_type("hello").remove(0);
    assert_eq!((hello["protocol"].as_u64(), hello["mock"].as_bool()), (Some(1), Some(true)));
    assert_eq!(hello["models"][0]["id"], "v4.1-small-mf");
    assert_eq!(h.rec.of_type("pong")[0]["nonce"], 42);
    assert_eq!(h.rec.of_type("state").len(), 1);
    assert_eq!(h.rec.of_type("devices").len(), 1);
}

#[test]
fn mock_asr_is_selectable_via_create_asr() {
    let cfg = merge_config(&default_config(), &json!({"asr": {"engine": "mock", "mock_texts": ["テストです"]}}));
    let e = crate::asr::create_asr(&cfg, &|_| {}).unwrap();
    assert_eq!(e.transcribe_utterance(&[0.0; 16]).unwrap(), "テストです");
    let _ = MockAsr::new(0, vec![]);
}

#[test]
fn full_backend_roundtrip_through_dispatcher() {
    let rec = Arc::new(Rec::default());
    let r2 = Arc::clone(&rec);
    let sink = Sink::new(move |m| r2.msgs.lock().unwrap().push(serde_json::to_value(&m).unwrap()));
    let opts = BackendOptions { mock: true, save_wavs: false, load_user_file: false, ..Default::default() };
    let backend = Backend::start(opts, Arc::new(FakePlatform { asr: Mutex::new(None) }), sink);
    backend.send(GuiMessage::Speak { text: "こんにちは、テストです。".into(), caption: None, ref_wavs: None, seed: Some(1), tag: None, delivery: None });
    assert!(wait(Duration::from_secs(5), || !rec.of_type("speak_done").is_empty()));
    assert_eq!(rec.of_type("speak_done")[0]["cancelled"], false);
    assert!(!rec.of_type("tts_audio").is_empty());
    backend.shutdown();
}

#[test]
fn bound_speculation_survives_discard_and_still_completes() {
    let (h, eng) = spec_harness();
    eng.close_gate();
    partial(&h, 1, PARTIAL);
    partial(&h, 1, PARTIAL);
    assert!(eng.wait_started()); // 投機合成の最中に確定(束縛)
    final_(&h, 1, FINAL);
    // 次の発話の確定が空でも(discard_specs が走っても)、束縛済みのリクエストは完了できる
    final_(&h, 2, "");
    eng.open_gate();
    assert!(wait3(|| h.done().len() == 1), "束縛済みの投機チャンクが失われた");
    assert_eq!(audio_pairs(&h), [(0, true), (1, false)]);
}

#[test]
fn device_fatal_state_is_not_reloaded() {
    let h = Harness::new(true);
    *lock(&h.app.engine) = Some(Arc::new(FailTts("Parent device is lost")));
    {
        let mut st = lock(&h.app.state);
        st.tts_phase = READY.into();
        st.tts_loaded_model = Some("v4.1-small-mf".into());
    }
    h.speak("一つ目。二つ目です。三つ目です。");
    for job in h.drain_jobs() {
        h.app.run_job(&job);
    }
    // 致命状態のままで、エンジンを作り直さず(LOADING に戻らず)、以降のチャンクも失敗として閉じる
    let st = lock(&h.app.state);
    assert!(st.tts_phase == ERROR && st.tts_fatal);
    drop(st);
    assert!(h.rec.of_type("state").iter().all(|s| s["tts"]["phase"] != "loading"));
}
