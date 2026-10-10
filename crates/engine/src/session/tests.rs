//! ASR ワーカー / VAD セグメンタ / ライブセッションのテスト(実モデル不要)。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use super::*;
use crate::asr::{AsrEngine, MockAsr};

fn wait_until(timeout: Duration, mut pred: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + timeout;
    while Instant::now() < end {
        if pred() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    pred()
}

/// 呼び出しごとにゲートで止められる ASR。呼ばれた順序を記録する。
struct GateAsr {
    calls: Mutex<Vec<(&'static str, i32)>>,
    gate: (Mutex<bool>, Condvar),
    entered: (Mutex<bool>, Condvar),
}

impl GateAsr {
    fn new(block_first: bool) -> Arc<Self> {
        Arc::new(Self { calls: Mutex::new(vec![]), gate: (Mutex::new(!block_first), Condvar::new()), entered: (Mutex::new(false), Condvar::new()) })
    }
    fn open(&self) {
        *self.gate.0.lock().unwrap() = true;
        self.gate.1.notify_all();
    }
    fn wait_entered(&self, d: Duration) -> bool {
        let g = self.entered.0.lock().unwrap();
        *self.entered.1.wait_timeout_while(g, d, |e| !*e).unwrap().0
    }
    fn run(&self, kind: &'static str, audio: &[f32]) -> Result<String> {
        let tag = audio.first().map_or(-1, |&v| v as i32);
        self.calls.lock().unwrap().push((kind, tag));
        *self.entered.0.lock().unwrap() = true;
        self.entered.1.notify_all();
        let g = self.gate.0.lock().unwrap();
        let _ = self.gate.1.wait_timeout_while(g, Duration::from_secs(5), |o| !*o).unwrap();
        Ok(format!("{kind}:{tag}"))
    }
    fn calls(&self) -> Vec<(&'static str, i32)> {
        self.calls.lock().unwrap().clone()
    }
}

impl AsrEngine for GateAsr {
    fn model_id(&self) -> String {
        "gate".into()
    }
    fn transcribe_utterance(&self, audio: &[f32]) -> Result<String> {
        self.run("final", audio)
    }
    fn transcribe_partial(&self, audio: &[f32]) -> Result<String> {
        self.run("partial", audio)
    }
}

fn audio(tag: i32) -> Vec<f32> {
    vec![tag as f32; 1600]
}

fn final_job(utt: u64, tag: i32) -> FinalJob {
    FinalJob { utterance: utt, audio: audio(tag), speech_end: 0.0, vad_end: 0.0, audio_ms: None }
}

type Partials = Arc<Mutex<Vec<(u64, String)>>>;
type Finals = Arc<Mutex<Vec<(u64, String)>>>;

fn worker_with(asr: Arc<dyn AsrEngine>) -> (Arc<AsrWorker>, Partials, Finals) {
    let partials: Partials = Arc::default();
    let finals: Finals = Arc::default();
    let (p, f) = (Arc::clone(&partials), Arc::clone(&finals));
    let w = AsrWorker::new(
        asr,
        Arc::new(move |u, t, _| p.lock().unwrap().push((u, t.to_string()))),
        Arc::new(move |j, t, _| f.lock().unwrap().push((j.utterance, t.to_string()))),
        None,
    );
    w.start();
    (w, partials, finals)
}

#[test]
fn inflight_partial_is_discarded_when_final_arrives() {
    let asr = GateAsr::new(true);
    let (w, partials, finals) = worker_with(asr.clone());
    w.submit_partial(PartialJob { utterance: 1, audio: audio(10) });
    assert!(asr.wait_entered(Duration::from_secs(2))); // partial 処理中
    w.submit_partial(PartialJob { utterance: 1, audio: audio(11) }); // 待機中 partial
    w.submit_final(final_job(1, 99)); // 確定 → 待機中/処理中 partial は無効
    asr.open();
    assert!(wait_until(Duration::from_secs(3), || !finals.lock().unwrap().is_empty()));
    w.stop(Duration::from_secs(2));
    assert_eq!(*finals.lock().unwrap(), [(1, "final:99".to_string())]);
    assert!(partials.lock().unwrap().is_empty()); // 処理中だった partial の結果も捨てる
    assert!(!asr.calls().contains(&("partial", 11))); // 待機中 partial はデコードすらしない
}

#[test]
fn partials_are_coalesced_to_latest() {
    let asr = GateAsr::new(true);
    let (w, partials, finals) = worker_with(asr.clone());
    w.submit_final(final_job(1, 1));
    assert!(asr.wait_entered(Duration::from_secs(2))); // final 処理中に partial が3つ届く
    for tag in [20, 21, 22] {
        w.submit_partial(PartialJob { utterance: 2, audio: audio(tag) });
    }
    asr.open();
    assert!(wait_until(Duration::from_secs(3), || !partials.lock().unwrap().is_empty()));
    w.stop(Duration::from_secs(2));
    assert_eq!(finals.lock().unwrap().iter().map(|f| f.0).collect::<Vec<_>>(), [1]);
    assert_eq!(*partials.lock().unwrap(), [(2, "partial:22".to_string())]);
    assert_eq!(w.stats().partials_dropped, 2);
}

#[test]
fn finals_are_never_dropped_and_keep_order() {
    let asr = GateAsr::new(false);
    let (w, _p, finals) = worker_with(asr);
    for utt in 1..8u64 {
        w.submit_partial(PartialJob { utterance: utt, audio: audio(utt as i32) });
        w.submit_final(final_job(utt, utt as i32));
    }
    assert!(wait_until(Duration::from_secs(3), || finals.lock().unwrap().len() >= 7));
    w.stop(Duration::from_secs(2));
    assert_eq!(finals.lock().unwrap().iter().map(|f| f.0).collect::<Vec<_>>(), (1..8).collect::<Vec<_>>());
}

#[test]
fn final_has_priority_over_pending_partial() {
    let asr = GateAsr::new(true);
    let (w, _p, _f) = worker_with(asr.clone());
    w.submit_final(final_job(1, 1));
    assert!(asr.wait_entered(Duration::from_secs(2)));
    w.submit_partial(PartialJob { utterance: 2, audio: audio(2) });
    w.submit_final(final_job(2, 3));
    asr.open();
    assert!(wait_until(Duration::from_secs(3), || w.stats().finals_done >= 2));
    w.stop(Duration::from_secs(2));
    assert_eq!(asr.calls(), [("final", 1), ("final", 3)]);
}

#[test]
fn wait_idle_waits_for_inflight_final() {
    let asr = GateAsr::new(true);
    let (w, _p, finals) = worker_with(asr.clone());
    w.submit_final(final_job(1, 1));
    assert!(asr.wait_entered(Duration::from_secs(2)));
    assert!(!w.wait_idle(Duration::from_millis(200))); // デコード中は idle ではない
    asr.open();
    assert!(w.wait_idle(Duration::from_secs(3)));
    assert_eq!(finals.lock().unwrap().iter().map(|f| f.0).collect::<Vec<_>>(), [1]);
    w.stop(Duration::from_secs(2));
}

// ---------------------------------------------------------------- VadSegmenter

/// フレーム番号(reset 回数, reset 後の相対フレーム) → イベント の台本で動く VAD
struct ScriptVad {
    script: HashMap<(usize, usize), VadEvent>,
    frame: usize,
    resets: Arc<Mutex<usize>>,
}

impl ScriptVad {
    fn boxed(script: &[((usize, usize), VadEvent)]) -> (Box<dyn Vad>, Arc<Mutex<usize>>) {
        let resets = Arc::new(Mutex::new(0));
        (Box::new(Self { script: script.iter().copied().collect(), frame: 0, resets: Arc::clone(&resets) }), resets)
    }
}

impl Vad for ScriptVad {
    fn process(&mut self, _frame: &[f32]) -> Option<VadEvent> {
        let ev = self.script.get(&(*self.resets.lock().unwrap(), self.frame)).copied();
        self.frame += 1;
        ev
    }
    fn reset(&mut self) {
        self.frame = 0;
        *self.resets.lock().unwrap() += 1;
    }
}

#[derive(Default)]
struct Recorder {
    partials: Mutex<Vec<(u64, usize)>>,
    finals: Mutex<Vec<FinalJob>>,
    events: Mutex<Vec<String>>,
}

struct RecSink(Arc<Recorder>);

impl JobSink for RecSink {
    fn submit_partial(&self, job: PartialJob) {
        self.0.partials.lock().unwrap().push((job.utterance, job.audio.len()));
    }
    fn submit_final(&self, job: FinalJob) {
        self.0.events.lock().unwrap().push(format!("final {}", job.utterance));
        self.0.finals.lock().unwrap().push(job);
    }
    fn stream_partial_cb(&self) -> crate::asr::StreamCb {
        Arc::new(|_, _| {})
    }
}

fn clock_at(t: Arc<Mutex<f64>>) -> impl Fn() -> f64 + Send + 'static {
    move || *t.lock().unwrap()
}

fn f(n: usize) -> usize {
    n * FRAME
}

#[test]
fn segmenter_emits_final_with_speech_end_timestamp() {
    // 発話: 2 フレーム目で開始、40 フレーム目で終了検出。end サンプル = 30 フレーム目
    let (vad, resets) = ScriptVad::boxed(&[((0, 2), VadEvent::Start(f(2) as i64)), ((0, 40), VadEvent::End(f(30) as i64))]);
    let rec = Arc::new(Recorder::default());
    let t = Arc::new(Mutex::new(100.0));
    let mut seg = VadSegmenter::new(vad, Box::new(RecSink(rec.clone())), 0.0).with_clock(clock_at(t));
    seg.feed(&vec![0.0; f(41)], Some(100.0));
    let finals = rec.finals.lock().unwrap();
    assert_eq!(finals.len(), 1);
    assert_eq!(finals[0].utterance, 1);
    assert_eq!(finals[0].audio.len(), f(41)); // start 前の 2 フレームを含む
    // ブロック末尾(41 フレーム)が arrival。end は 30 フレーム → 11 フレーム前
    assert!((finals[0].speech_end - (100.0 - f(11) as f64 / 16000.0)).abs() < 1e-6);
    assert_eq!(seg.utterance_id, 2);
    assert_eq!(*resets.lock().unwrap(), 1);
}

struct StreamAsr {
    events: Mutex<Vec<String>>,
    recorder: Arc<Recorder>,
}

impl AsrEngine for StreamAsr {
    fn model_id(&self) -> String {
        "stream".into()
    }
    fn transcribe_utterance(&self, _a: &[f32]) -> Result<String> {
        Ok(String::new())
    }
    fn transcribe_partial(&self, _a: &[f32]) -> Result<String> {
        Ok(String::new())
    }
    fn is_streaming(&self) -> bool {
        true
    }
    fn begin_stream(&self, u: u64, _cb: crate::asr::StreamCb) {
        self.recorder.events.lock().unwrap().push(format!("begin {u}"));
    }
    fn feed_stream(&self, _u: u64, frame: &[f32]) {
        self.events.lock().unwrap().push(format!("audio {}", frame[0] as i32));
        self.recorder.events.lock().unwrap().push("audio".into());
    }
    fn end_stream(&self, u: u64) {
        self.recorder.events.lock().unwrap().push(format!("end {u}"));
    }
}

#[test]
fn segmenter_streams_during_speech_and_ends_before_final_job() {
    let rec = Arc::new(Recorder::default());
    let asr = Arc::new(StreamAsr { events: Mutex::default(), recorder: rec.clone() });
    let (vad, _) = ScriptVad::boxed(&[((0, 0), VadEvent::Start(0)), ((0, 20), VadEvent::End(f(15) as i64))]);
    let mut seg = VadSegmenter::new(vad, Box::new(RecSink(rec.clone())), 0.0).with_stream_asr(asr);
    seg.feed(&vec![1.0; f(10)], None);
    {
        let ev = rec.events.lock().unwrap();
        assert_eq!(ev[0], "begin 1");
        assert!(ev.iter().any(|e| e == "audio"));
    }
    assert!(rec.finals.lock().unwrap().is_empty());
    seg.feed(&vec![1.0; f(11)], None);
    let ev = rec.events.lock().unwrap();
    assert_eq!(ev[ev.len() - 2..], ["end 1".to_string(), "final 1".to_string()]);
}

#[test]
fn segmenter_stream_includes_audio_before_vad_start() {
    let rec = Arc::new(Recorder::default());
    let asr = Arc::new(StreamAsr { events: Mutex::default(), recorder: rec.clone() });
    let (vad, _) = ScriptVad::boxed(&[((0, 3), VadEvent::Start(f(3) as i64)), ((0, 20), VadEvent::End(f(16) as i64))]);
    let mut seg = VadSegmenter::new(vad, Box::new(RecSink(rec.clone())), 0.0).with_stream_asr(asr.clone());
    let samples: Vec<f32> = (0..21).flat_map(|i| vec![i as f32; FRAME]).collect();
    seg.feed(&samples, None);
    assert_eq!(asr.events.lock().unwrap()[..4], ["audio 0", "audio 1", "audio 2", "audio 3"]);
    let finals = rec.finals.lock().unwrap();
    assert_eq!(finals.len(), 1);
    let heads: Vec<i32> = (0..4).map(|i| finals[0].audio[i * FRAME] as i32).collect();
    assert_eq!(heads, [0, 1, 2, 3]);
}

#[test]
fn segmenter_speech_end_after_reset_uses_absolute_position() {
    let (vad, _) = ScriptVad::boxed(&[
        ((0, 0), VadEvent::Start(0)),
        ((0, 20), VadEvent::End(f(10) as i64)),
        ((1, 5), VadEvent::Start(f(5) as i64)),
        ((1, 30), VadEvent::End(f(25) as i64)),
    ]);
    let rec = Arc::new(Recorder::default());
    let mut seg = VadSegmenter::new(vad, Box::new(RecSink(rec.clone())), 0.0).with_clock(|| 0.0);
    seg.feed(&vec![0.0; f(21)], Some(10.0)); // 1 発話目
    seg.feed(&vec![0.0; f(40)], Some(20.0)); // 2 発話目(31 フレームで終了)
    let finals = rec.finals.lock().unwrap();
    assert_eq!(finals.iter().map(|j| j.utterance).collect::<Vec<_>>(), [1, 2]);
    // 2 発話目: reset は絶対 21 フレーム目。end = 21 + 25 = 46、ブロック末尾 = 61
    assert!((finals[1].speech_end - (20.0 - f(15) as f64 / 16000.0)).abs() < 1e-6);
}

#[test]
fn segmenter_submits_partials_on_interval_and_handles_unaligned_blocks() {
    let (vad, _) = ScriptVad::boxed(&[((0, 0), VadEvent::Start(0))]);
    let rec = Arc::new(Recorder::default());
    let t = Arc::new(Mutex::new(0.0));
    let mut seg = VadSegmenter::new(vad, Box::new(RecSink(rec.clone())), 0.5).with_clock(clock_at(t.clone()));
    for i in 0..100 {
        // 30ms 相当(480 サンプル)ずつ、512 に揃わないブロック
        *t.lock().unwrap() = i as f64 * 0.03;
        seg.feed(&vec![1.0; 480], None);
    }
    let partials = rec.partials.lock().unwrap();
    assert!(!partials.is_empty(), "partial が投げられていない");
    assert!(partials.iter().all(|p| p.0 == 1));
    assert!(partials.len() as f64 <= 3.0 / 0.5 + 1.0);
    // 端数は持ち越され、音声は欠けない
    assert_eq!(seg.buffered_samples(), 100 * 480);
}

#[test]
fn segmenter_partial_window_is_per_engine() {
    let run = |max: Option<f64>| {
        let (vad, _) = ScriptVad::boxed(&[((0, 0), VadEvent::Start(0))]);
        let rec = Arc::new(Recorder::default());
        let t = Arc::new(Mutex::new(0.0));
        let mut seg = VadSegmenter::new(vad, Box::new(RecSink(rec.clone())), 0.5).with_max_partial(max).with_clock(clock_at(t.clone()));
        for i in 0..20 {
            *t.lock().unwrap() = i as f64;
            seg.feed(&vec![1.0; 16000], None); // 1 秒ずつ
        }
        rec.partials.lock().unwrap().iter().map(|p| p.1).max().unwrap()
    };
    assert_eq!(run(Some(12.0)), 12 * 16000); // 末尾 12 秒だけ
    assert!(run(None) > 18 * 16000); // 発話の先頭から全部(partial の文字列が先頭から伸びる)
}

// ---------------------------------------------------------------- LiveSession

#[derive(Default)]
struct HostLog {
    events: Mutex<Vec<(String, String)>>,
    timing: Mutex<Option<Timing>>,
}

impl HostLog {
    fn has(&self, kind: &str) -> bool {
        self.events.lock().unwrap().iter().any(|e| e.0 == kind)
    }
    fn of(&self, kind: &str) -> Vec<String> {
        self.events.lock().unwrap().iter().filter(|e| e.0 == kind).map(|e| e.1.clone()).collect()
    }
}

impl SessionHost for HostLog {
    fn on_asr_model_ready(&self, m: &str) {
        self.events.lock().unwrap().push(("ready".into(), m.into()));
    }
    fn on_asr_error(&self, m: &str) {
        self.events.lock().unwrap().push(("error".into(), m.into()));
    }
    fn on_mic_level(&self, _r: f32, db: f32) {
        self.events.lock().unwrap().push(("level".into(), db.to_string()));
    }
    fn on_asr_partial(&self, u: u64, t: &str, _ms: Option<u64>) {
        self.events.lock().unwrap().push(("partial".into(), format!("{u}:{t}")));
    }
    fn on_asr_final(&self, u: u64, t: &str, timing: Timing) {
        *self.timing.lock().unwrap() = Some(timing);
        self.events.lock().unwrap().push(("final".into(), format!("{u}:{t}")));
    }
}

/// start() の中で全ブロックを流し込むソース
struct ListSource {
    on_block: OnBlock,
    blocks: Vec<Vec<f32>>,
    stopped: Arc<AtomicBool>,
}

impl AudioSource for ListSource {
    fn start(&mut self) -> Result<()> {
        for b in &self.blocks {
            (self.on_block)(b.clone());
        }
        Ok(())
    }
    fn stop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
    }
}

fn session_with(host: Arc<HostLog>, asr: AsrSource, block_value: f32, stopped: Arc<AtomicBool>) -> LiveSession {
    let blocks = vec![vec![block_value; FRAME]; 60];
    let (vad, _) = ScriptVad::boxed(&[((0, 5), VadEvent::Start(f(5) as i64)), ((0, 50), VadEvent::End(f(40) as i64))]);
    LiveSession::start(
        host,
        SessionConfig { partial_interval_ms: 0, utterance_start: 1 },
        asr,
        Box::new(move |on_block| Ok(Box::new(ListSource { on_block, blocks, stopped }) as Box<dyn AudioSource>)),
        Box::new(move || Ok(vad)),
    )
}

#[test]
fn live_session_end_to_end_with_mock_asr() {
    let host = Arc::new(HostLog::default());
    let asr: Arc<dyn AsrEngine> = Arc::new(MockAsr::new(50, vec!["テストです".into()]));
    let stopped = Arc::new(AtomicBool::new(false));
    let s = session_with(host.clone(), AsrSource::Preloaded(asr), 0.0, stopped.clone());
    assert!(wait_until(Duration::from_secs(5), || host.has("final")));
    s.stop();
    assert_eq!(host.of("final"), ["1:テストです"]);
    let t = host.timing.lock().unwrap().unwrap();
    assert!(t.asr_ms.unwrap() >= 40);
    assert!(t.speech_end.unwrap() <= t.vad_end.unwrap());
    assert!(stopped.load(Ordering::SeqCst), "stop は音声ソースの解放を保証する");
}

#[test]
fn mic_first_level_flows_before_asr_ready_and_audio_is_kept() {
    // ASR ロード中でもレベルメータが即座に動き、ロード前の音声は捨てられない。
    let host = Arc::new(HostLog::default());
    let load = AsrSource::Load(Box::new(|| {
        std::thread::sleep(Duration::from_millis(400));
        Ok(Arc::new(MockAsr::new(0, vec!["ロード中の発話".into()])) as Arc<dyn AsrEngine>)
    }));
    let s = session_with(host.clone(), load, 0.1, Arc::new(AtomicBool::new(false)));
    // ロード遅延(400ms)より早い時点でレベルが届いていること
    assert!(wait_until(Duration::from_secs(2), || host.has("level")), "level must flow before ASR load completes");
    assert!(!host.has("ready"), "ASR must still be loading here");
    assert!(wait_until(Duration::from_secs(5), || host.has("final")));
    s.stop();
    // ロード前に投入された音声も pending 経由で発話として処理される
    assert_eq!(host.of("final"), ["1:ロード中の発話"]);
}

#[test]
fn asr_load_failure_keeps_level_meter_alive() {
    let host = Arc::new(HostLog::default());
    let load = AsrSource::Load(Box::new(|| Err(anyhow::anyhow!("boom"))));
    let s = session_with(host.clone(), load, 0.1, Arc::new(AtomicBool::new(false)));
    assert!(wait_until(Duration::from_secs(2), || host.has("error")));
    assert!(host.has("level"));
    s.stop();
    assert!(!host.has("final"));
}
