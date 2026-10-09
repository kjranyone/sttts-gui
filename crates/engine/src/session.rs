//! ライブセッション: マイク → VAD → ASR ワーカー → 確定文をアプリへ通知。
//!
//! スレッド構成(VAD をリアルタイムに保ち、音声を一切捨てない):
//!
//! - マイクは最優先で開く(ASR モデルのロードと並行)。準備完了前もレベルメータは
//!   即座に動き、その間の音声は保持してロード完了後に古い順に VAD へ流す。
//! - 音声ソース(オーディオコールバック等)→ 無制限キューへ `(到着時刻, ブロック)` を積む。
//! - VAD スレッド(`asr-vad`): レベルメータ・VAD・発話バッファ管理のみを行う。
//!   重いデコードは行わず、`AsrWorker` へジョブを投げるだけなので常に実時間で回る。
//! - ASR ワーカー(`asr-worker`): デコード専用。確定(final)ジョブを最優先で FIFO 処理し、
//!   部分(partial)ジョブは「最新の1件」だけを保持(古いものは上書き=合体)。
//!   確定が投入された発話の partial は、待機中のものも処理中のものも結果を破棄する。
//!
//! 発話終了時刻(`speech_end`)は VAD の end サンプル位置とブロック到着時刻から逆算する
//! (min_silence の待ち時間も含めた「ユーザーが話し終えた瞬間」を基準に計測するため)。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use sttts_i18n::{tr, trf};

use crate::asr::{AsrEngine, StreamCb};
use crate::util::{join_timeout, lock, now};

pub const SAMPLE_RATE: usize = 16000;
/// Silero VAD の 1 フレーム(32ms @ 16kHz)
pub const FRAME: usize = 512;
const MIN_UTTERANCE_SECONDS: f64 = 0.25;
const MIN_PARTIAL_SECONDS: f64 = 0.6;
const LEVEL_INTERVAL: f64 = 0.1;
/// Silero の start 判定直前 160ms もストリーミング ASR に送る
const PREROLL_FRAMES: usize = 5;

/// VAD が返す発話の開始/終了(サンプル位置)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    Start(i64),
    End(i64),
}

/// 512 サンプル単位で発話区間を検出する VAD
pub trait Vad: Send {
    fn process(&mut self, frame: &[f32]) -> Option<VadEvent>;
    fn reset(&mut self);
}

/// 音声ソース(マイク / WAV)。`stop` は冪等で、戻った時点でデバイス解放を保証する。
pub trait AudioSource: Send {
    fn start(&mut self) -> Result<()>;
    fn stop(&mut self);
}

pub type OnBlock = Arc<dyn Fn(Vec<f32>) + Send + Sync>;
pub type SourceFactory = Box<dyn FnOnce(OnBlock) -> Result<Box<dyn AudioSource>> + Send>;
pub type VadFactory = Box<dyn FnOnce() -> Result<Box<dyn Vad>> + Send>;
/// ASR の用意: 事前ロード済み、またはロード処理(マイクのオープンと並行して走らせる)
pub enum AsrSource {
    Preloaded(Arc<dyn AsrEngine>),
    Load(Box<dyn FnOnce() -> Result<Arc<dyn AsrEngine>> + Send>),
}

pub struct PartialJob {
    pub utterance: u64,
    pub audio: Vec<f32>,
}

pub struct FinalJob {
    pub utterance: u64,
    pub audio: Vec<f32>,
    /// `util::now()` 基準の発話終了推定時刻(秒)
    pub speech_end: f64,
    /// VAD が終了を検出した時刻
    pub vad_end: f64,
    pub audio_ms: Option<u64>,
}

/// 確定の計測値
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub speech_end: Option<f64>,
    pub vad_end: Option<f64>,
    pub asr_ms: Option<u64>,
    pub audio_ms: Option<u64>,
}

/// セッションからアプリへの通知
pub trait SessionHost: Send + Sync {
    fn on_asr_model_ready(&self, model_id: &str);
    fn on_asr_error(&self, message: &str);
    /// セッションスレッドが(停止要求なしに)終わった。ライブ中の表示を戻す
    fn on_session_ended(&self) {}
    fn on_mic_level(&self, rms: f32, db: f32);
    fn on_utterance_audio(&self, _utterance: u64, _audio: &[f32]) {}
    fn on_asr_partial(&self, utterance: u64, text: &str, asr_ms: Option<u64>);
    fn on_asr_final(&self, utterance: u64, text: &str, timing: Timing);
}

/// VadSegmenter がジョブを投げる先
pub trait JobSink {
    fn submit_partial(&self, job: PartialJob);
    fn submit_final(&self, job: FinalJob);
    /// ストリーミング ASR の途中結果の受け口
    fn stream_partial_cb(&self) -> StreamCb;
}

// ---------------------------------------------------------------- ASR ワーカー

type OnPartial = Arc<dyn Fn(u64, &str, u64) + Send + Sync>;
type OnFinal = Arc<dyn Fn(&FinalJob, &str, u64) + Send + Sync>;
type OnError = Arc<dyn Fn(&str) + Send + Sync>;

#[derive(Debug, Default, Clone, Copy)]
pub struct WorkerStats {
    pub partials_done: u64,
    pub partials_dropped: u64,
    pub finals_done: u64,
}

#[derive(Default)]
struct WState {
    finals: VecDeque<FinalJob>,
    partial: Option<PartialJob>,
    finalized_upto: u64,
    busy: bool,
    stop: bool,
    stats: WorkerStats,
}

/// ASR デコード専用ワーカー。final を優先し、partial は最新1件に合体する。
pub struct AsrWorker {
    asr: Arc<dyn AsrEngine>,
    on_partial: OnPartial,
    on_final: OnFinal,
    on_error: Option<OnError>,
    st: Mutex<WState>,
    cv: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
}

enum Job {
    Final(FinalJob),
    Partial(PartialJob),
}

impl AsrWorker {
    pub fn new(asr: Arc<dyn AsrEngine>, on_partial: OnPartial, on_final: OnFinal, on_error: Option<OnError>) -> Arc<Self> {
        Arc::new(Self {
            asr,
            on_partial,
            on_final,
            on_error,
            st: Mutex::new(WState::default()),
            cv: Condvar::new(),
            thread: Mutex::new(None),
        })
    }

    pub fn start(self: &Arc<Self>) {
        let me = Arc::clone(self);
        let h = std::thread::Builder::new().name("asr-worker".into()).spawn(move || me.run()).expect("spawn asr-worker");
        *lock(&self.thread) = Some(h);
    }

    pub fn stop(&self, timeout: Duration) {
        lock(&self.st).stop = true;
        self.cv.notify_all();
        if let Some(h) = lock(&self.thread).take() {
            join_timeout(h, timeout);
        }
    }

    pub fn stats(&self) -> WorkerStats {
        lock(&self.st).stats
    }

    pub fn submit_partial(&self, job: PartialJob) {
        let mut st = lock(&self.st);
        if job.utterance <= st.finalized_upto {
            st.stats.partials_dropped += 1;
            return;
        }
        if st.partial.is_some() {
            st.stats.partials_dropped += 1; // 古い partial は上書き
        }
        st.partial = Some(job);
        self.cv.notify_one();
    }

    pub fn submit_final(&self, job: FinalJob) {
        let mut st = lock(&self.st);
        st.finalized_upto = st.finalized_upto.max(job.utterance);
        st.finals.push_back(job);
        let upto = st.finalized_upto;
        if st.partial.as_ref().is_some_and(|p| p.utterance <= upto) {
            st.partial = None;
            st.stats.partials_dropped += 1;
        }
        self.cv.notify_one();
    }

    pub fn stream_partial(&self, utterance: u64, text: &str) {
        let stale = {
            let st = lock(&self.st);
            utterance <= st.finalized_upto || st.stop
        };
        if !stale {
            (self.on_partial)(utterance, text, 0);
        }
    }

    pub fn pending(&self) -> usize {
        let st = lock(&self.st);
        st.finals.len() + usize::from(st.partial.is_some())
    }

    /// キューが空で、デコード中でもない状態になるまで待つ(ベンチの終了判定用)。
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let end = Instant::now() + timeout;
        let mut st = lock(&self.st);
        while !st.finals.is_empty() || st.partial.is_some() || st.busy {
            let Some(remaining) = end.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else {
                return false;
            };
            st = self.cv.wait_timeout(st, remaining).unwrap_or_else(|e| e.into_inner()).0;
        }
        true
    }

    fn next_job(&self) -> Option<Job> {
        let mut st = lock(&self.st);
        while st.finals.is_empty() && st.partial.is_none() && !st.stop {
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        if let Some(j) = st.finals.pop_front() {
            st.busy = true;
            return Some(Job::Final(j));
        }
        if st.stop {
            return None;
        }
        let j = st.partial.take()?;
        st.busy = true;
        Some(Job::Partial(j))
    }

    fn job_done(&self) {
        lock(&self.st).busy = false;
        self.cv.notify_all();
    }

    fn run(&self) {
        while let Some(job) = self.next_job() {
            self.process(job);
            self.job_done();
        }
    }

    fn process(&self, job: Job) {
        let t0 = Instant::now();
        let result = match &job {
            Job::Final(j) => {
                if self.asr.is_streaming() {
                    self.asr.finish_stream(j.utterance, &j.audio)
                } else {
                    self.asr.transcribe_utterance(&j.audio)
                }
            }
            Job::Partial(j) => self.asr.transcribe_partial(&j.audio),
        };
        let asr_ms = t0.elapsed().as_millis() as u64;
        match (job, result) {
            (Job::Final(j), Ok(text)) => {
                lock(&self.st).stats.finals_done += 1;
                (self.on_final)(&j, &text, asr_ms);
            }
            (Job::Final(j), Err(e)) => {
                // 1件の失敗でワーカーを止めない
                if let Some(cb) = &self.on_error {
                    cb(&decode_failed(&e));
                }
                (self.on_final)(&j, "", asr_ms);
            }
            (Job::Partial(j), Ok(text)) => {
                let stale = {
                    let mut st = lock(&self.st);
                    let stale = j.utterance <= st.finalized_upto;
                    if stale {
                        st.stats.partials_dropped += 1; // 処理中に確定が来た partial は捨てる
                    } else {
                        st.stats.partials_done += 1;
                    }
                    stale
                };
                if !stale {
                    (self.on_partial)(j.utterance, &text, asr_ms);
                }
            }
            (Job::Partial(_), Err(e)) => {
                if let Some(cb) = &self.on_error {
                    cb(&decode_failed(&e));
                }
            }
        }
    }
}

/// `Arc<AsrWorker>` を `JobSink` として使うためのラッパ(ストリーミング通知に自身の Arc が要る)
pub struct ArcWorker(pub Arc<AsrWorker>);

impl JobSink for ArcWorker {
    fn submit_partial(&self, job: PartialJob) {
        self.0.submit_partial(job);
    }
    fn submit_final(&self, job: FinalJob) {
        self.0.submit_final(job);
    }
    fn stream_partial_cb(&self) -> StreamCb {
        let w = Arc::clone(&self.0);
        Arc::new(move |utt, text| w.stream_partial(utt, text))
    }
}

// ---------------------------------------------------------------- VAD セグメンタ

type UtteranceHook = Box<dyn Fn(u64, &[f32]) + Send>;

/// VAD + 発話バッファ管理(デコードはしない)。`feed()` を実時間で呼ぶ。
pub struct VadSegmenter {
    vad: Box<dyn Vad>,
    sink: Box<dyn JobSink>,
    /// <=0 で partial 無効
    partial_interval: f64,
    /// partial に渡す音声の上限(秒)。None で発話の先頭から全部
    max_partial: Option<f64>,
    on_level: Option<Box<dyn Fn(f32, f32) + Send>>,
    on_utterance: Option<UtteranceHook>,
    stream_asr: Option<Arc<dyn AsrEngine>>,
    clock: Box<dyn Fn() -> f64 + Send>,
    pub utterance_id: u64,
    utterance: Vec<Vec<f32>>,
    preroll: VecDeque<Vec<f32>>,
    spoken_samples: usize,
    speaking: bool,
    last_partial: f64,
    last_level: f64,
    /// 512 フレーム整列用の持ち越し
    tail: Vec<f32>,
    /// VAD に与えた総サンプル数
    samples_fed: i64,
    /// 直近 `vad.reset()` 時点の `samples_fed`
    vad_base: i64,
}

impl VadSegmenter {
    pub fn new(vad: Box<dyn Vad>, sink: Box<dyn JobSink>, partial_interval: f64) -> Self {
        Self {
            vad,
            sink,
            partial_interval,
            max_partial: Some(12.0),
            on_level: None,
            on_utterance: None,
            stream_asr: None,
            clock: Box::new(now),
            utterance_id: 1,
            utterance: Vec::new(),
            preroll: VecDeque::new(),
            spoken_samples: 0,
            speaking: false,
            last_partial: 0.0,
            last_level: f64::NEG_INFINITY,
            tail: Vec::new(),
            samples_fed: 0,
            vad_base: 0,
        }
    }

    pub fn with_level(mut self, f: impl Fn(f32, f32) + Send + 'static) -> Self {
        self.on_level = Some(Box::new(f));
        self
    }

    pub fn with_utterance_hook(mut self, f: impl Fn(u64, &[f32]) + Send + 'static) -> Self {
        self.on_utterance = Some(Box::new(f));
        self
    }

    pub fn with_stream_asr(mut self, asr: Arc<dyn AsrEngine>) -> Self {
        self.stream_asr = Some(asr);
        self
    }

    pub fn with_max_partial(mut self, seconds: Option<f64>) -> Self {
        self.max_partial = seconds;
        self
    }

    pub fn with_clock(mut self, f: impl Fn() -> f64 + Send + 'static) -> Self {
        self.clock = Box::new(f);
        self
    }

    pub fn with_utterance_start(mut self, id: u64) -> Self {
        self.utterance_id = id;
        self
    }

    /// 溜めている音声のサンプル数(テスト用)
    pub fn buffered_samples(&self) -> usize {
        self.utterance.iter().map(Vec::len).sum::<usize>() + self.tail.len()
    }

    pub fn feed(&mut self, block: &[f32], arrival: Option<f64>) {
        let now = (self.clock)();
        let arrival = arrival.unwrap_or(now);
        if let Some(cb) = &self.on_level
            && now - self.last_level > LEVEL_INTERVAL
        {
            let (rms, db) = level_of(block);
            cb(rms, db);
            self.last_level = now;
        }

        let mut tail = std::mem::take(&mut self.tail);
        tail.extend_from_slice(block);
        let n_frames = tail.len() / FRAME;
        // このブロック末尾サンプルの絶対位置(到着時刻 arrival に対応)
        let block_end_abs = self.samples_fed + tail.len() as i64;
        for i in 0..n_frames {
            let frame = &tail[i * FRAME..(i + 1) * FRAME];
            let event = self.vad.process(frame);
            self.samples_fed += FRAME as i64;
            if matches!(event, Some(VadEvent::Start(_))) {
                self.speaking = true;
                self.utterance = self.preroll.drain(..).collect();
                self.spoken_samples = 0;
                self.last_partial = now;
                if let Some(asr) = &self.stream_asr {
                    asr.begin_stream(self.utterance_id, self.sink.stream_partial_cb());
                    for leading in &self.utterance {
                        asr.feed_stream(self.utterance_id, leading);
                    }
                }
            }
            if self.speaking {
                self.utterance.push(frame.to_vec());
                self.spoken_samples += FRAME;
                if let Some(asr) = &self.stream_asr {
                    asr.feed_stream(self.utterance_id, frame);
                }
            } else {
                if self.preroll.len() == PREROLL_FRAMES {
                    self.preroll.pop_front();
                }
                self.preroll.push_back(frame.to_vec());
            }
            if let Some(VadEvent::End(end)) = event {
                let end_abs = self.vad_base + end;
                let speech_end = arrival - (block_end_abs - end_abs).max(0) as f64 / SAMPLE_RATE as f64;
                self.finish_utterance(speech_end, now);
            } else if self.speaking && self.partial_interval > 0.0 && now - self.last_partial > self.partial_interval {
                let buffered: usize = self.utterance.iter().map(Vec::len).sum();
                if buffered as f64 >= SAMPLE_RATE as f64 * MIN_PARTIAL_SECONDS {
                    let skip = self.max_partial.map_or(0, |s| buffered.saturating_sub((SAMPLE_RATE as f64 * s) as usize));
                    let audio: Vec<f32> = self.utterance.iter().flatten().skip(skip).copied().collect();
                    self.sink.submit_partial(PartialJob { utterance: self.utterance_id, audio });
                }
                self.last_partial = now;
            }
        }
        self.tail = tail[n_frames * FRAME..].to_vec();
    }

    fn finish_utterance(&mut self, speech_end: f64, now: f64) {
        self.speaking = false;
        let audio: Vec<f32> = std::mem::take(&mut self.utterance).into_iter().flatten().collect();
        self.preroll.clear();
        let valid = self.spoken_samples as f64 >= SAMPLE_RATE as f64 * MIN_UTTERANCE_SECONDS;
        self.spoken_samples = 0;
        if let Some(asr) = &self.stream_asr {
            if valid {
                asr.end_stream(self.utterance_id);
            } else {
                asr.abort_stream(self.utterance_id);
            }
        }
        if valid {
            if let Some(cb) = &self.on_utterance {
                cb(self.utterance_id, &audio);
            }
            let audio_ms = (audio.len() as f64 / SAMPLE_RATE as f64 * 1000.0) as u64;
            self.sink.submit_final(FinalJob { utterance: self.utterance_id, audio, speech_end, vad_end: now, audio_ms: Some(audio_ms) });
        }
        self.utterance_id += 1;
        self.vad.reset();
        self.vad_base = self.samples_fed;
    }
}

/// ブロックの RMS と dBFS
pub fn level_of(block: &[f32]) -> (f32, f32) {
    let rms = if block.is_empty() { 0.0 } else { (block.iter().map(|x| x * x).sum::<f32>() / block.len() as f32).sqrt() };
    (rms, 20.0 * rms.max(1e-6).log10())
}

// ---------------------------------------------------------------- ライブセッション

pub struct SessionConfig {
    /// <=0 で partial 無効(ストリーミング ASR のときも無効)
    pub partial_interval_ms: u64,
    /// 発話 id の開始値(セッションをまたいで衝突しないように)
    pub utterance_start: u64,
}

pub struct LiveSession {
    host: Arc<dyn SessionHost>,
    audio_tx: mpsc::Sender<Option<(f64, Vec<f32>)>>,
    queued: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
    source: Arc<Mutex<Option<Box<dyn AudioSource>>>>,
    worker: Arc<Mutex<Option<Arc<AsrWorker>>>>,
    asr: Arc<Mutex<Option<Arc<dyn AsrEngine>>>>,
}

struct SessionParts {
    host: Arc<dyn SessionHost>,
    cfg: SessionConfig,
    asr: AsrSource,
    source_factory: SourceFactory,
    vad_factory: VadFactory,
    audio_tx: mpsc::Sender<Option<(f64, Vec<f32>)>>,
    audio_rx: mpsc::Receiver<Option<(f64, Vec<f32>)>>,
    queued: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    source: Arc<Mutex<Option<Box<dyn AudioSource>>>>,
    worker: Arc<Mutex<Option<Arc<AsrWorker>>>>,
    asr_slot: Arc<Mutex<Option<Arc<dyn AsrEngine>>>>,
}

impl LiveSession {
    /// セッションスレッドを起動する(マイクを最優先で開き、ASR のロードは並行して進める)。
    pub fn start(host: Arc<dyn SessionHost>, cfg: SessionConfig, asr: AsrSource, source_factory: SourceFactory, vad_factory: VadFactory) -> Self {
        // 無制限キュー: ASR が遅くても音声は捨てない(VAD スレッドは軽量なので溜まらない)
        let (audio_tx, audio_rx) = mpsc::channel();
        let queued = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let source = Arc::new(Mutex::new(None));
        let worker = Arc::new(Mutex::new(None));
        let asr_slot = Arc::new(Mutex::new(None));
        let parts = SessionParts {
            host: Arc::clone(&host),
            cfg,
            asr,
            source_factory,
            vad_factory,
            audio_tx: audio_tx.clone(),
            audio_rx,
            queued: Arc::clone(&queued),
            stop: Arc::clone(&stop),
            source: Arc::clone(&source),
            worker: Arc::clone(&worker),
            asr_slot: Arc::clone(&asr_slot),
        };
        let h = std::thread::Builder::new().name("asr-vad".into()).spawn(move || run_session(parts)).expect("spawn asr-vad");
        Self { host, audio_tx, queued, stop, thread: Mutex::new(Some(h)), source, worker, asr: asr_slot }
    }

    /// セッションを停止する。戻った時点で音声ソース(マイク)の解放を保証する。
    ///
    /// スレッドが重い処理で join に間に合わなくても、ここでソースを閉じてしまう
    /// (`AudioSource::stop` は冪等)。放置するとプロセスが生きている限りマイクを掴み続け、
    /// 次回開始時の二重オープンがドライバクラッシュを引き起こす(2026-10 の BugCheck 0xD1)。
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(asr) = lock(&self.asr).clone() {
            asr.abort_all_streams();
        }
        let _ = self.audio_tx.send(None);
        if let Some(h) = lock(&self.thread).take()
            && !join_timeout(h, Duration::from_secs(10))
        {
            self.host.on_asr_error(tr!(
                "The session thread did not stop in time. Force-closing the source",
                "セッションスレッドが時間内に止まりませんでした。ソースを強制的に閉じます",
                "会话线程未能及时停止。正在强制关闭音源"
            ));
        }
        if let Some(src) = lock(&self.source).as_mut() {
            src.stop();
        }
        // stop と競合して VAD が新しいストリームを作った場合も回収する。
        if let Some(asr) = lock(&self.asr).clone() {
            asr.abort_all_streams();
        }
    }

    /// 投入済みの音声の ASR(と確定コールバック)がすべて終わるまで待つ。
    pub fn wait_asr_idle(&self, timeout: Duration) -> bool {
        let end = Instant::now() + timeout;
        // VAD スレッドがキュー上のブロックを処理し終えるまで待つ
        while self.queued.load(Ordering::SeqCst) > 0 && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(20));
        }
        let Some(w) = lock(&self.worker).clone() else {
            return true;
        };
        w.wait_idle(end.saturating_duration_since(Instant::now()))
    }
}

fn run_session(p: SessionParts) {
    let (host, stop) = (Arc::clone(&p.host), Arc::clone(&p.stop));
    run_session_inner(p);
    if !stop.load(Ordering::SeqCst) {
        host.on_session_ended();
    }
}

fn run_session_inner(p: SessionParts) {
    let SessionParts { host, cfg, asr, source_factory, vad_factory, audio_tx: tx, audio_rx, queued, stop, source, worker, asr_slot } = p;

    // マイクを最優先で開く。ASR ロード完了を待つと初回(モデル取得に数十秒〜)の間
    // レベルメータが完全に動かなくなるため、ロードは並行で進め、準備完了前の音声は
    // 捨てず保持してロード後に古い順に VAD へ流す。
    {
        let queued_in = Arc::clone(&queued);
        let on_block: OnBlock = Arc::new(move |b: Vec<f32>| {
            queued_in.fetch_add(1, Ordering::SeqCst);
            let _ = tx.send(Some((now(), b)));
        });
        match source_factory(on_block) {
            Ok(s) => {
                // 開始中に stop() が来ても取りこぼさないよう、先に登録してから start する。
                // start の間ロックを持つので、並行する stop() は開始の完了を待ってから必ず閉じる
                let mut slot = lock(&source);
                *slot = Some(s);
                let started = slot.as_mut().map_or(Ok(()), |s| s.start());
                if let Err(e) = started {
                    *slot = None;
                    drop(slot);
                    host.on_asr_error(&mic_open_failed(&e));
                    return;
                }
                if stop.load(Ordering::SeqCst) {
                    if let Some(s) = slot.as_mut() {
                        s.stop();
                    }
                    return;
                }
            }
            Err(e) => {
                host.on_asr_error(&mic_open_failed(&e));
                return;
            }
        }
    }

    // ASR ロード(並行)
    let (ready_tx, ready_rx) = mpsc::channel::<Result<Arc<dyn AsrEngine>, String>>();
    {
        let host = Arc::clone(&host);
        std::thread::Builder::new()
            .name("asr-loader".into())
            .spawn(move || {
                let r = match asr {
                    AsrSource::Preloaded(e) => Ok(e),
                    AsrSource::Load(f) => f().map_err(|e| format!("{e:#}")),
                };
                match &r {
                    Ok(e) => host.on_asr_model_ready(&e.model_id()),
                    Err(msg) => host.on_asr_error(&trf!(
                        "Failed to load the ASR model: {msg}",
                        "ASRモデルのロードに失敗: {msg}",
                        "ASR 模型加载失败:{msg}"
                    )),
                }
                let _ = ready_tx.send(r);
            })
            .expect("spawn asr-loader");
    }

    let mut seg: Option<VadSegmenter> = None;
    let mut asr_failed = false;
    // ASR 準備完了前の音声(到着時刻付き)。ロード失敗時の無限成長を防ぐ上限(30ms/ブロック換算で約2分)。
    let mut pending: VecDeque<(f64, Vec<f32>)> = VecDeque::new();
    let mut last_level = f64::NEG_INFINITY;
    let mut vad_factory = Some(vad_factory);
    let mut ready_rx = Some(ready_rx);

    let emit_level = |host: &Arc<dyn SessionHost>, last_level: &mut f64, block: &[f32]| {
        let t = now();
        if t - *last_level > LEVEL_INTERVAL {
            let (rms, db) = level_of(block);
            host.on_mic_level(rms, db);
            *last_level = t;
        }
    };

    'outer: while !stop.load(Ordering::SeqCst) {
        let item = match audio_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(None) => break, // 終端センチネル
            Ok(Some(x)) => {
                queued.fetch_sub(1, Ordering::SeqCst);
                Some(x)
            }
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => break,
        };

        if let Some((arrival, block)) = item {
            if let Some(s) = seg.as_mut() {
                s.feed(&block, Some(arrival));
            } else if asr_failed {
                // ロード失敗: レベルメータは生かし、文字起こしは行わない
                emit_level(&host, &mut last_level, &block);
            } else {
                emit_level(&host, &mut last_level, &block);
                if pending.len() >= 4000 {
                    pending.pop_front();
                }
                pending.push_back((arrival, block));
            }
        }

        // audio_q が空でも、ロード完了を検知したらパイプラインを組んで保持していた音声を流す(次の発話を待たない)。
        if seg.is_none() && !asr_failed {
            let Some(rx) = ready_rx.as_ref() else { continue };
            match rx.try_recv() {
                Ok(Ok(engine)) => {
                    ready_rx = None;
                    *lock(&asr_slot) = Some(Arc::clone(&engine));
                    let on_partial: OnPartial = {
                        let host = Arc::clone(&host);
                        Arc::new(move |u, t, ms| host.on_asr_partial(u, t, Some(ms)))
                    };
                    let on_final: OnFinal = {
                        let host = Arc::clone(&host);
                        Arc::new(move |j, t, ms| {
                            host.on_asr_final(
                                j.utterance,
                                t,
                                Timing { speech_end: Some(j.speech_end), vad_end: Some(j.vad_end), asr_ms: Some(ms), audio_ms: j.audio_ms },
                            )
                        })
                    };
                    let on_error: OnError = {
                        let host = Arc::clone(&host);
                        Arc::new(move |m| host.on_asr_error(m))
                    };
                    let w = AsrWorker::new(Arc::clone(&engine), on_partial, on_final, Some(on_error));
                    w.start();
                    *lock(&worker) = Some(Arc::clone(&w));
                    let vad = match vad_factory.take().map(|f| f()) {
                        Some(Ok(v)) => v,
                        Some(Err(e)) => {
                            host.on_asr_error(&trf!(
                                "Failed to initialize VAD: {e:#}",
                                "VAD の初期化に失敗: {e:#}",
                                "VAD 初始化失败:{e:#}"
                            ));
                            break 'outer;
                        }
                        None => break 'outer,
                    };
                    let streaming = engine.is_streaming();
                    let interval = if streaming || cfg.partial_interval_ms == 0 { 0.0 } else { (cfg.partial_interval_ms as f64 / 1000.0).max(0.4) };
                    let mut s = VadSegmenter::new(vad, Box::new(ArcWorker(Arc::clone(&w))), interval)
                        .with_level({
                            let host = Arc::clone(&host);
                            move |r, d| host.on_mic_level(r, d)
                        })
                        .with_utterance_hook({
                            let host = Arc::clone(&host);
                            move |u, a| host.on_utterance_audio(u, a)
                        })
                        .with_max_partial(engine.max_partial_seconds())
                        .with_utterance_start(cfg.utterance_start.max(1));
                    if streaming {
                        s = s.with_stream_asr(Arc::clone(&engine));
                    }
                    for (arr, blk) in pending.drain(..) {
                        s.feed(&blk, Some(arr)); // 古い順に流す
                    }
                    seg = Some(s);
                }
                Ok(Err(_)) => {
                    ready_rx = None;
                    asr_failed = true;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => asr_failed = true,
            }
        }
    }

    if let Some(s) = lock(&source).as_mut() {
        s.stop();
    }
    let engine = lock(&asr_slot).clone();
    if let Some(e) = engine.filter(|_| !stop.load(Ordering::SeqCst)) {
        e.abort_all_streams();
    }
    if let Some(w) = lock(&worker).clone() {
        w.stop(Duration::from_secs(10));
    }
}

fn decode_failed(e: &anyhow::Error) -> String {
    trf!("ASR decode failed: {e:#}", "ASR デコード失敗: {e:#}", "ASR 解码失败:{e:#}")
}

fn mic_open_failed(e: &anyhow::Error) -> String {
    trf!("Could not open the microphone: {e:#}", "マイクを開けませんでした: {e:#}", "无法打开麦克风:{e:#}")
}

#[cfg(test)]
mod tests;
