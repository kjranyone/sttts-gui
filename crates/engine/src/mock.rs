//! GUI 開発用のモックセッション(`--mock`)。台本どおりに partial → final とマイクレベルを発生させる。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::app::SessionHandle;
use crate::session::{SessionHost, Timing};
use crate::util::{lock, now};

const SCRIPT: [&[&str]; 3] = [
    &["モックの", "モックの文字起こし", "モックの文字起こしテストです"],
    &["次の文です", "次の文です。これはストリーミング表示の確認です"],
    &["三つ目", "三つ目の発話です。チャンク分割と", "三つ目の発話です。チャンク分割と自動発話を確認しています"],
];

pub struct MockSession {
    stop: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl MockSession {
    pub fn start(host: Arc<dyn SessionHost>) -> Arc<dyn SessionHandle> {
        let stop = Arc::new(AtomicBool::new(false));
        let s2 = Arc::clone(&stop);
        let h = std::thread::Builder::new().name("mock-session".into()).spawn(move || run(host, &s2)).expect("spawn mock-session");
        Arc::new(Self { stop, thread: Mutex::new(Some(h)) })
    }
}

impl SessionHandle for MockSession {
    fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = lock(&self.thread).take() {
            let _ = h.join();
        }
    }
}

/// 停止要求を見ながら眠る。停止なら false。
fn nap(stop: &AtomicBool, secs: f64) -> bool {
    let mut left = secs;
    while left > 0.0 {
        if stop.load(Ordering::SeqCst) {
            return false;
        }
        let d = left.min(0.05);
        std::thread::sleep(Duration::from_secs_f64(d));
        left -= d;
    }
    !stop.load(Ordering::SeqCst)
}

fn run(host: Arc<dyn SessionHost>, stop: &AtomicBool) {
    // モデル準備完了を通知
    if !nap(stop, 0.2) {
        return;
    }
    host.on_asr_model_ready("mock-asr");

    for (i, partials) in SCRIPT.iter().enumerate() {
        let utt = i as u64 + 1;
        for text in *partials {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            // マイクレベル風の値も流す
            host.on_mic_level(0.05, -26.0);
            host.on_asr_partial(utt, text, None);
            if !nap(stop, 0.45) {
                return;
            }
        }
        // 計測フィールドも模擬(話し終わり 280ms 後に VAD が確定、ASR は即時)
        let t = now();
        host.on_asr_final(utt, partials[partials.len() - 1], Timing { speech_end: Some(t - 0.28), vad_end: Some(t), asr_ms: Some(0), audio_ms: Some(1500) });
        if !nap(stop, 1.2) {
            return;
        }
    }

    // 以降も疑似レベルを流し続ける(mock は実マイクを持たないが、レベルメータの UI 経路を常時確認できるようにする)
    let t0 = now();
    while nap(stop, 0.45) {
        let db = -30.0 + 9.0 * ((now() - t0) * 1.7).sin();
        host.on_mic_level(10f64.powf(db / 20.0) as f32, db as f32);
    }
}
