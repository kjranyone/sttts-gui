//! 公開 API だけを使ったモックモードの往復テスト(`Backend` に `GuiMessage` を送り、`BackendMessage` を受ける)。
//!
//! speak → speak_accepted → tts_chunk_start → tts_audio → speak_done の流れと、計測フィールドを検証する。
//! 実モデル・実デバイスには触れない(外界は `Platform` の偽物)。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use base64::Engine as _;
use serde_json::Value;
use sttts_engine::asr::{AsrEngine, Progress};
use sttts_engine::session::{AudioSource, OnBlock, Vad};
use sttts_engine::tts::TtsEngine;
use sttts_engine::{Backend, BackendOptions, Platform, Sink};
use sttts_protocol::{AudioDeviceInfo, BackendMessage, GuiMessage, PROTOCOL_VERSION};

struct NoHardware;

impl Platform for NoHardware {
    fn create_tts(&self, _cfg: &Value, _p: &dyn Fn(&str)) -> Result<Arc<dyn TtsEngine>> {
        bail!("mock mode never loads a real TTS")
    }
    fn create_asr(&self, _cfg: &Value, _p: Progress) -> Result<Arc<dyn AsrEngine>> {
        bail!("mock mode never loads a real ASR")
    }
    fn open_source(&self, _c: &Value, _w: &[PathBuf], _b: OnBlock, _e: Option<Box<dyn FnOnce() + Send>>) -> Result<Box<dyn AudioSource>> {
        bail!("no audio device in tests")
    }
    fn create_vad(&self, _c: &Value) -> Result<Box<dyn Vad>> {
        bail!("no vad in tests")
    }
    fn list_devices(&self) -> (Vec<AudioDeviceInfo>, Vec<AudioDeviceInfo>) {
        (Vec::new(), Vec::new())
    }
}

fn start_mock() -> (Backend, Arc<Mutex<Vec<BackendMessage>>>) {
    let msgs: Arc<Mutex<Vec<BackendMessage>>> = Arc::default();
    let m2 = Arc::clone(&msgs);
    let sink = Sink::new(move |m| m2.lock().unwrap().push(m));
    let opts = BackendOptions { mock: true, save_wavs: false, load_user_file: false, ..Default::default() };
    (Backend::start(opts, Arc::new(NoHardware), sink), msgs)
}

fn wait_for(msgs: &Mutex<Vec<BackendMessage>>, timeout: Duration, pred: impl Fn(&BackendMessage) -> bool) -> bool {
    let end = Instant::now() + timeout;
    while Instant::now() < end {
        if msgs.lock().unwrap().iter().any(&pred) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

#[test]
fn mock_backend_speak_roundtrip() {
    let (backend, msgs) = start_mock();
    assert!(wait_for(&msgs, Duration::from_secs(5), |m| matches!(m, BackendMessage::Hello { .. })));
    {
        let g = msgs.lock().unwrap();
        let BackendMessage::Hello { protocol, mock, models, .. } = &g[0] else { panic!("first message must be hello") };
        assert_eq!(*protocol, PROTOCOL_VERSION);
        assert!(*mock);
        assert!(!models.is_empty());
    }
    backend.send(GuiMessage::Speak {
        text: "こんにちは。これはテストです。今日も良い一日になりますように。".into(),
        caption: None,
        ref_wavs: None,
        seed: None,
        tag: None,
        delivery: None,
    });
    assert!(wait_for(&msgs, Duration::from_secs(30), |m| matches!(m, BackendMessage::SpeakDone { .. })));
    backend.shutdown();

    // "こんにちは。"(6文字, first_min_chars=1で確定) + 残り25文字(min_chars=16で確定) = 2チャンク
    let expected_chunks = 2usize;
    let g = msgs.lock().unwrap();
    let (mut accepted, mut starts, mut audio, mut chunk_done) = (0, 0, 0, 0);
    for m in g.iter() {
        match m {
            BackendMessage::SpeakAccepted { origin, .. } => {
                assert_eq!(origin, "manual");
                accepted += 1;
            }
            BackendMessage::TtsChunkStart { text, .. } => {
                assert!(!text.is_empty());
                starts += 1;
            }
            BackendMessage::TtsAudio { wav_base64, sample_rate, chunk, first_chunk, first_chunk_ms, e2e_ms, rtf, .. } => {
                // 計測フィールド: 先頭チャンクだけ first_chunk_ms を持ち、手動発話なので e2e_ms は無い
                assert_eq!(*first_chunk, *chunk == 0);
                assert_eq!(first_chunk_ms.is_some(), *chunk == 0);
                assert!(e2e_ms.is_none());
                assert!(rtf.is_some());
                let wav = base64::engine::general_purpose::STANDARD.decode(wav_base64).expect("valid base64");
                assert!(wav.len() > 44 && &wav[..4] == b"RIFF", "wav header");
                assert_eq!(*sample_rate, 48000);
                audio += 1;
            }
            BackendMessage::TtsChunkDone { chunk, .. } => {
                assert_eq!(chunk_done, *chunk as usize);
                chunk_done += 1;
            }
            BackendMessage::SpeakDone { chunks, cancelled, failed, .. } => {
                assert!(!cancelled && !failed);
                assert_eq!(*chunks as usize, expected_chunks);
            }
            BackendMessage::Error { message, .. } => panic!("backend error: {message}"),
            BackendMessage::Log { level, message } => assert_ne!(level, "error", "{message}"),
            _ => {}
        }
    }
    assert_eq!((accepted, starts, audio, chunk_done), (1, expected_chunks, expected_chunks, expected_chunks));
}

#[test]
fn mock_session_streams_partials_and_auto_speaks() {
    let (backend, msgs) = start_mock();
    backend.send(GuiMessage::StartSession);
    assert!(wait_for(&msgs, Duration::from_secs(10), |m| matches!(m, BackendMessage::AsrFinal { .. })));
    // 自動発話(既定 ON)で確定文が合成される
    assert!(wait_for(&msgs, Duration::from_secs(30), |m| matches!(m, BackendMessage::SpeakDone { .. })));
    backend.send(GuiMessage::StopSession);
    assert!(wait_for(&msgs, Duration::from_secs(10), |m| matches!(m, BackendMessage::State { mic_running: false, .. })));
    backend.shutdown();
    let g = msgs.lock().unwrap();
    assert!(g.iter().any(|m| matches!(m, BackendMessage::AsrPartial { .. })));
    assert!(g.iter().any(|m| matches!(m, BackendMessage::MicLevel { .. })));
    assert!(g.iter().any(|m| matches!(m, BackendMessage::SpeakAccepted { origin, .. } if origin == "auto")));
}

#[test]
fn shutdown_is_prompt_and_idempotent() {
    let (backend, _msgs) = start_mock();
    let t0 = Instant::now();
    backend.shutdown();
    backend.shutdown();
    assert!(t0.elapsed() < Duration::from_secs(5));
}
