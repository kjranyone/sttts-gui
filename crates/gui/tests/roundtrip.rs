//! モックバックエンドとの stdio NDJSON 往復統合テスト。
//!
//! 実際に `python -m sttts_server --mock --stdio` を子プロセス起動し、
//! speak → speak_accepted → tts_chunk_start → tts_audio → speak_done の流れを検証する。
//! モックエンジンは標準ライブラリのみで動くため、Python さえあれば実行できる。

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine as _;
use sttts_protocol::{AnyMessage, BackendMessage, GuiMessage};

fn python_program() -> Option<String> {
    if let Ok(p) = std::env::var("STTTS_PYTHON") {
        return Some(p);
    }
    // リポジトリの venv を優先
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend/.venv/Scripts/python.exe");
    if root.exists() {
        return Some(root.to_string_lossy().into_owned());
    }
    which_python()
}

fn which_python() -> Option<String> {
    for probe in ["python", "python3", "py"] {
        if Command::new(probe)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return Some(probe.to_string());
        }
    }
    None
}

#[test]
fn mock_backend_speak_roundtrip() {
    let Some(python) = python_program() else {
        eprintln!("python が見つからないためスキップ");
        return;
    };
    let backend_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../backend/src")
        .canonicalize()
        .expect("backend/src");

    let mut child = Command::new(&python)
        .args(["-m", "sttts_server", "--stdio", "--mock", "--output-dir"])
        .arg(std::env::temp_dir().join("sttts-test-output"))
        .env("PYTHONPATH", &backend_src)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("backend spawn");

    // 万一ハングしたときの保険(テスト完了フラグが立つまで待ってから kill を試みる)
    let pid = child.id();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watchdog_done = done.clone();
    let watchdog = std::thread::spawn(move || {
        // テスト完了フラグが立ったら即終了。60秒経っても立たなければ kill する。
        for _ in 0..600 {
            if watchdog_done.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .status();
        }
    });

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().flatten() {
            eprintln!("[py] {line}");
        }
    });

    let send = |stdin: &mut std::process::ChildStdin, msg: &GuiMessage| {
        let line = serde_json::to_string(msg).unwrap();
        stdin.write_all(line.as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    };

    let deadline = Instant::now() + Duration::from_secs(45);
    let mut got_hello = false;
    let mut got_accepted = false;
    let mut got_chunk_start = false;
    let mut got_audio = false;
    let mut got_done = false;
    let mut sample_rate = 0u32;
    let mut chunk_done_seen = 0usize;
    // "こんにちは。"(6文字, first_min_chars=1で確定) + 残り25文字(min_chars=16で確定) = 2チャンク
    let expected_chunks = 2usize;

    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    while Instant::now() < deadline && !got_done {
        line.clear();
        let n = reader.read_line(&mut line).expect("read stdout");
        if n == 0 {
            panic!("backend closed stdout unexpectedly");
        }
        let msg: AnyMessage = serde_json::from_str(line.trim()).expect("valid NDJSON");
        match msg {
            AnyMessage::Known(BackendMessage::Hello { protocol, mock, models, .. }) => {
                assert_eq!(protocol, sttts_protocol::PROTOCOL_VERSION);
                assert!(mock);
                assert!(!models.is_empty());
                got_hello = true;
                send(
                    &mut stdin,
                    &GuiMessage::Speak {
                        text: "こんにちは。これはテストです。今日も良い一日になりますように。".into(),
                        caption: None,
                        ref_wavs: None,
                        seed: None,
                        tag: None,
                    },
                );
            }
            AnyMessage::Known(BackendMessage::SpeakAccepted { origin, .. }) => {
                assert_eq!(origin, "manual");
                got_accepted = true;
            }
            AnyMessage::Known(BackendMessage::TtsChunkStart { text, .. }) => {
                assert!(!text.is_empty());
                got_chunk_start = true;
            }
            AnyMessage::Known(BackendMessage::TtsAudio { wav_base64, sample_rate: sr, .. }) => {
                let wav = base64::engine::general_purpose::STANDARD
                    .decode(&wav_base64)
                    .expect("valid base64");
                assert!(wav.len() > 44, "wav too small");
                assert_eq!(&wav[..4], b"RIFF", "wav header");
                sample_rate = sr;
                got_audio = true;
            }
            AnyMessage::Known(BackendMessage::TtsChunkDone { chunk, gen_ms, .. }) => {
                assert!(gen_ms > 0);
                assert_eq!(chunk_done_seen, chunk as usize);
                chunk_done_seen += 1;
            }
            AnyMessage::Known(BackendMessage::SpeakDone { chunks, cancelled, failed, .. }) => {
                assert!(!cancelled);
                assert!(!failed);
                assert_eq!(chunks as usize, expected_chunks);
                got_done = true;
            }
            AnyMessage::Known(BackendMessage::Log { level, message }) => {
                if level == "error" {
                    panic!("backend error log: {message}");
                }
            }
            AnyMessage::Known(BackendMessage::Error { message, .. }) => {
                panic!("backend error: {message}");
            }
            AnyMessage::Known(other) => {
                eprintln!("event: {}", other_msg_type(&other));
            }
            AnyMessage::Unknown(v) => {
                panic!("unexpected unknown message: {v}");
            }
        }
    }

    assert!(got_hello, "no hello");
    assert!(got_accepted, "no speak_accepted");
    assert!(got_chunk_start, "no tts_chunk_start");
    assert!(got_audio, "no tts_audio");
    assert!(got_done, "no speak_done");
    assert_eq!(chunk_done_seen, expected_chunks, "chunk_done count mismatch");
    assert_eq!(sample_rate, 48000);

    send(&mut stdin, &GuiMessage::Shutdown);
    let _ = child.wait();
    done.store(true, std::sync::atomic::Ordering::SeqCst);
    watchdog.join().ok();
}

fn other_msg_type(_m: &BackendMessage) -> &'static str {
    "other"
}
