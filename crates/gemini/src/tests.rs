//! 偽 Gemini サーバ(ローカルホストの WebSocket)でプロトコルを検証する。
//! 応答順は実 API(gemini-3.5-transcribe-live、サーバ VAD 無効)に合わせる:
//! interim … → activityEnd 後に inputTranscription → generationComplete(turnComplete は来ない)。
//! 実 API・実マイク・GPU には触れない。

use std::net::TcpListener;
use std::sync::atomic::AtomicBool;

use tungstenite::handshake::server::{Request, Response};

use super::*;

#[derive(Clone)]
struct Script {
    /// None なら setup を受けたら応答せず切断する
    setup_reply: Option<String>,
    interim: Option<&'static str>,
    /// activityEnd 後に送る inputTranscription(順に)
    finals: Vec<&'static str>,
    complete: bool,
    /// activityEnd 後に生で送る文字列(不正 JSON・error など)
    raw_after_end: Option<String>,
    /// activityEnd 後に最後に Close を送る
    close_after_end: bool,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            setup_reply: Some(r#"{"setupComplete":{}}"#.into()),
            interim: Some("こん"),
            finals: vec!["こんにちは"],
            complete: true,
            raw_after_end: None,
            close_after_end: false,
        }
    }
}

#[derive(Default)]
struct Log {
    path: String,
    api_key: Option<String>,
    msgs: Vec<Value>,
    connections: usize,
    /// サーバ側でクライアントの切断を観測した
    client_gone: bool,
}

struct FakeServer {
    endpoint: String,
    log: Arc<Mutex<Log>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl FakeServer {
    fn start(script: Script) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Mutex::new(Log::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (l, s) = (log.clone(), stop.clone());
        let handle = thread::spawn(move || serve(listener, script, l, s));
        Self { endpoint: format!("ws://127.0.0.1:{port}"), log, stop, handle: Some(handle) }
    }

    fn msgs(&self) -> Vec<Value> {
        lock(&self.log).msgs.clone()
    }

    fn kinds(&self) -> Vec<String> {
        self.msgs()
            .iter()
            .map(|m| match m.get("realtime_input") {
                Some(ri) => ri.as_object().unwrap().keys().next().unwrap().clone(),
                None => "setup".to_owned(),
            })
            .collect()
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn serve(listener: TcpListener, script: Script, log: Arc<Mutex<Log>>, stop: Arc<AtomicBool>) {
    let stream = loop {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        match listener.accept() {
            Ok((s, _)) => break s,
            Err(_) => thread::sleep(Duration::from_millis(5)),
        }
    };
    stream.set_nonblocking(false).unwrap();
    let l2 = log.clone();
    #[allow(clippy::result_large_err)] // accept_hdr のコールバック型が固定
    let cb = move |req: &Request, resp: Response| {
        let mut l = lock(&l2);
        l.connections += 1;
        l.path = req.uri().path().to_owned();
        l.api_key = req.headers().get("x-goog-api-key").and_then(|v| v.to_str().ok()).map(str::to_owned);
        Ok(resp)
    };
    let Ok(mut ws) = tungstenite::accept_hdr(stream, cb) else { return };
    ws.get_ref().set_read_timeout(Some(Duration::from_millis(20))).unwrap();
    let mut got_audio = false;
    loop {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        let msg = match ws.read() {
            Ok(m) => m,
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) =>
            {
                continue;
            }
            Err(_) => {
                lock(&log).client_gone = true;
                return;
            }
        };
        let Message::Text(t) = msg else {
            if matches!(msg, Message::Close(_)) {
                lock(&log).client_gone = true;
                return;
            }
            continue;
        };
        let v: Value = serde_json::from_str(t.as_str()).unwrap();
        lock(&log).msgs.push(v.clone());
        if v.get("setup").is_some() {
            match &script.setup_reply {
                Some(r) => ws.send(Message::text(r.clone())).unwrap(),
                None => {
                    let _ = ws.close(None);
                    let _ = ws.flush();
                    return;
                }
            }
            continue;
        }
        let ri = &v["realtime_input"];
        if ri.get("audio").is_some() && !got_audio {
            got_audio = true;
            if let Some(i) = script.interim {
                // 実サーバはバイナリフレームで JSON を返す
                let m = json!({"serverContent": {"interimInputTranscription": {"text": i}}});
                ws.send(Message::binary(m.to_string().into_bytes())).unwrap();
            }
        }
        if ri.get("activityEnd").is_some() {
            for f in &script.finals {
                let m = json!({"serverContent": {"inputTranscription": {"text": f}}});
                ws.send(Message::binary(m.to_string().into_bytes())).unwrap();
            }
            if script.complete {
                let m = json!({"serverContent": {"generationComplete": true}});
                ws.send(Message::text(m.to_string())).unwrap();
            }
            if let Some(raw) = &script.raw_after_end {
                ws.send(Message::text(raw.clone())).unwrap();
            }
            if script.close_after_end {
                let _ = ws.close(None);
                let _ = ws.flush();
            }
        }
    }
}

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let until = Instant::now() + Duration::from_secs(3);
    while !cond() {
        assert!(Instant::now() < until, "待機がタイムアウト: {what}");
        thread::sleep(Duration::from_millis(5));
    }
}

fn client(endpoint: &str, timeout_s: f64) -> GeminiLive {
    GeminiLive::load(
        GeminiOptions {
            api_key: Some("dummy-key".into()),
            timeout_s,
            endpoint: Some(endpoint.into()),
            ..Default::default()
        },
        &|_| {},
    )
    .unwrap()
}

fn block() -> Vec<f32> {
    vec![0.1; 1600]
}

fn no_partial() -> PartialCallback {
    Arc::new(|_, _| {})
}

// ---------- Python 版 test_gemini_stream.py の移植 ----------

#[test]
fn stream_sends_audio_before_vad_end() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    let partials = Arc::new(Mutex::new(Vec::<(u64, String)>::new()));
    let p = partials.clone();
    g.begin_stream(3, Arc::new(move |u, t| lock(&p).push((u, t.to_owned())))).unwrap();
    // 100ms 分(3200 byte)を送ると、VAD 終了前に WebSocket へ音声が流れる
    g.feed_stream(3, &block());
    wait_until("音声が届く", || srv.kinds().iter().any(|k| k == "audio"));
    assert!(!srv.kinds().iter().any(|k| k == "activityEnd"));
    g.end_stream(3);
    assert_eq!(g.finish_stream(3, &block()).unwrap(), "こんにちは");
    assert_eq!(*lock(&partials), vec![(3, "こん".to_owned())]);
    assert!(lock(&g.streams).is_empty());

    // 発話区切りはローカル VAD が決める: サーバ自動 VAD 無効 + activityStart/End を明示送信
    let msgs = srv.msgs();
    let rtc = &msgs[0]["setup"]["realtimeInputConfig"];
    assert_eq!(rtc["automatic_activity_detection"], json!({"disabled": true}));
    let kinds = srv.kinds();
    assert_eq!(kinds.first().map(String::as_str), Some("setup"));
    assert_eq!(kinds[1], "activityStart");
    assert_eq!(kinds.last().map(String::as_str), Some("activityEnd"));
    assert!(!kinds.iter().any(|k| k == "audioStreamEnd" || k == "audio_stream_end"));
}

#[test]
fn stream_keeps_final_text_without_completion_signal() {
    // 確定テキスト受信後に完了シグナルが来なくても、タイムアウトで捨てずに返す
    let srv = FakeServer::start(Script { complete: false, ..Default::default() });
    let g = client(&srv.endpoint, 0.5);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    assert_eq!(g.finish_stream(1, &block()).unwrap(), "こんにちは");
    assert!(lock(&g.streams).is_empty());
}

#[test]
fn stream_times_out_without_any_final() {
    let srv = FakeServer::start(Script { finals: vec![], complete: false, ..Default::default() });
    let g = client(&srv.endpoint, 0.5);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    let err = g.finish_stream(1, &block()).unwrap_err().to_string();
    assert!(err.contains("timed out"), "{err}");
    // タイムアウト後は接続も解放されている
    wait_until("サーバがクライアント切断を観測", || lock(&srv.log).client_gone);
}

#[test]
fn abort_all_streams_stops_open_stream() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    let stream = g.get(1).unwrap();
    wait_until("接続", || lock(&srv.log).connections == 1);
    g.abort_all_streams();
    // 戻り時点でスレッドは終了している
    assert!(lock(&stream.outcome).done);
    assert!(lock(&stream.thread).is_none());
    assert!(lock(&g.streams).is_empty());
    wait_until("サーバがクライアント切断を観測", || lock(&srv.log).client_gone);
}

// ---------- ワイヤ形式 ----------

#[test]
fn wire_format_matches_sdk() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &[0.5, -0.5, 1.5, -1.5, 0.0]);
    assert_eq!(g.finish_stream(1, &[]).unwrap(), "こんにちは");

    {
        let l = lock(&srv.log);
        // SDK 実出力どおり(二重スラッシュを含む)
        assert_eq!(
            l.path,
            "//ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        );
        assert_eq!(l.api_key.as_deref(), Some("dummy-key"));
    }
    let msgs = srv.msgs();
    assert_eq!(
        msgs[0],
        json!({"setup": {
            "model": "models/gemini-3.5-transcribe-live",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": {"language_codes": ["ja-JP"], "mode": "VERBATIM"},
            "realtimeInputConfig": {"automatic_activity_detection": {"disabled": true}},
        }})
    );
    assert_eq!(msgs[1], json!({"realtime_input": {"activityStart": {}}}));
    let audio = &msgs[2]["realtime_input"]["audio"];
    assert_eq!(audio["mime_type"], "audio/pcm;rate=16000");
    let pcm = base64::engine::general_purpose::STANDARD.decode(audio["data"].as_str().unwrap()).unwrap();
    let samples: Vec<i16> = pcm.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    // 0.5*32767=16383.5 → 0 方向へ切り捨て、範囲外はクリップ
    assert_eq!(samples, vec![16383, -16383, 32767, -32767, 0]);
    assert_eq!(msgs[3], json!({"realtime_input": {"activityEnd": {}}}));
    assert_eq!(msgs.len(), 4);
}

#[test]
fn audio_is_batched_into_100ms_chunks() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    // 100ms(1600 サンプル)ずつ 5 回 → 3200 byte 以上で都度送信、残りは終了時に flush
    for _ in 0..5 {
        g.feed_stream(1, &block());
    }
    g.finish_stream(1, &[]).unwrap();
    let total: usize = srv
        .msgs()
        .iter()
        .filter_map(|m| m.pointer("/realtime_input/audio/data").and_then(Value::as_str))
        .map(|d| base64::engine::general_purpose::STANDARD.decode(d).unwrap().len())
        .sum();
    assert_eq!(total, 5 * 3200);
}

#[test]
fn small_frames_are_flushed_after_100ms_without_end() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &vec![0.1; 160]); // 10ms 分: 3200 byte 未満
    wait_until("時間経過で flush", || srv.kinds().iter().any(|k| k == "audio"));
    g.abort_stream(1);
}

#[test]
fn transcription_fragments_accumulate() {
    // 前方一致なら置換、そうでなければ連結
    let srv = FakeServer::start(Script { finals: vec!["こん", "こんにちは", "世界"], ..Default::default() });
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    assert_eq!(g.finish_stream(1, &[]).unwrap(), "こんにちは 世界");
}

#[test]
fn transcribe_utterance_one_shot() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    assert_eq!(g.transcribe_utterance(&block()).unwrap(), "こんにちは");
}

// ---------- 失敗系 ----------

#[test]
fn connection_refused_is_reported_by_finish() {
    // 閉じたポートへ接続する
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    // Windows は拒否応答まで約 2 秒かかる
    let g = client(&format!("ws://127.0.0.1:{port}"), 10.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    let err = g.finish_stream(1, &block()).unwrap_err().to_string();
    assert!(err.starts_with("Gemini Live transcription failed:"), "{err}");
    assert!(!err.contains("dummy-key"));
    assert!(lock(&g.streams).is_empty());
}

#[test]
fn server_closing_before_setup_reply_fails() {
    let srv = FakeServer::start(Script { setup_reply: None, ..Default::default() });
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    let err = g.finish_stream(1, &[]).unwrap_err().to_string();
    assert!(err.starts_with("Gemini Live transcription failed:"), "{err}");
}

#[test]
fn invalid_json_from_server_fails() {
    let srv = FakeServer::start(Script { finals: vec![], complete: false, raw_after_end: Some("not json {".into()), ..Default::default() });
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    let err = g.finish_stream(1, &[]).unwrap_err().to_string();
    assert!(err.contains("Failed to parse response"), "{err}");
}

#[test]
fn invalid_json_in_setup_reply_fails() {
    let srv = FakeServer::start(Script { setup_reply: Some("garbage".into()), ..Default::default() });
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    let err = g.finish_stream(1, &[]).unwrap_err().to_string();
    assert!(err.contains("Failed to parse response"), "{err}");
}

#[test]
fn api_error_message_fails() {
    let srv = FakeServer::start(Script {
        finals: vec![],
        complete: false,
        raw_after_end: Some(r#"{"error":{"code":429,"message":"quota exceeded"}}"#.into()),
        ..Default::default()
    });
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    let err = g.finish_stream(1, &[]).unwrap_err().to_string();
    assert!(err.contains("quota exceeded"), "{err}");
}

#[test]
fn server_close_after_final_is_an_error_like_sdk() {
    // 完了シグナル無しで切断されると SDK は ConnectionClosed を投げる(確定済みテキストがあっても失敗)
    let srv = FakeServer::start(Script { complete: false, close_after_end: true, ..Default::default() });
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    let err = g.finish_stream(1, &[]).unwrap_err().to_string();
    assert!(err.starts_with("Gemini Live transcription failed:"), "{err}");
}

#[test]
fn empty_setup_reply_and_unknown_messages_are_ignored() {
    let srv = FakeServer::start(Script {
        setup_reply: Some("{}".into()),
        raw_after_end: Some(r#"{"usageMetadata":{"totalTokenCount":3}}"#.into()),
        ..Default::default()
    });
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.feed_stream(1, &block());
    assert_eq!(g.finish_stream(1, &[]).unwrap(), "こんにちは");
}

#[test]
fn panicking_partial_callback_does_not_break_stream() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, Arc::new(|_, _| panic!("boom"))).unwrap();
    g.feed_stream(1, &block());
    assert_eq!(g.finish_stream(1, &[]).unwrap(), "こんにちは");
}

// ---------- 管理系 ----------

#[test]
fn finish_unknown_utterance_errors() {
    let g = client("ws://127.0.0.1:1", 1.0);
    let err = g.finish_stream(9, &[]).unwrap_err().to_string();
    assert_eq!(err, "Gemini stream unavailable for utterance 9");
    // 未知の発話への feed/end/abort は無害
    g.feed_stream(9, &block());
    g.end_stream(9);
    g.abort_stream(9);
}

#[test]
fn duplicate_utterance_is_rejected() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    let err = g.begin_stream(1, no_partial()).unwrap_err().to_string();
    assert_eq!(err, "duplicate Gemini utterance 1");
    g.abort_all_streams();
}

#[test]
fn abort_before_connect_completes_is_clean() {
    let srv = FakeServer::start(Script::default());
    let g = client(&srv.endpoint, 2.0);
    g.begin_stream(1, no_partial()).unwrap();
    g.abort_stream(1);
    assert!(lock(&g.streams).is_empty());
}

#[test]
fn drop_aborts_streams() {
    let srv = FakeServer::start(Script::default());
    {
        let g = client(&srv.endpoint, 2.0);
        g.begin_stream(1, no_partial()).unwrap();
        wait_until("接続", || lock(&srv.log).connections == 1);
    }
    wait_until("サーバがクライアント切断を観測", || lock(&srv.log).client_gone);
}

// ---------- load / キー解決 ----------

#[test]
fn load_without_key_errors_with_python_message() {
    // 環境変数に依存しないよう、解決ロジック側を直接検証する
    assert_eq!(resolve_api_key_with(None, |_| None), None);
    assert_eq!(resolve_api_key_with(Some(""), |_| None), None);
    let msg = NO_KEY_MESSAGE;
    assert!(msg.starts_with("Gemini API キーがありません。GUI の「キー」欄に"));
    assert!(msg.ends_with("環境変数 GEMINI_API_KEY でも可)"));
}

#[test]
fn api_key_resolution_order() {
    let env = |k: &str| match k {
        "GEMINI_API_KEY" => Some("g".to_owned()),
        "GOOGLE_API_KEY" => Some("o".to_owned()),
        _ => None,
    };
    assert_eq!(resolve_api_key_with(Some("cfg"), env).as_deref(), Some("cfg"));
    assert_eq!(resolve_api_key_with(None, env).as_deref(), Some("g"));
    assert_eq!(resolve_api_key_with(Some(""), env).as_deref(), Some("g"));
    let only_google = |k: &str| (k == "GOOGLE_API_KEY").then(|| "o".to_owned());
    assert_eq!(resolve_api_key_with(None, only_google).as_deref(), Some("o"));
    let empty_gemini = |k: &str| Some(if k == "GEMINI_API_KEY" { String::new() } else { "o".to_owned() });
    assert_eq!(resolve_api_key_with(None, empty_gemini).as_deref(), Some("o"));
}

#[test]
fn language_mode_and_model_normalization() {
    assert_eq!(normalize_language("ja"), "ja-JP");
    assert_eq!(normalize_language("ja-JP"), "ja-JP");
    assert_eq!(normalize_language("en"), "en");
    let progress = Mutex::new(Vec::<String>::new());
    let g = GeminiLive::load(
        GeminiOptions { model: String::new(), api_key: Some("k".into()), language: "ja".into(), mode: "smart".into(), ..Default::default() },
        &|m| lock(&progress).push(m.to_owned()),
    )
    .unwrap();
    assert_eq!(g.model_id(), DEFAULT_MODEL);
    assert_eq!(g.language, "ja-JP");
    assert_eq!(g.mode, "SMART");
    assert_eq!(
        *lock(&progress),
        vec![
            "Gemini 接続準備: gemini-3.5-transcribe-live (SMART)".to_owned(),
            "ASR準備完了: Gemini (SMART)".to_owned()
        ]
    );
    let g = GeminiLive::load(
        GeminiOptions { api_key: Some("k".into()), mode: "bogus".into(), ..Default::default() },
        &|_| {},
    )
    .unwrap();
    assert_eq!(g.mode, "VERBATIM");
    // Debug にキーを出さない
    assert!(!format!("{g:?}").contains("api_key"));
}
