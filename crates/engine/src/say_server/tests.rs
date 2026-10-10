use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;

use super::*;
use crate::tts::MockTts;

fn paths(dir: &Path) -> ServerPaths {
    ServerPaths { registration: dir.join("server.json"), lock: dir.join("server.lock") }
}

fn request(dir: &Path, command: Value) -> RunRequest {
    RunRequest { cwd: dir.to_path_buf(), root: dir.to_path_buf(), cfg: json!({}), lang: "en".into(), command }
}

/// 偽の常駐プロセス: コマンドの `text` を返し、ログを 2 行流す。TTS の読み込み回数を数える
fn start(p: &ServerPaths, idle: Duration, loads: Arc<AtomicUsize>) -> JoinHandle<Result<()>> {
    let p = p.clone();
    std::thread::spawn(move || {
        let mut cache = EngineCache::default();
        serve(&p, idle, move |req, log| {
            let model = req.cfg["tts"]["model"].as_str().unwrap_or("m1").to_string();
            let loads = Arc::clone(&loads);
            cache.get_or_load(&model, || {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(MockTts::new(&model, 0.0, 0.0)) as Arc<dyn TtsEngine>)
            })?;
            log("loading");
            log("synthesizing");
            if req.command["fail"] == true {
                bail!("bad line");
            }
            Ok(json!({ "said": req.command["text"], "cwd": std::env::current_dir()? }))
        })
    })
}

/// 依頼の処理は作業フォルダ(プロセス全体の設定)を変えるので、依頼を流すテストは 1 つずつ動かす
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn wait_registered(p: &ServerPaths) -> Registration {
    let mut reg = None;
    assert!(wait_until(Duration::from_secs(5), || {
        reg = read_registration(p);
        reg.is_some()
    }));
    reg.unwrap()
}

fn no_spawn() -> Result<()> {
    bail!("spawn not expected")
}

#[test]
fn requests_reuse_the_loaded_engine_and_stream_logs() {
    let _serial = crate::util::lock(&SERIAL);
    let dir = tempfile::tempdir().unwrap();
    let p = paths(dir.path());
    let loads = Arc::new(AtomicUsize::new(0));
    let h = start(&p, Duration::from_secs(30), Arc::clone(&loads));
    wait_registered(&p);

    let logs = std::cell::RefCell::new(Vec::new());
    let log = |m: &str| logs.borrow_mut().push(m.to_string());
    for text in ["one", "two"] {
        let v = run_remote(&p, &request(dir.path(), json!({ "text": text })), no_spawn, &log).unwrap().unwrap();
        assert_eq!(v["said"], text);
        // 呼び出し側の作業フォルダで実行する
        assert_eq!(std::fs::canonicalize(v["cwd"].as_str().unwrap()).unwrap(), std::fs::canonicalize(dir.path()).unwrap());
    }
    assert_eq!(*logs.borrow(), ["loading", "synthesizing", "loading", "synthesizing"]);
    assert_eq!(loads.load(Ordering::SeqCst), 1, "the engine is loaded once and reused");

    // モデルが変わったら読み込み直す
    let mut req = request(dir.path(), json!({ "text": "three" }));
    req.cfg = json!({ "tts": { "model": "m2" } });
    run_remote(&p, &req, no_spawn, &|_| {}).unwrap().unwrap();
    assert_eq!(loads.load(Ordering::SeqCst), 2);

    // 合成のエラーはそのまま呼び出し側のエラーになる(常駐プロセスは動き続ける)
    let err = run_remote(&p, &request(dir.path(), json!({ "fail": true })), no_spawn, &|_| {}).unwrap_err();
    assert!(err.to_string().contains("bad line"), "{err}");
    assert!(run_remote(&p, &request(dir.path(), json!({ "text": "four" })), no_spawn, &|_| {}).unwrap().is_some());

    assert!(stop(&p).unwrap());
    h.join().unwrap().unwrap();
    assert!(read_registration(&p).is_none(), "registration is removed on exit");
    assert!(!stop(&p).unwrap(), "nothing to stop");
}

#[test]
fn the_server_exits_when_idle_and_releases_its_lock() {
    let dir = tempfile::tempdir().unwrap();
    let p = paths(dir.path());
    let h = start(&p, Duration::from_millis(300), Arc::new(AtomicUsize::new(0)));
    wait_registered(&p);
    h.join().unwrap().unwrap();
    assert!(read_registration(&p).is_none());
    assert!(crate::util::try_lock_file(&p.lock).is_ok());
}

#[test]
fn only_one_server_runs_at_a_time() {
    let dir = tempfile::tempdir().unwrap();
    let p = paths(dir.path());
    let first = start(&p, Duration::from_secs(30), Arc::new(AtomicUsize::new(0)));
    let reg = wait_registered(&p);
    // 2 つ目はロックが取れないのですぐ戻り、登録を書き換えない
    let second = start(&p, Duration::from_secs(30), Arc::new(AtomicUsize::new(0)));
    second.join().unwrap().unwrap();
    assert_eq!(read_registration(&p), Some(reg));
    assert!(stop(&p).unwrap());
    first.join().unwrap().unwrap();
}

#[test]
fn a_request_with_a_wrong_token_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let p = paths(dir.path());
    let h = start(&p, Duration::from_secs(30), Arc::new(AtomicUsize::new(0)));
    let reg = wait_registered(&p);
    let mut s = connect(&reg).unwrap();
    send_line(&mut s, &Request::Stop { token: "nope".into(), only_if_idle: false, sent_ms: now_ms() }).unwrap();
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line).unwrap();
    assert!(line.contains("bad request"), "{line}");
    assert!(read_registration(&p).is_some(), "still running");
    assert!(stop(&p).unwrap());
    h.join().unwrap().unwrap();
}

#[test]
fn stop_if_idle_stops_a_waiting_server_but_not_a_busy_one() {
    let _serial = crate::util::lock(&SERIAL);
    let dir = tempfile::tempdir().unwrap();
    let p = paths(dir.path());

    // 合成中: 応答が無いので止めない。合成の後に届いた停止の依頼も無視する
    let p2 = p.clone();
    let h = std::thread::spawn(move || {
        serve(&p2, Duration::from_secs(30), |_, _| {
            std::thread::sleep(Duration::from_millis(1500));
            Ok(json!("done"))
        })
    });
    wait_registered(&p);
    let p3 = p.clone();
    let dir_path = dir.path().to_path_buf();
    let busy = std::thread::spawn(move || run_remote(&p3, &request(&dir_path, json!({})), no_spawn, &|_| {}).unwrap());
    std::thread::sleep(Duration::from_millis(200));
    assert!(!stop_if_idle(&p), "a busy server is not stopped");
    assert_eq!(busy.join().unwrap(), Some(json!("done")));
    std::thread::sleep(Duration::from_millis(200));
    assert!(read_registration(&p).is_some(), "the late stop request was ignored");

    // 待機中: 止まる
    assert!(stop_if_idle(&p));
    h.join().unwrap().unwrap();
    assert!(read_registration(&p).is_none());
}

#[test]
fn without_a_server_the_caller_synthesizes_itself() {
    let dir = tempfile::tempdir().unwrap();
    let p = paths(dir.path());
    // 起動できなければ None(呼び出し側がその場で合成する)
    assert_eq!(run_remote(&p, &request(dir.path(), json!({})), no_spawn, &|_| {}).unwrap(), None);
    // 応答しない古い登録は無視して起動し直す
    crate::say::write_atomic(&p.registration, &serde_json::to_vec(&Registration { port: 1, token: "x".into(), pid: 1, build: "old".into() }).unwrap()).unwrap();
    let spawned = AtomicUsize::new(0);
    let r = run_remote(&p, &request(dir.path(), json!({})), || {
        spawned.fetch_add(1, Ordering::SeqCst);
        bail!("no exe in tests")
    }, &|_| {});
    assert_eq!(r.unwrap(), None);
    assert_eq!(spawned.load(Ordering::SeqCst), 1);
}

#[test]
fn engine_cache_drops_the_old_model_before_loading_a_new_one() {
    let mut cache = EngineCache::default();
    let a = cache.get_or_load("a", || Ok(Arc::new(MockTts::new("a", 0.0, 0.0)) as Arc<dyn TtsEngine>)).unwrap();
    let weak = Arc::downgrade(&a);
    drop(a);
    cache
        .get_or_load("b", || {
            assert!(weak.upgrade().is_none(), "the previous model is released before loading");
            Ok(Arc::new(MockTts::new("b", 0.0, 0.0)) as Arc<dyn TtsEngine>)
        })
        .unwrap();
    assert!(cache.get_or_load("b", || bail!("must reuse")).is_ok());
}
