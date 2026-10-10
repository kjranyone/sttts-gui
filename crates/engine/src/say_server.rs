//! `sttts-say` の常駐プロセス。モデルを GPU に読み込んだまま待ち、`speak` / `render` / `audition` の依頼を順番に
//! 処理する。呼び出しごとのモデルの読み込みとカーネルのコンパイル(1 回 4〜6 秒)を省くため。
//!
//! - 依頼は 127.0.0.1 の TCP で、1 行の JSON を送り、ログ(`{"log": ..}`)を受けたあと結果(`{"result": ..}` か
//!   `{"error": ..}`)を受ける。接続先と合言葉(token)は一時フォルダの登録ファイルにある(利用者ごとの一時フォルダ)。
//! - 常駐プロセスは最初の依頼で自動的に起動し、[`IDLE_EXIT`] の間依頼が無ければ終わる(GPU を解放する)。
//!   `sttts-say stop` でも止まる。同時に動くのは 1 つだけ(サーバーロック)。
//! - 実行ファイルが更新されていたら(`build` が違えば)古い常駐プロセスを止めて起動し直す。
//! - GUI が GPU を使おうとしてロックが取れないとき、待機中の常駐プロセスには終わってもらう([`stop_if_idle`])。
//!   合成中なら終わらない(GUI は従来どおりエラーを出す)。

use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sttts_i18n::tr;

use crate::tts::TtsEngine;

/// 依頼が無いまま、これだけ経ったら常駐プロセスは終わる
pub const IDLE_EXIT: Duration = Duration::from_secs(10 * 60);
/// 起動した常駐プロセスが受け付けを始めるまで待つ上限
const START_TIMEOUT: Duration = Duration::from_secs(30);
/// 接続の上限(常駐プロセスは同じ機械にいるので短くてよい)
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// 登録ファイルとサーバーロックの場所
#[derive(Debug, Clone)]
pub struct ServerPaths {
    pub registration: PathBuf,
    pub lock: PathBuf,
}

impl Default for ServerPaths {
    /// GPU ロックと同じ一時フォルダ(GPU と同じく、利用者の機械に 1 つ)
    fn default() -> Self {
        let dir = std::env::temp_dir();
        Self { registration: dir.join("sttts-say-server.json"), lock: dir.join("sttts-say-server.lock") }
    }
}

/// 常駐プロセスの登録(接続先・合言葉・どの実行ファイルか)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Registration {
    pub port: u16,
    pub token: String,
    pub pid: u32,
    /// 実行ファイルの版(version と実行ファイルの更新時刻)。違えば起動し直す
    pub build: String,
}

/// 合成の依頼。呼び出し側の作業フォルダで実行するので、相対パスも出力も直接実行したときと同じになる
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRequest {
    pub cwd: PathBuf,
    pub root: PathBuf,
    /// 呼び出し側で読んだ設定(backend.json を重ねたもの)
    pub cfg: Value,
    /// 表示言語(進捗とエラーの文言)
    pub lang: String,
    /// 実行するコマンド(CLI が解釈する)
    pub command: Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Request {
    Run { token: String, run: Box<RunRequest> },
    /// `sent_ms` は送った時刻(UNIX ミリ秒)。`only_if_idle` で、受け付けが遅れた(= 合成中だった)ら止まらない
    Stop { token: String, only_if_idle: bool, sent_ms: u64 },
}

/// 待機中かどうかの判定: 停止の依頼がこれより遅れて届いたら、合成中だったとみなす
const IDLE_REPLY: Duration = Duration::from_millis(500);

thread_local! {
    /// このスレッドが常駐プロセスとして依頼を処理しているときの合言葉(自分自身に停止を頼まないため)
    static SERVING: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// この実行ファイルの版。同じ version でも作り直したら変わるように更新時刻を含める
pub fn current_build() -> String {
    let mtime = std::env::current_exe()
        .and_then(std::fs::metadata)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    format!("{}+{mtime}", env!("CARGO_PKG_VERSION"))
}

/// 常駐プロセスが動いているか(サーバーロックが取られているか)。接続を試さない(空の接続で常駐プロセスを待たせない)
fn alive(paths: &ServerPaths) -> bool {
    crate::util::try_lock_file(&paths.lock).is_err()
}

fn read_registration(paths: &ServerPaths) -> Option<Registration> {
    std::fs::read_to_string(&paths.registration).ok().and_then(|t| serde_json::from_str(&t).ok())
}

fn connect(reg: &Registration) -> Result<TcpStream> {
    let addr = SocketAddr::from(([127, 0, 0, 1], reg.port));
    let s = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)?;
    s.set_nodelay(true).ok();
    Ok(s)
}

fn send_line(stream: &mut TcpStream, v: &impl Serialize) -> Result<()> {
    let mut line = serde_json::to_vec(v)?;
    line.push(b'\n');
    stream.write_all(&line)?;
    stream.flush()?;
    Ok(())
}

// ---------------------------------------------------------------- 常駐プロセス側

/// 読み込んだ TTS の使い回し。モデルの指定(`key`)が変わったら、前のモデルを捨ててから読み込む(GPU メモリを空ける)
#[derive(Default)]
pub struct EngineCache {
    current: Option<(String, Arc<dyn TtsEngine>)>,
}

impl EngineCache {
    pub fn get_or_load(&mut self, key: &str, load: impl FnOnce() -> Result<Arc<dyn TtsEngine>>) -> Result<Arc<dyn TtsEngine>> {
        if let Some((k, e)) = &self.current
            && k == key
        {
            return Ok(Arc::clone(e));
        }
        self.current = None;
        let e = load()?;
        self.current = Some((key.to_string(), Arc::clone(&e)));
        Ok(e)
    }
}

/// 依頼を受けて処理する。`handle` はコマンドを実行して結果を返す(ログは渡した関数へ)。
/// [`IDLE_EXIT`](`idle`)の間依頼が無いか、停止の依頼で戻る。別の常駐プロセスが動いていればすぐ戻る。
pub fn serve(paths: &ServerPaths, idle: Duration, mut handle: impl FnMut(&RunRequest, Rc<dyn Fn(&str)>) -> Result<Value>) -> Result<()> {
    let _lock = match crate::util::try_lock_file(&paths.lock) {
        Ok(f) => f,
        Err(_) => return Ok(()), // 別の常駐プロセスが動いている
    };
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    listener.set_nonblocking(true)?;
    let token = format!("{:032x}", rand::random::<u128>());
    let reg = Registration { port: listener.local_addr()?.port(), token: token.clone(), pid: std::process::id(), build: current_build() };
    crate::say::write_atomic(&paths.registration, &serde_json::to_vec(&reg)?)?;
    SERVING.with(|s| *s.borrow_mut() = Some(token.clone()));
    let mut last = Instant::now();
    loop {
        let stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                if last.elapsed() >= idle {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let stop = serve_one(stream, &token, &mut handle);
        last = Instant::now();
        if stop {
            break;
        }
    }
    SERVING.with(|s| *s.borrow_mut() = None);
    // 自分の登録のときだけ消す(後から起動した別の常駐プロセスの登録は残す)
    if read_registration(paths).is_some_and(|r| r.pid == reg.pid && r.token == reg.token) {
        let _ = std::fs::remove_file(&paths.registration);
    }
    Ok(())
}

/// 1 接続を処理する。停止の依頼なら true
fn serve_one(stream: TcpStream, token: &str, handle: &mut impl FnMut(&RunRequest, Rc<dyn Fn(&str)>) -> Result<Value>) -> bool {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(CONNECT_TIMEOUT));
    let Ok(mut writer) = stream.try_clone() else { return false };
    let mut line = String::new();
    if BufReader::new(&stream).read_line(&mut line).is_err() {
        return false;
    }
    let reply = |w: &mut TcpStream, v: Value| {
        let _ = send_line(w, &v);
    };
    match serde_json::from_str::<Request>(&line) {
        Ok(Request::Stop { token: t, only_if_idle, sent_ms }) if t == token => {
            // 依頼は 1 つずつ処理する。待機中だけ止める依頼が合成の後で届いたら(依頼側はもう諦めている)止まらない
            let stale = only_if_idle && u128::from(now_ms().saturating_sub(sent_ms)) > IDLE_REPLY.as_millis();
            reply(&mut writer, json!({ "stopped": !stale }));
            !stale
        }
        Ok(Request::Run { token: t, run }) if t == token => {
            let log_stream = std::cell::RefCell::new(writer.try_clone().ok());
            let log: Rc<dyn Fn(&str)> = Rc::new(move |m: &str| {
                if let Some(w) = log_stream.borrow_mut().as_mut() {
                    let _ = send_line(w, &json!({ "log": m }));
                }
            });
            let result = std::env::set_current_dir(&run.cwd)
                .with_context(|| run.cwd.display().to_string())
                .and_then(|()| {
                    sttts_i18n::set_lang(sttts_i18n::Lang::from_tag(&run.lang).unwrap_or_else(sttts_i18n::lang));
                    handle(&run, log)
                });
            match result {
                Ok(v) => reply(&mut writer, json!({ "result": v })),
                Err(e) => reply(&mut writer, json!({ "error": format!("{e:#}") })),
            }
            let _ = writer.shutdown(Shutdown::Both);
            false
        }
        _ => {
            reply(&mut writer, json!({ "error": "bad request" }));
            false
        }
    }
}

// ---------------------------------------------------------------- 呼び出し側

/// 常駐プロセスに合成を依頼する。常駐プロセスが無ければ `spawn` で起動して待つ。
/// 常駐プロセスを使えないとき(起動できない・応答しない)は `Ok(None)` を返す(呼び出し側がその場で合成する)。
pub fn run_remote(paths: &ServerPaths, req: &RunRequest, spawn: impl Fn() -> Result<()>, log: &dyn Fn(&str)) -> Result<Option<Value>> {
    let Some(reg) = ensure_server(paths, &spawn) else { return Ok(None) };
    let Ok(mut stream) = connect(&reg) else { return Ok(None) };
    send_line(&mut stream, &Request::Run { token: reg.token.clone(), run: Box::new(req.clone()) })?;
    for line in BufReader::new(stream).lines() {
        let line = line.map_err(|e| anyhow!("{}: {e}", lost()))?;
        let v: Value = serde_json::from_str(&line)?;
        if let Some(m) = v.get("log").and_then(Value::as_str) {
            log(m);
        } else if let Some(r) = v.get("result") {
            return Ok(Some(r.clone()));
        } else if let Some(e) = v.get("error").and_then(Value::as_str) {
            bail!("{e}");
        }
    }
    bail!("{}", lost())
}

fn lost() -> &'static str {
    tr!(
        "The sttts-say background process stopped unexpectedly",
        "sttts-say の常駐プロセスが途中で止まりました",
        "sttts-say 的常驻进程意外停止"
    )
}

/// 動いている常駐プロセス(この実行ファイルと同じ版)。無ければ起動して受け付けを待つ。使えなければ None
fn ensure_server(paths: &ServerPaths, spawn: &dyn Fn() -> Result<()>) -> Option<Registration> {
    if alive(paths)
        && let Some(reg) = read_registration(paths)
    {
        if reg.build == current_build() {
            return Some(reg);
        }
        // 古い版が動いていれば止める(合成中ならそれが終わるまで待つことになる)
        let _ = request_stop(&reg, false, None);
        wait_until(START_TIMEOUT, || !alive(paths));
    }
    spawn().ok()?;
    let mut found = None;
    wait_until(START_TIMEOUT, || {
        found = read_registration(paths).filter(|r| r.build == current_build() && alive(paths));
        found.is_some()
    });
    found
}

fn wait_until(limit: Duration, mut ok: impl FnMut() -> bool) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < limit {
        if ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    ok()
}

/// 停止を頼む。`reply_timeout` までに応答が無ければ(合成中で受け付けられない)false
fn request_stop(reg: &Registration, only_if_idle: bool, reply_timeout: Option<Duration>) -> Result<bool> {
    let mut s = connect(reg)?;
    s.set_read_timeout(reply_timeout)?;
    send_line(&mut s, &Request::Stop { token: reg.token.clone(), only_if_idle, sent_ms: now_ms() })?;
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line)?;
    Ok(serde_json::from_str::<Value>(&line).ok().and_then(|v| v["stopped"].as_bool()).unwrap_or(false))
}

/// `sttts-say stop`: 常駐プロセスを止める(合成中ならそれが終わってから)。止めたら true、動いていなければ false
pub fn stop(paths: &ServerPaths) -> Result<bool> {
    let Some(reg) = read_registration(paths).filter(|_| alive(paths)) else { return Ok(false) };
    request_stop(&reg, false, None)?;
    wait_until(START_TIMEOUT, || !alive(paths));
    Ok(true)
}

/// 待機中の常駐プロセスに終わってもらう(GUI が GPU を使う前)。合成中で応答が無ければ止めない。
/// 止めて常駐プロセスが終わったら true。常駐プロセス自身(の合成中の GPU 初期化)からは何もしない。
pub fn stop_if_idle(paths: &ServerPaths) -> bool {
    let Some(reg) = read_registration(paths) else { return false };
    if SERVING.with(|s| s.borrow().as_deref() == Some(reg.token.as_str())) || !alive(paths) {
        return false;
    }
    match request_stop(&reg, true, Some(IDLE_REPLY)) {
        Ok(true) => wait_until(Duration::from_secs(10), || !alive(paths)),
        _ => false,
    }
}

/// 常駐プロセスを、呼び出し元と切り離して起動する(標準入出力は使わない)。
///
/// 呼び出し元の標準出力・標準エラーを継がせないこと: 継ぐと、呼び出し元の出力をパイプで受けている側
/// (シェルやエージェント)は、常駐プロセスが終わるまで出力の終わりを受け取れず、呼び出しが返らない。
/// Windows は継承できるハンドルを全部子に渡すので、起動の間だけ標準入出力を継承不可にする。
pub fn spawn_detached(exe: &Path, args: &[&str]) -> Result<()> {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        let _guard = NoInheritStdHandles::new();
        cmd.spawn().with_context(|| exe.display().to_string())?;
    }
    #[cfg(unix)]
    {
        // 呼び出し元の端末の Ctrl+C(プロセスグループへのシグナル)で一緒に止まらないように
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        cmd.spawn().with_context(|| exe.display().to_string())?;
    }
    Ok(())
}

/// 標準入出力のハンドルを継承不可にし、戻すときに元の設定に戻す
#[cfg(windows)]
struct NoInheritStdHandles(Vec<(windows::Win32::Foundation::HANDLE, bool)>);

#[cfg(windows)]
impl NoInheritStdHandles {
    fn new() -> Self {
        use windows::Win32::Foundation::{GetHandleInformation, HANDLE_FLAG_INHERIT, HANDLE_FLAGS, SetHandleInformation};
        use windows::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE};
        let mut saved = Vec::new();
        for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            // SAFETY: 自プロセスの標準ハンドルの属性を読み書きするだけ(ハンドルは閉じない)
            unsafe {
                let Ok(h) = GetStdHandle(which) else { continue };
                if h.is_invalid() {
                    continue;
                }
                let mut flags = 0u32;
                if GetHandleInformation(h, &mut flags).is_ok() && flags & HANDLE_FLAG_INHERIT.0 != 0 {
                    let _ = SetHandleInformation(h, HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0));
                    saved.push((h, true));
                }
            }
        }
        Self(saved)
    }
}

#[cfg(windows)]
impl Drop for NoInheritStdHandles {
    fn drop(&mut self) {
        use windows::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
        for (h, _) in &self.0 {
            // SAFETY: new() で継承不可にしたハンドルを元に戻す
            unsafe {
                let _ = SetHandleInformation(*h, HANDLE_FLAG_INHERIT.0, HANDLE_FLAG_INHERIT);
            }
        }
    }
}

#[cfg(test)]
mod tests;
