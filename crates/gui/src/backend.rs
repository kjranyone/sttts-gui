//! Python バックエンドの子プロセス管理(stdio NDJSON)。

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use async_channel::Sender;
use sttts_protocol::{AnyMessage, GuiMessage};

/// リポジトリルート(crates/gui の2つ上)。リリース時は STTTS_ROOT で上書き。
pub fn repo_root() -> PathBuf {
    if let Ok(root) = std::env::var("STTTS_ROOT") {
        return PathBuf::from(root);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root not found")
}

pub struct BackendSpawn {
    pub program: String,
    pub args: Vec<String>,
    pub pythonpath: Option<String>,
}

/// 既定のバックエンド起動コマンドを解決する。
/// 優先順: STTTS_BACKEND_CMD 環境変数 > backend/.venv の python > PATH上の python。
/// venv 無しの場合は PYTHONPATH 指定で動く(モックモードは標準ライブラリのみで動作する)。
pub fn default_spawn(mock: bool, output_dir: &Path) -> BackendSpawn {
    let root = repo_root();
        let output = output_dir.to_string_lossy().into_owned();

    if let Ok(cmd) = std::env::var("STTTS_BACKEND_CMD") {
        // "program arg1 arg2" 形式を単純分割(ベース実装: 空白を含むパスは STTTS_ROOT 併用で回避)
        let mut parts = cmd.split_whitespace().map(str::to_string);
        let program = parts.next().unwrap_or_else(|| "python".into());
        let mut args: Vec<String> = parts.collect();
        args.push("--stdio".into());
        if mock {
            args.push("--mock".into());
        }
        args.push("--output-dir".into());
        args.push(output.clone());
        return BackendSpawn {
            program,
            args,
            pythonpath: Some(root.join("backend/src").to_string_lossy().into_owned()),
        };
    }

    let venv_python = root.join("backend/.venv/Scripts/python.exe");
    let (program, needs_pythonpath) = if venv_python.exists() {
        (venv_python.to_string_lossy().into_owned(), false)
    } else {
        ("python".to_string(), true)
    };

    let mut args = vec![
        "-m".to_string(),
        "sttts_server".to_string(),
        "--stdio".to_string(),
        "--output-dir".to_string(),
        output.clone(),
    ];
    if mock {
        args.push("--mock".to_string());
    }

    BackendSpawn {
        program,
        args,
        pythonpath: needs_pythonpath.then(|| root.join("backend/src").to_string_lossy().into_owned()),
    }
}

pub struct BackendHandle {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
}

impl BackendHandle {
    /// 子プロセスを起動し、stdout/stderr の読み取りスレッドを立てる。
    /// バックエンドメッセージは `tx_events` へ、stderr 行は `tx_stderr` へ流す。
    pub fn spawn(
        cfg: BackendSpawn,
        tx_events: Sender<AnyMessage>,
        tx_stderr: Sender<String>,
    ) -> Result<Self> {
        let mut cmd = Command::new(&cfg.program);
        cmd.args(&cfg.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(pp) = &cfg.pythonpath {
            cmd.env("PYTHONPATH", pp);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("バックエンド起動失敗: {} {:?}", cfg.program, cfg.args))?;
        let stdin = child.stdin.take().context("stdin not piped")?;
        let stdout = child.stdout.take().context("stdout not piped")?;
        let stderr = child.stderr.take().context("stderr not piped")?;

        std::thread::Builder::new()
            .name("backend-stdout".into())
            .spawn({
                let tx_stderr = tx_stderr.clone();
                move || {
                    let reader = BufReader::new(stdout);
                    for line in reader.lines() {
                        match line {
                            Ok(l) if l.trim().is_empty() => continue,
                            Ok(l) => match serde_json::from_str::<AnyMessage>(&l) {
                                Ok(msg) => {
                                    if tx_events.send_blocking(msg).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx_stderr.send_blocking(format!(
                                        "unparseable backend output ({e}): {l}"
                                    ));
                                }
                            },
                            Err(_) => break,
                        }
                    }
                }
            })?;

        std::thread::Builder::new()
            .name("backend-stderr".into())
            .spawn(move || {
                for line in BufReader::new(stderr).lines().flatten() {
                    if tx_stderr.send_blocking(line).is_err() {
                        break;
                    }
                }
            })?;

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
        })
    }

    pub fn send(&self, msg: &GuiMessage) -> Result<()> {
        let line = serde_json::to_string(msg)?;
        let mut stdin = self.stdin.lock().map_err(|e| anyhow!("stdin poisoned: {e}"))?;
        stdin.write_all(line.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }

    /// 終了要求 → 生存確認 → 残れば kill。
    pub fn shutdown(&self) {
        let _ = self.send(&GuiMessage::Shutdown);
        std::thread::sleep(Duration::from_millis(300));
        if let Ok(mut child) = self.child.lock() {
            match child.try_wait() {
                Ok(Some(_)) => {}
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
    }
}

impl Drop for BackendHandle {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
