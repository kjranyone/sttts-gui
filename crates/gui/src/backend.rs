//! バックエンド(`sttts-engine`)の起動。GUI と同じプロセスで動き、メッセージはチャネルで往復する。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_channel::Sender;
use sttts_engine::{Backend, BackendOptions, RealPlatform, Sink};
use sttts_protocol::{AnyMessage, GuiMessage};

/// アプリのルート(`data/` `output/` の置き場)。開発時はリポジトリルート(crates/gui の2つ上)。
/// 配布時は STTTS_ROOT で上書き。
pub fn repo_root() -> PathBuf {
    if let Ok(root) = std::env::var("STTTS_ROOT") {
        return PathBuf::from(root);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root not found")
}

pub struct BackendHandle {
    backend: Backend,
}

impl BackendHandle {
    /// エンジンを起動する(スレッドを立てるだけで即座に戻る)。メッセージは `tx_events` へ流れる。
    pub fn start(mock: bool, root: &Path, output_dir: &Path, tx_events: Sender<AnyMessage>) -> Self {
        let opts = BackendOptions { mock, output_dir: output_dir.to_path_buf(), root: root.to_path_buf(), ..Default::default() };
        let sink = Sink::new(move |msg| {
            let _ = tx_events.send_blocking(AnyMessage::Known(msg));
        });
        Self { backend: Backend::start(opts, Arc::new(RealPlatform), sink) }
    }

    /// RAM / GPU メモリの監視対象(エンジンは同じプロセスで動く)
    pub fn pid(&self) -> u32 {
        std::process::id()
    }

    pub fn send(&self, msg: &GuiMessage) {
        self.backend.send(msg.clone());
    }

    /// 終了要求 → マイクと TTS を止めてから戻る。
    pub fn shutdown(&self) {
        self.backend.shutdown();
    }
}
