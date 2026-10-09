//! バックエンド(`sttts-engine`)の起動。GUI と同じプロセスで動き、メッセージはチャネルで往復する。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_channel::Sender;
use sttts_engine::{Backend, BackendOptions, RealPlatform, Sink};
use sttts_protocol::{AnyMessage, GuiMessage};

/// アプリのルート(`data/` `output/` の置き場)。1 回だけ決めてプロセス中は変えない。
///
/// 1. 環境変数 `STTTS_ROOT`
/// 2. 開発中(exe がリポジトリの `target/` 配下): リポジトリルート
/// 3. exe の隣に `data/` を作れる(ポータブル配置): exe のあるフォルダ
/// 4. それ以外(Program Files 等、書き込めない場所): `%LOCALAPPDATA%\sttts-gui`
pub fn repo_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        let exe_dir = std::env::current_exe().ok().and_then(|e| e.canonicalize().ok()).and_then(|e| e.parent().map(Path::to_path_buf));
        resolve_root(
            std::env::var_os("STTTS_ROOT").filter(|v| !v.is_empty()).map(PathBuf::from),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().ok(),
            exe_dir,
            std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()).map(PathBuf::from),
            writable_data_dir,
        )
    })
    .clone()
}

fn resolve_root(
    env_root: Option<PathBuf>,
    repo: Option<PathBuf>,
    exe_dir: Option<PathBuf>,
    local_app_data: Option<PathBuf>,
    writable: impl Fn(&Path) -> bool,
) -> PathBuf {
    if let Some(root) = env_root {
        return root;
    }
    if let (Some(repo), Some(exe)) = (&repo, &exe_dir)
        && exe.starts_with(repo.join("target"))
    {
        return repo.clone();
    }
    if let Some(exe) = exe_dir.filter(|d| writable(d)) {
        return exe;
    }
    if let Some(base) = local_app_data {
        return base.join("sttts-gui");
    }
    std::env::temp_dir().join("sttts-gui")
}

/// `dir/data` を作って書き込めるか(Program Files のような保護された場所では false)
fn writable_data_dir(dir: &Path) -> bool {
    let data = dir.join("data");
    if std::fs::create_dir_all(&data).is_err() {
        return false;
    }
    let probe = data.join(".write-test");
    let ok = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
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

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn env_root_wins() {
        let r = resolve_root(Some(p("D:/x")), Some(p("C:/repo")), Some(p("C:/repo/target/release")), Some(p("C:/l")), |_| true);
        assert_eq!(r, p("D:/x"));
    }

    #[test]
    fn dev_build_uses_repo_root() {
        let r = resolve_root(None, Some(p("C:/repo")), Some(p("C:/repo/target/release")), Some(p("C:/l")), |_| true);
        assert_eq!(r, p("C:/repo"));
    }

    #[test]
    fn distributed_exe_uses_its_folder_when_writable() {
        // ビルドした PC のリポジトリが無い/別の場所にある配布先
        let r = resolve_root(None, None, Some(p("E:/apps/sttts")), Some(p("C:/l")), |_| true);
        assert_eq!(r, p("E:/apps/sttts"));
        let r = resolve_root(None, Some(p("C:/repo")), Some(p("E:/apps/sttts")), Some(p("C:/l")), |_| true);
        assert_eq!(r, p("E:/apps/sttts"));
    }

    #[test]
    fn protected_folder_falls_back_to_local_app_data() {
        let r = resolve_root(None, None, Some(p("C:/Program Files/sttts")), Some(p("C:/Users/u/AppData/Local")), |_| false);
        assert_eq!(r, p("C:/Users/u/AppData/Local/sttts-gui"));
    }
}
