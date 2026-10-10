//! アプリのルート(`data/` `output/` の置き場)の解決。GUI と CLI(`sttts-say`)で同じ場所を使う。

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// アプリのルート。1 回だけ決めてプロセス中は変えない。
///
/// 1. 環境変数 `STTTS_ROOT`
/// 2. 開発中(exe がリポジトリの `target/` 配下): リポジトリルート
/// 3. Windows で exe の隣に `data/` を作れる(ポータブル配置): exe のあるフォルダ
/// 4. それ以外: OS のユーザーデータの場所([`user_data_dir`])
pub fn app_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        let exe_dir = std::env::current_exe().ok().and_then(|e| e.canonicalize().ok()).and_then(|e| e.parent().map(Path::to_path_buf));
        resolve_root(
            std::env::var_os("STTTS_ROOT").filter(|v| !v.is_empty()).map(PathBuf::from),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().ok(),
            // exe の隣に置くポータブル運用は Windows の慣習。macOS / Linux は exe の置き場(/usr/local/bin 等)を汚さない
            exe_dir,
            cfg!(windows),
            user_data_dir(std::env::consts::OS, |k| std::env::var_os(k).filter(|v| !v.is_empty())),
            writable_data_dir,
        )
    })
    .clone()
}

/// OS ごとのユーザーデータの場所(アプリ名まで含む)。
///
/// - Windows: `%LOCALAPPDATA%\sttts-gui`
/// - macOS: `~/Library/Application Support/sttts-gui`
/// - Linux 等: `$XDG_DATA_HOME/sttts-gui`(無ければ `~/.local/share/sttts-gui`)
pub fn user_data_dir(os: &str, env: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let base = match os {
        "windows" => PathBuf::from(env("LOCALAPPDATA")?),
        "macos" => PathBuf::from(env("HOME")?).join("Library").join("Application Support"),
        _ => match env("XDG_DATA_HOME") {
            Some(d) => PathBuf::from(d),
            None => PathBuf::from(env("HOME")?).join(".local").join("share"),
        },
    };
    Some(base.join("sttts-gui"))
}

fn resolve_root(
    env_root: Option<PathBuf>,
    repo: Option<PathBuf>,
    exe_dir: Option<PathBuf>,
    portable: bool,
    user_data: Option<PathBuf>,
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
    if let Some(exe) = exe_dir.filter(|d| portable && writable(d)) {
        return exe;
    }
    if let Some(dir) = user_data {
        return dir;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn env_root_wins() {
        let r = resolve_root(Some(p("D:/x")), Some(p("C:/repo")), Some(p("C:/repo/target/release")), true, Some(p("C:/l/sttts-gui")), |_| true);
        assert_eq!(r, p("D:/x"));
    }

    #[test]
    fn dev_build_uses_repo_root() {
        let r = resolve_root(None, Some(p("C:/repo")), Some(p("C:/repo/target/release")), true, Some(p("C:/l/sttts-gui")), |_| true);
        assert_eq!(r, p("C:/repo"));
    }

    #[test]
    fn distributed_exe_uses_its_folder_when_writable() {
        // ビルドした PC のリポジトリが無い/別の場所にある配布先
        let r = resolve_root(None, None, Some(p("E:/apps/sttts")), true, Some(p("C:/l/sttts-gui")), |_| true);
        assert_eq!(r, p("E:/apps/sttts"));
        let r = resolve_root(None, Some(p("C:/repo")), Some(p("E:/apps/sttts")), true, Some(p("C:/l/sttts-gui")), |_| true);
        assert_eq!(r, p("E:/apps/sttts"));
    }

    #[test]
    fn protected_folder_falls_back_to_user_data() {
        let r = resolve_root(None, None, Some(p("C:/Program Files/sttts")), true, Some(p("C:/Users/u/AppData/Local/sttts-gui")), |_| false);
        assert_eq!(r, p("C:/Users/u/AppData/Local/sttts-gui"));
    }

    #[test]
    fn user_data_follows_each_os_convention() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| OsString::from(v));
        assert_eq!(user_data_dir("windows", env(&[("LOCALAPPDATA", "C:/U/AppData/Local")])), Some(p("C:/U/AppData/Local/sttts-gui")));
        assert_eq!(user_data_dir("macos", env(&[("HOME", "/Users/u")])), Some(p("/Users/u/Library/Application Support/sttts-gui")));
        assert_eq!(user_data_dir("linux", env(&[("HOME", "/home/u")])), Some(p("/home/u/.local/share/sttts-gui")));
        assert_eq!(user_data_dir("linux", env(&[("HOME", "/home/u"), ("XDG_DATA_HOME", "/data")])), Some(p("/data/sttts-gui")));
        assert_eq!(user_data_dir("linux", env(&[])), None);
    }

    #[test]
    fn without_portable_folder_the_user_data_dir_is_used() {
        // macOS / Linux は exe の隣に置かない(書き込めても)
        let r = resolve_root(None, Some(p("/src/repo")), Some(p("/usr/local/bin")), false, Some(p("/home/u/.local/share/sttts-gui")), |_| true);
        assert_eq!(r, p("/home/u/.local/share/sttts-gui"));
        // 開発中のビルドはどの OS でもリポジトリを使う
        let r = resolve_root(None, Some(p("/src/repo")), Some(p("/src/repo/target/debug")), false, Some(p("/home/u/.local/share/sttts-gui")), |_| true);
        assert_eq!(r, p("/src/repo"));
    }
}
