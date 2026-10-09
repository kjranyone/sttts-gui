//! HuggingFace キャッシュ上のモデルの場所。`HF_HUB_CACHE` / `HF_HOME` を尊重し、スナップショットが複数あるときは
//! 必要なファイルを持つ最新のものを選ぶ。

use std::path::PathBuf;
use std::time::SystemTime;

/// キャッシュのルート(`.../huggingface/hub`)
pub fn hub_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(d).join("hub"));
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(PathBuf::from(home).join(".cache").join("huggingface").join("hub"))
}

/// `repo_dir`(例: `models--Aratako--Semantic-DACVAE-Japanese-32dim`)のスナップショットのうち、
/// `required`(相対パス)が実在するもの。複数あれば更新時刻が新しいほうを返す。
pub fn find_snapshot(repo_dir: &str, required: &str) -> Option<PathBuf> {
    let base = hub_dir()?.join(repo_dir).join("snapshots");
    let mut found: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(base)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join(required).exists())
        .map(|p| (p.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH), p))
        .collect();
    found.sort_by_key(|f| std::cmp::Reverse(f.0));
    found.into_iter().next().map(|(_, p)| p)
}
