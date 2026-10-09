//! HuggingFace キャッシュ上のモデルの場所。`huggingface_hub` と同じ環境変数(`HF_HUB_CACHE` / `HF_HOME` / `XDG_CACHE_HOME`)を尊重し、スナップショットが複数あるときは
//! 必要なファイルを持つ最新のものを選ぶ。

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, anyhow, bail};

/// キャッシュのルート(`.../huggingface/hub`)
pub fn hub_dir() -> Option<PathBuf> {
    for k in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Some(d) = std::env::var_os(k) {
            return Some(PathBuf::from(d));
        }
    }
    if let Some(d) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(d).join("hub"));
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(d).join("huggingface").join("hub"));
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

/// `org/name` → `models--org--name`
pub fn repo_folder(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// キャッシュに `files` が全部揃ったスナップショットがあればそれを返し、無ければ HuggingFace から `files` だけを
/// ダウンロードして `huggingface_hub` と同じ配置(`snapshots/<sha>/`、`refs/main`)で保存する。
pub fn ensure_files(repo: &str, files: &[&str], progress: &dyn Fn(&str)) -> Result<PathBuf> {
    let folder = repo_folder(repo);
    if let Some(snap) = find_snapshot(&folder, files[0]).filter(|s| files.iter().all(|f| s.join(f).exists())) {
        return Ok(snap);
    }
    let hub = hub_dir().ok_or_else(|| anyhow!("HF キャッシュの場所を決められません"))?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .into();
    progress(&format!("{repo}: リビジョンを確認中"));
    let info = agent
        .get(&format!("https://huggingface.co/api/models/{repo}/revision/main"))
        .call()
        .with_context(|| format!("HF API: {repo}"))?
        .body_mut()
        .read_to_string()
        .context("HF API の応答")?;
    let info: serde_json::Value = serde_json::from_str(&info).context("HF API の応答")?;
    let sha = info["sha"].as_str().ok_or_else(|| anyhow!("HF API に sha がありません"))?.to_string();
    let root = hub.join(&folder);
    let snap = root.join("snapshots").join(&sha);
    fs::create_dir_all(&snap).with_context(|| format!("create {}", snap.display()))?;
    for f in files {
        let dest = snap.join(f);
        if dest.exists() {
            continue;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        download(&agent, &format!("https://huggingface.co/{repo}/resolve/{sha}/{f}"), &dest, f, progress)?;
    }
    fs::create_dir_all(root.join("refs"))?;
    fs::write(root.join("refs").join("main"), &sha)?;
    Ok(snap)
}

fn download(agent: &ureq::Agent, url: &str, dest: &Path, name: &str, progress: &dyn Fn(&str)) -> Result<()> {
    let mut resp = agent.get(url).call().with_context(|| format!("download {url}"))?;
    let total: u64 = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(0);
    let part = dest.with_extension("part");
    let mut out = File::create(&part).with_context(|| format!("create {}", part.display()))?;
    let mut body = resp.body_mut().as_reader();
    let (mut buf, mut done, mut last) = (vec![0u8; 1 << 20], 0u64, Instant::now());
    loop {
        let n = body.read(&mut buf).with_context(|| format!("read {url}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if last.elapsed() > Duration::from_secs(2) {
            last = Instant::now();
            progress(&format!("{name}: {} / {} MB", done >> 20, total >> 20));
        }
    }
    out.flush()?;
    drop(out);
    if total > 0 && done != total {
        let _ = fs::remove_file(&part);
        bail!("{name}: ダウンロードが途中で切れました ({done} / {total} bytes)");
    }
    fs::rename(&part, dest)?;
    Ok(())
}
