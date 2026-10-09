//! HuggingFace Hub の最小実装(Irodori / Whisper / Nemotron で共有)。キャッシュの場所は `huggingface_hub` と同じ規則で、
//! スナップショットが無ければ API でファイル一覧を取り、resolve URL から取得する。
//! 保存先は hub キャッシュ互換(`blobs/` `snapshots/<rev>/` `refs/main`)。

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

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

/// `org/name` → `models--org--name`
pub fn repo_folder(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// キャッシュ済みのスナップショット。`refs/main` が指すものを優先し、
/// 無ければ `required`(相対パス)を全て持つうち更新時刻が最新のもの。
pub fn find_snapshot(repo: &str, required: &[&str]) -> Option<PathBuf> {
    let root = hub_dir()?.join(repo_folder(repo));
    let ok = |p: &Path| required.iter().all(|r| p.join(r).exists());
    if let Ok(rev) = fs::read_to_string(root.join("refs").join("main")) {
        let p = root.join("snapshots").join(rev.trim());
        if ok(&p) {
            return Some(p);
        }
    }
    let mut found: Vec<_> = fs::read_dir(root.join("snapshots"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| ok(p))
        .map(|p| (p.metadata().and_then(|m| m.modified()).ok(), p))
        .collect();
    found.sort_by_key(|f| std::cmp::Reverse(f.0));
    found.into_iter().next().map(|(_, p)| p)
}

/// `snapshot_download` 相当(必要なファイルだけ)。`want` が真のファイルのみ取得し、
/// スナップショットのディレクトリを返す。取得済みのファイルは再取得しない。
pub fn snapshot_download(repo: &str, want: &dyn Fn(&str) -> bool, progress: &dyn Fn(&str)) -> Result<PathBuf> {
    let root = hub_dir().context("HF キャッシュの場所を決められません")?.join(repo_folder(repo));
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .tls_config(ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build())
        .build()
        .into();
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());

    let info = agent
        .get(&format!("{endpoint}/api/models/{repo}"))
        .call()
        .with_context(|| format!("HF API: {repo}"))?
        .body_mut()
        .read_to_string()?;
    let info: serde_json::Value = serde_json::from_str(&info)?;
    let rev = info["sha"].as_str().context("HF API: sha が無い")?.to_string();
    let files: Vec<String> = info["siblings"]
        .as_array()
        .context("HF API: siblings が無い")?
        .iter()
        .filter_map(|s| s["rfilename"].as_str().map(String::from))
        .filter(|f| want(f))
        .collect();
    if files.is_empty() {
        bail!("{repo}: 取得対象のファイルがありません");
    }

    let snap = root.join("snapshots").join(&rev);
    let blobs = root.join("blobs");
    fs::create_dir_all(&blobs)?;
    fs::create_dir_all(&snap)?;
    for f in &files {
        let dst = snap.join(f);
        if dst.exists() {
            continue;
        }
        progress(&format!("取得中: {f}"));
        if let Some(p) = dst.parent() {
            fs::create_dir_all(p)?;
        }
        let url = format!("{endpoint}/{repo}/resolve/{rev}/{f}");
        let mut resp = agent.get(&url).call().with_context(|| format!("HF download: {f}"))?;
        let etag = ["x-linked-etag", "etag"]
            .iter()
            .find_map(|h| resp.headers().get(*h).and_then(|v| v.to_str().ok()))
            .map(|e| e.trim_start_matches("W/").trim_matches('"').to_string())
            .filter(|e| !e.is_empty() && e.chars().all(|c| c.is_ascii_alphanumeric()))
            .unwrap_or_else(|| format!("{rev}-{}", f.replace('/', "_")));
        let blob = blobs.join(&etag);
        if !blob.exists() {
            let part = blobs.join(format!("{etag}.incomplete"));
            {
                let mut out = fs::File::create(&part)?;
                let mut body = resp.body_mut().with_config().limit(u64::MAX).reader();
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = body.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    out.write_all(&buf[..n])?;
                }
                out.flush()?;
            }
            fs::rename(&part, &blob)?;
        }
        if fs::hard_link(&blob, &dst).is_err() {
            fs::copy(&blob, &dst)?;
        }
    }
    let refs = root.join("refs");
    fs::create_dir_all(&refs)?;
    fs::write(refs.join("main"), &rev)?;
    Ok(snap)
}

/// HF snapshot のシンボリックリンクを実ファイルに展開したディレクトリを返す。
///
/// onnxruntime は外部データ(`*.onnx.data`)のパスがモデルディレクトリ外
/// (symlink 先の `blobs/`)へ出ると拒否する。ハードリンク(不可ならコピー)で
/// snapshot と同階層の `_resolved/<revision>` に実体を置く。冪等。
pub fn materialize_snapshot(snapshot: &Path) -> Result<PathBuf> {
    let name = snapshot.file_name().context("snapshot 名が不正")?;
    let dst = snapshot
        .parent()
        .and_then(Path::parent)
        .context("snapshot の親が不正")?
        .join("_resolved")
        .join(name);
    walk(snapshot, snapshot, &dst)?;
    Ok(dst)
}

fn walk(base: &Path, dir: &Path, dst: &Path) -> Result<()> {
    for e in fs::read_dir(dir)? {
        let path = e?.path();
        // metadata は symlink を辿る(壊れたリンクは無視)
        let Ok(meta) = fs::metadata(&path) else { continue };
        if meta.is_dir() {
            walk(base, &path, dst)?;
            continue;
        }
        let target = dst.join(path.strip_prefix(base)?);
        if target.metadata().map(|m| m.len() == meta.len()).unwrap_or(false) {
            continue;
        }
        if let Some(p) = target.parent() {
            fs::create_dir_all(p)?;
        }
        let _ = fs::remove_file(&target);
        let real = fs::canonicalize(&path)?;
        if fs::hard_link(&real, &target).is_err() {
            fs::copy(&real, &target)?;
        }
    }
    Ok(())
}

/// キャッシュに `files` が全部揃ったスナップショットがあればそれを返し、無ければ `files` だけダウンロードする。
pub fn ensure_files(repo: &str, files: &[&str], progress: &dyn Fn(&str)) -> Result<PathBuf> {
    if let Some(snap) = find_snapshot(repo, files) {
        return Ok(snap);
    }
    snapshot_download(repo, &|f| files.contains(&f), progress)
}
