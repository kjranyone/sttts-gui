//! モデルファイルの場所と取得。HF キャッシュ(`irodori::hub` と同じ規則)に無ければ、HuggingFace から
//! 最小限のファイルだけをダウンロードして、`huggingface_hub` と同じ配置(`snapshots/<sha>/`)で保存する。

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};

/// 推論に必要なファイル(`model.safetensors` が本体)
pub const FILES: [&str; 4] = ["config.json", "generation_config.json", "tokenizer.json", "model.safetensors"];

fn repo_dir(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// `repo`(例: `kotoba-tech/kotoba-whisper-v2.0`)の、必要なファイルが揃ったスナップショット。無ければ None。
pub fn find(repo: &str) -> Option<PathBuf> {
    let dir = repo_dir(repo);
    let snap = irodori::hub::find_snapshot(&dir, FILES[3])?;
    FILES.iter().all(|f| snap.join(f).exists()).then_some(snap)
}

/// キャッシュにあればそれを返し、無ければダウンロードする。
pub fn ensure(repo: &str, progress: &dyn Fn(&str)) -> Result<PathBuf> {
    if let Some(p) = find(repo) {
        return Ok(p);
    }
    let hub = irodori::hub::hub_dir().ok_or_else(|| anyhow!("HF キャッシュの場所を決められません"))?;
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
    let root = hub.join(repo_dir(repo));
    let snap = root.join("snapshots").join(&sha);
    fs::create_dir_all(&snap).with_context(|| format!("create {}", snap.display()))?;
    for f in FILES {
        let dest = snap.join(f);
        if dest.exists() {
            continue;
        }
        download(&agent, &format!("https://huggingface.co/{repo}/resolve/{sha}/{f}"), &dest, f, progress)?;
    }
    fs::create_dir_all(root.join("refs"))?;
    fs::write(root.join("refs").join("main"), &sha)?;
    Ok(snap)
}

fn download(agent: &ureq::Agent, url: &str, dest: &Path, name: &str, progress: &dyn Fn(&str)) -> Result<()> {
    let mut resp = agent.get(url).call().with_context(|| format!("download {url}"))?;
    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
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
