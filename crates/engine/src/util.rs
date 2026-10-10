//! 小さな共通部品。

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// 毒化(他スレッドの panic)を無視してロックする。状態は常に単純な値なので続行してよい。
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// プロセス内の単調時計(秒)。発話終了時刻などの計測に使う(Python の `time.monotonic()` 相当)。
pub fn now() -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// 秒 → ms(小数1桁)
pub fn ms(t: f64) -> f64 {
    (t * 10_000.0).round() / 10.0
}

/// `JoinHandle` を時間制限つきで待つ。間に合わなければ放置して false を返す(呼び出し側が資源を強制解放する)。
pub fn join_timeout(h: JoinHandle<()>, timeout: Duration) -> bool {
    let end = Instant::now() + timeout;
    while !h.is_finished() {
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let _ = h.join();
    true
}

/// 重い GPU ロード(モデルの転送・カーネルのコンパイル)を直列化するガード。
/// TTS のウォームアップと kotoba のプリロードが同じ GPU を同時に初期化しないようにする。
pub fn gpu_load_guard() -> MutexGuard<'static, ()> {
    static GPU_LOAD: Mutex<()> = Mutex::new(());
    lock(&GPU_LOAD)
}

/// GPU を使うプロセスを 1 つに限るロックファイル(GUI と `sttts-say` の同時初期化を防ぐ)。
/// GPU はマシン全体の資源なので、ルートではなく一時フォルダに置く。
pub fn gpu_lock_path() -> std::path::PathBuf {
    std::env::temp_dir().join("sttts-gpu.lock")
}

/// GPU のプロセス間ロックを取り、プロセスの終了まで持ち続ける(2 回目以降は何もしない)。
/// 別のプロセスが持っていればエラー(待たない。誰が持っているかを利用者に伝えて止める)。
pub fn hold_gpu_process_lock() -> anyhow::Result<()> {
    static HELD: Mutex<Option<std::fs::File>> = Mutex::new(None);
    let mut held = lock(&HELD);
    if held.is_none() {
        *held = Some(try_lock_file(&gpu_lock_path())?);
    }
    Ok(())
}

/// `path` を排他ロックしたファイルを返す。ロックはファイルを閉じる(プロセス終了を含む)と外れる。
pub fn try_lock_file(path: &std::path::Path) -> anyhow::Result<std::fs::File> {
    use anyhow::Context as _;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
        .with_context(|| format!("{}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
            "{}",
            sttts_i18n::tr!(
                "Another sttts process (sttts-gui or sttts-say) is using the GPU. Close it and try again.",
                "別の sttts プロセス(sttts-gui か sttts-say)が GPU を使用中です。終了してから再実行してください。",
                "另一个 sttts 进程(sttts-gui 或 sttts-say)正在使用 GPU。请先关闭它再重试。"
            )
        ),
        Err(std::fs::TryLockError::Error(e)) => Err(e).with_context(|| format!("{}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_lock_excludes_a_second_holder_until_released() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.lock");
        let first = try_lock_file(&path).unwrap();
        assert!(try_lock_file(&path).is_err(), "2 つ目の保持者は拒否される");
        drop(first);
        assert!(try_lock_file(&path).is_ok(), "解放後は取れる");
    }
}
