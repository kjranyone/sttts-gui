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
