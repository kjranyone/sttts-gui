//! 小さな共通部品。

use std::sync::{Mutex, MutexGuard};
use std::sync::OnceLock;
use std::time::Instant;

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
