//! PC リソース(RAM / GPU 専用メモリ)の監視。
//!
//! Windows のパフォーマンスカウンタ(PDH)と DXGI の列挙だけを使う読み取り専用の実装で、
//! GPU デバイスを開いたり計算を投げたりしない(バックエンドの XPU/CUDA 初期化と干渉しない)。
//! 2 秒ごとにサンプルを `tx` へ送る。送信先が閉じたら終了する。

use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use async_channel::Sender;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SysSample {
    pub ram_used: u64,
    pub ram_total: u64,
    /// GPU 専用メモリ(アダプタ全体)。取得できない環境では 0
    pub vram_used: u64,
    pub vram_total: u64,
    /// バックエンドプロセスの GPU 専用メモリ
    pub app_vram: Option<u64>,
}

impl SysSample {
    pub fn vram_ratio(&self) -> Option<f64> {
        (self.vram_total > 0).then(|| self.vram_used as f64 / self.vram_total as f64)
    }
}

/// 監視スレッドを起動する。`backend_pid` はバックエンドの PID(0 = 未起動)。
pub fn spawn(tx: Sender<SysSample>, backend_pid: Arc<AtomicU32>) {
    #[cfg(windows)]
    {
        let _ = std::thread::Builder::new()
            .name("sysmon".into())
            .spawn(move || imp::run(tx, backend_pid));
    }
    #[cfg(not(windows))]
    {
        let _ = (tx, backend_pid);
    }
}

#[cfg(windows)]
mod imp {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    use async_channel::Sender;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    use windows::Win32::System::Performance::{
        PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA,
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
        PdhOpenQueryW,
    };
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows::core::{PCWSTR, w};

    use super::SysSample;

    /// 専用メモリが最大のアダプタ(= 外付け GPU)の (LUID 文字列の接頭辞[小文字], 専用メモリ総量)。
    /// カウンタのインスタンス名は大文字 16 進なので、比較は小文字に揃えて行う
    fn primary_adapter() -> Option<(String, u64)> {
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
            let mut best: Option<(String, u64)> = None;
            let mut i = 0;
            while let Ok(adapter) = factory.EnumAdapters1(i) {
                i += 1;
                let Ok(desc) = adapter.GetDesc1() else { continue };
                let total = desc.DedicatedVideoMemory as u64;
                if total > best.as_ref().map_or(0, |(_, t)| *t) {
                    let luid = desc.AdapterLuid;
                    best = Some((format!("luid_0x{:08x}_0x{:08x}", luid.HighPart as u32, luid.LowPart).to_ascii_lowercase(), total));
                }
            }
            best
        }
    }

    fn ram() -> (u64, u64) {
        let mut st = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
        if unsafe { GlobalMemoryStatusEx(&mut st) }.is_ok() {
            (st.ullTotalPhys.saturating_sub(st.ullAvailPhys), st.ullTotalPhys)
        } else {
            (0, 0)
        }
    }

    struct Query {
        query: PDH_HQUERY,
        adapter: PDH_HCOUNTER,
        process: PDH_HCOUNTER,
    }

    impl Query {
        fn open() -> Option<Self> {
            unsafe {
                let mut query = PDH_HQUERY::default();
                if PdhOpenQueryW(PCWSTR::null(), 0, &mut query) != 0 {
                    return None;
                }
                let mut adapter = PDH_HCOUNTER::default();
                let mut process = PDH_HCOUNTER::default();
                if PdhAddEnglishCounterW(query, w!("\\GPU Adapter Memory(*)\\Dedicated Usage"), 0, &mut adapter) != 0
                    || PdhAddEnglishCounterW(query, w!("\\GPU Process Memory(*)\\Dedicated Usage"), 0, &mut process) != 0
                {
                    PdhCloseQuery(query);
                    return None;
                }
                Some(Self { query, adapter, process })
            }
        }

        /// (インスタンス名, 値) の一覧
        fn read(counter: PDH_HCOUNTER) -> Vec<(String, i64)> {
            unsafe {
                let (mut size, mut count) = (0u32, 0u32);
                if PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut size, &mut count, None) != PDH_MORE_DATA {
                    return Vec::new();
                }
                // 8 バイト境界を保証するため u64 で確保する
                let mut buf = vec![0u64; (size as usize).div_ceil(8)];
                let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
                if PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut size, &mut count, Some(items)) != 0 {
                    return Vec::new();
                }
                std::slice::from_raw_parts(items, count as usize)
                    .iter()
                    .filter(|it| it.FmtValue.CStatus == 0)
                    .filter_map(|it| {
                        let name = it.szName.to_string().ok()?;
                        Some((name, it.FmtValue.Anonymous.largeValue))
                    })
                    .collect()
            }
        }

        fn collect(&self) -> bool {
            unsafe { PdhCollectQueryData(self.query) == 0 }
        }
    }

    impl Drop for Query {
        fn drop(&mut self) {
            unsafe {
                PdhCloseQuery(self.query);
            }
        }
    }

    pub fn run(tx: Sender<SysSample>, backend_pid: Arc<AtomicU32>) {
        let adapter = primary_adapter();
        let query = Query::open();
        // ワイルドカードのインスタンスは 1 回目の収集では値が出ないので、先に 1 回空収集する
        if let Some(q) = &query {
            q.collect();
            std::thread::sleep(Duration::from_millis(300));
        }
        loop {
            let (ram_used, ram_total) = ram();
            let mut sample = SysSample { ram_used, ram_total, ..Default::default() };
            if let (Some((prefix, total)), Some(q)) = (&adapter, &query)
                && q.collect()
            {
                sample.vram_total = *total;
                sample.vram_used = Query::read(q.adapter)
                    .iter()
                    .filter(|(n, _)| n.to_ascii_lowercase().starts_with(prefix.as_str()))
                    .map(|(_, v)| (*v).max(0) as u64)
                    .sum();
                let pid = backend_pid.load(Ordering::Relaxed);
                if pid != 0 {
                    let key = format!("pid_{pid}_{prefix}");
                    sample.app_vram = Some(
                        Query::read(q.process)
                            .iter()
                            .filter(|(n, _)| n.to_ascii_lowercase().starts_with(&key))
                            .map(|(_, v)| (*v).max(0) as u64)
                            .sum(),
                    );
                }
            }
            if tx.send_blocking(sample).is_err() {
                return;
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn samples_ram_and_does_not_panic_without_gpu() {
        let (tx, rx) = async_channel::unbounded();
        spawn(tx, Arc::new(AtomicU32::new(0)));
        let s = rx.recv_blocking().expect("sample");
        assert!(s.ram_total > 0 && s.ram_used > 0 && s.ram_used <= s.ram_total);
        if s.vram_total > 0 {
            assert!(s.vram_used > 0, "GPU 専用メモリが 0: PDH のインスタンス取得に失敗している");
        }
        println!("sample: {s:?}");
    }
}
