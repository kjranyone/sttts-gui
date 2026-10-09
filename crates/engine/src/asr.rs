//! ASR エンジンの共通インターフェース。
//!
//! 全エンジンは 16kHz mono f32 の発話音声を受け取って文字列にする。Gemini のようなストリーミング型は
//! `is_streaming()` が true で、VAD の開始〜終了に合わせて `begin_stream` / `feed_stream` / `end_stream` が呼ばれ、
//! 確定は `finish_stream` で受け取る。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;

use crate::config::{get_i64, get_str};

/// ストリーミング ASR の途中結果コールバック `(utterance, text)`
pub type StreamCb = Arc<dyn Fn(u64, &str) + Send + Sync>;

/// ロード進捗の通知先
pub type Progress<'a> = &'a dyn Fn(&str);

pub trait AsrEngine: Send + Sync {
    fn model_id(&self) -> String;

    /// 発話全体の確定用デコード
    fn transcribe_utterance(&self, audio: &[f32]) -> Result<String>;

    /// 発話途中の部分デコード(表示用)
    fn transcribe_partial(&self, audio: &[f32]) -> Result<String>;

    /// partial に渡す音声の上限(秒、発話の末尾から)。None なら発話の先頭から全部渡す。
    /// 先頭からの全音声を渡すと partial の文字列が確定と同じく発話の先頭から伸びていき、
    /// 話している途中の逐次読み上げに使える。
    fn max_partial_seconds(&self) -> Option<f64> {
        Some(12.0)
    }

    /// 使い終わったエンジンの解放(モデル・接続)
    fn unload(&self) {}

    // ---- ストリーミング型(既定は非対応) ----
    fn is_streaming(&self) -> bool {
        false
    }
    fn begin_stream(&self, _utterance: u64, _on_partial: StreamCb) {}
    fn feed_stream(&self, _utterance: u64, _frame: &[f32]) {}
    fn end_stream(&self, _utterance: u64) {}
    fn abort_stream(&self, _utterance: u64) {}
    fn finish_stream(&self, _utterance: u64, audio: &[f32]) -> Result<String> {
        self.transcribe_utterance(audio)
    }
    fn abort_all_streams(&self) {}
}

/// 固定テキストを返す ASR(ベンチ・テスト用、`asr.engine = "mock"`)。`latency_ms` で推論時間を模倣する。
///
/// 確定ごとに `texts` を順に返す。partial は次に確定するテキストの先頭を音声長に比例して返す(1秒あたり約6文字)。
pub struct MockAsr {
    latency: Duration,
    texts: Vec<String>,
    finals: AtomicUsize,
}

pub const MOCK_TEXTS: [&str; 3] = [
    "こんにちは、今日はいい天気ですね。",
    "音声合成のレイテンシを測定しています。",
    "これは三つ目の発話です。チャンク分割を確認します。",
];

impl MockAsr {
    pub fn new(latency_ms: u64, texts: Vec<String>) -> Self {
        let texts = if texts.is_empty() { MOCK_TEXTS.iter().map(|s| (*s).to_string()).collect() } else { texts };
        Self { latency: Duration::from_millis(latency_ms), texts, finals: AtomicUsize::new(0) }
    }

    fn current(&self) -> &str {
        &self.texts[self.finals.load(Ordering::Relaxed) % self.texts.len()]
    }
}

impl AsrEngine for MockAsr {
    fn model_id(&self) -> String {
        "mock-asr".into()
    }

    fn transcribe_utterance(&self, _audio: &[f32]) -> Result<String> {
        std::thread::sleep(self.latency);
        let text = self.current().to_string();
        self.finals.fetch_add(1, Ordering::Relaxed);
        Ok(text)
    }

    fn transcribe_partial(&self, audio: &[f32]) -> Result<String> {
        std::thread::sleep(self.latency);
        let text = self.current();
        let n = ((audio.len() as f64 / 16000.0 * 6.0) as usize).clamp(1, text.chars().count());
        Ok(text.chars().take(n).collect())
    }
}

/// 設定(`asr` セクション)から実エンジンを作る。mock 以外は実モデルのロードを伴う。
pub fn create_asr(cfg: &Value, progress: Progress) -> Result<Arc<dyn AsrEngine>> {
    let engine = get_str(cfg, "asr", "engine").unwrap_or("kotoba").to_lowercase();
    match engine.as_str() {
        "mock" => {
            let delay = get_i64(cfg, "asr", "mock_load_delay_ms", 0).max(0) as u64;
            if delay > 0 {
                std::thread::sleep(Duration::from_millis(delay));
            }
            let texts: Vec<String> = cfg["asr"]["mock_texts"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            Ok(Arc::new(MockAsr::new(get_i64(cfg, "asr", "mock_latency_ms", 0).max(0) as u64, texts)))
        }
        other => crate::engines::create_real_asr(other, cfg, progress),
    }
}
