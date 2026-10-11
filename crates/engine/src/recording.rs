//! GUI で合成した発話の保存(動画編集などにそのまま使える形)。
//!
//! 1 発話 = 1 つの WAV。起動ごとのフォルダに、喋った順の番号とセリフの頭を名前にして置く
//! (`output/2026-10-11_153012/003_そっか今日もおつかれさま.wav`)。フォルダごと動画編集ソフトへ入れれば喋った順に並ぶ。
//! 隣に字幕(`.srt`。チャンクごとの区間)と記録(`.json`。テキスト・caption・seed・区間)を置く。

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::tts::{Segment, join_wavs};

/// ファイル名に入れるセリフの頭の最大文字数
const STEM_TEXT_CHARS: usize = 24;

pub struct Recorder {
    output_dir: PathBuf,
    /// この起動のフォルダ(最初の保存で作る)
    session_dir: Option<PathBuf>,
    count: u32,
}

#[derive(Serialize)]
struct Record<'a> {
    text: &'a str,
    caption: Option<&'a str>,
    seed: Option<i64>,
    sample_rate: u32,
    duration_ms: u64,
    segments: &'a [Segment],
}

impl Recorder {
    pub fn new(output_dir: PathBuf) -> Self {
        Self { output_dir, session_dir: None, count: 0 }
    }

    /// 1 発話のチャンク(`(テキスト, WAV)` を順に)をつないで保存し、WAV のパスを返す
    pub fn save(&mut self, chunks: Vec<(String, Vec<u8>)>, caption: Option<&str>, seed: Option<i64>) -> Result<PathBuf> {
        let joined = join_wavs(chunks)?;
        let dir = self
            .session_dir
            .get_or_insert_with(|| self.output_dir.join(chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string()))
            .clone();
        std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
        self.count += 1;
        let text: String = joined.segments.iter().map(|s| s.text.as_str()).collect();
        let wav = dir.join(format!("{}.wav", file_stem(self.count, &text)));
        std::fs::write(&wav, &joined.wav).with_context(|| wav.display().to_string())?;
        std::fs::write(wav.with_extension("srt"), srt(&joined.segments))?;
        let record = Record {
            text: &text,
            caption,
            seed,
            sample_rate: joined.sample_rate,
            duration_ms: joined.duration_ms(),
            segments: &joined.segments,
        };
        std::fs::write(wav.with_extension("json"), serde_json::to_vec_pretty(&record)?)?;
        Ok(wav)
    }
}

/// `003_セリフの頭`。ファイル名に使えない文字・記号(絵文字を含む)は落とし、空白は `_` にする
fn file_stem(n: u32, text: &str) -> String {
    let head: String = text
        .chars()
        .filter_map(|c| match c {
            c if c.is_whitespace() => Some('_'),
            c if c.is_alphanumeric() || "ー〜・、。!！？-_".contains(c) => Some(c),
            _ => None,
        })
        .take(STEM_TEXT_CHARS)
        .collect();
    let head = head.trim_matches('_');
    if head.is_empty() { format!("{n:03}") } else { format!("{n:03}_{head}") }
}

/// チャンクごとの区間を SRT にする
fn srt(segments: &[Segment]) -> String {
    let time = |ms: u64| format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000);
    segments
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}\n{} --> {}\n{}\n\n", i + 1, time(s.start_ms), time(s.end_ms), s.text.trim()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::wav_bytes;

    #[test]
    fn file_names_keep_the_order_and_the_start_of_the_line() {
        assert_eq!(file_stem(3, "そっか、今日もおつかれさま。あったかいお茶でも淹れて"), "003_そっか、今日もおつかれさま。あったかいお茶でも淹");
        assert_eq!(file_stem(12, "Hello there: a/b?"), "012_Hello_there_ab");
        assert_eq!(file_stem(1, "🫶"), "001");
    }

    #[test]
    fn srt_lists_each_chunk_with_its_time_range() {
        let segs = [
            Segment { text: "おはよう。".into(), start_ms: 0, end_ms: 1234 },
            Segment { text: "眠いね。".into(), start_ms: 1234, end_ms: 3_725_007 },
        ];
        assert_eq!(srt(&segs), "1\n00:00:00,000 --> 00:00:01,234\nおはよう。\n\n2\n00:00:01,234 --> 01:02:05,007\n眠いね。\n\n");
    }

    #[test]
    fn utterances_are_saved_in_order_in_one_folder_per_run() {
        let dir = tempfile::tempdir().unwrap();
        let mut rec = Recorder::new(dir.path().to_path_buf());
        let tone = |n: usize| wav_bytes(&vec![0.1; n], 1000).unwrap();
        let a = rec.save(vec![("おはよう。".into(), tone(500)), ("眠いね。".into(), tone(250))], Some("眠そうに"), Some(7)).unwrap();
        let b = rec.save(vec![("またね。".into(), tone(100))], None, None).unwrap();
        assert_eq!(a.parent(), b.parent(), "one folder per run");
        assert_eq!(a.file_name().unwrap(), "001_おはよう。眠いね。.wav");
        assert_eq!(b.file_name().unwrap(), "002_またね。.wav");

        let joined = hound::WavReader::open(&a).unwrap();
        assert_eq!(joined.len(), 750);
        let record: serde_json::Value = serde_json::from_slice(&std::fs::read(a.with_extension("json")).unwrap()).unwrap();
        assert_eq!(record["text"], "おはよう。眠いね。");
        assert_eq!(record["caption"], "眠そうに");
        assert_eq!(record["seed"], 7);
        assert_eq!(record["duration_ms"], 750);
        assert_eq!(record["segments"][1]["start_ms"], 500);
        assert!(std::fs::read_to_string(a.with_extension("srt")).unwrap().contains("00:00:00,500 --> 00:00:00,750\n眠いね。"));
    }
}
