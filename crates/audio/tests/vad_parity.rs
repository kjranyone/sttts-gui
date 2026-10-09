//! Silero VAD の Python 参照(pip の silero_vad.VADIterator, onnx=True)とのパリティ。
//!
//! 参照は `tests/data/vad_ref.json`(生成スクリプトは使い捨て)。無ければ skip。

use std::path::PathBuf;

use serde_json::Value;
use sttts_audio::{FRAME, SileroVad, VadEvent, load_wav_16k};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn refs() -> Option<Value> {
    let s = std::fs::read_to_string(data("vad_ref.json")).ok()?;
    serde_json::from_str(&s).ok()
}

fn ev_one(e: &Value) -> Option<VadEvent> {
    if e.is_null() {
        return None;
    }
    match (e.get("start"), e.get("end")) {
        (Some(s), _) => Some(VadEvent::Start(s.as_i64().unwrap())),
        (_, Some(s)) => Some(VadEvent::End(s.as_i64().unwrap())),
        _ => panic!("bad event {e}"),
    }
}

fn ev_list(v: &Value) -> Vec<VadEvent> {
    v.as_array().unwrap().iter().filter_map(ev_one).collect()
}

fn run(vad: &mut SileroVad, audio: &[f32]) -> (Vec<f32>, Vec<VadEvent>) {
    let mut probs = Vec::new();
    let mut events = Vec::new();
    // 確率とイベントを両方取るため、probability を呼んで step に流す(try_process と同じ経路)
    for f in audio.as_chunks::<FRAME>().0.iter() {
        let p = vad.probability(f).unwrap();
        probs.push(p);
        if let Some(e) = vad.step(p as f64) {
            events.push(e);
        }
    }
    (probs, events)
}

#[test]
fn frame_probabilities_and_events_match_python() {
    let Some(refs) = refs() else {
        eprintln!("skip: tests/data/vad_ref.json がありません");
        return;
    };
    let mut worst = 0.0f32;
    for name in ["multi", "noise", "silence", "speech_noise"] {
        let audio = load_wav_16k(data(&format!("{name}.wav"))).unwrap();
        for (th, ms) in [(0.5f32, 280u32), (0.3, 100), (0.1, 100)] {
            let key = format!("{name}|{th}|{ms}");
            let r = &refs[&key];
            let want_probs: Vec<f32> = r["probs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let want_ev = ev_list(&r["events"]);
            let mut vad = SileroVad::new(th, ms).unwrap();
            let (probs, events) = run(&mut vad, &audio);
            assert_eq!(probs.len(), want_probs.len(), "{key}: フレーム数");
            let diff = probs
                .iter()
                .zip(&want_probs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            worst = worst.max(diff);
            assert!(diff < 1e-4, "{key}: 確率の最大差 {diff}");
            assert_eq!(events, want_ev, "{key}: イベント列");
        }
    }
    eprintln!("確率の最大絶対差: {worst:e}");
}

/// `process` 経由(公開 API)でも同じイベントが出ること。reset 後に再実行しても同一。
#[test]
fn process_api_and_reset() {
    let Some(refs) = refs() else {
        eprintln!("skip: tests/data/vad_ref.json がありません");
        return;
    };
    let audio = load_wav_16k(data("multi.wav")).unwrap();
    let want = ev_list(&refs["multi|0.5|280"]["events"]);
    assert!(!want.is_empty());
    let mut vad = SileroVad::new(0.5, 280).unwrap();
    for _ in 0..2 {
        let got: Vec<VadEvent> = audio
            .as_chunks::<FRAME>()
            .0
            .iter()
            .filter_map(|f| vad.process(f))
            .collect();
        assert_eq!(got, want);
        vad.reset();
    }
}

/// 確率列を直接流す状態機械(VADIterator の移植)の厳密一致。
#[test]
fn state_machine_matches_python() {
    let Some(refs) = refs() else {
        eprintln!("skip: tests/data/vad_ref.json がありません");
        return;
    };
    let probs: Vec<f64> = refs["step"]["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for (key, th, ms) in [
        ("0.5|280", 0.5f32, 280u32),
        ("0.5|100", 0.5, 100),
        ("0.3|500", 0.3, 500),
    ] {
        let want = refs["step"]["cases"][key].as_array().unwrap();
        let mut vad = SileroVad::new(th, ms).unwrap();
        for (i, p) in probs.iter().enumerate() {
            assert_eq!(vad.step(*p), ev_one(&want[i]), "{key}: フレーム {i}");
        }
    }
}

/// 48kHz の TTS 音声を Rust のリサンプラで 16k 化しても、soxr 版と近い位置で発話区間が取れる。
#[test]
fn rust_resampler_vs_soxr_events() {
    let Some(refs) = refs() else {
        eprintln!("skip: tests/data/vad_ref.json がありません");
        return;
    };
    let name = refs["resample48"]["file"].as_str().unwrap();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../output")
        .join(name);
    if !path.exists() {
        eprintln!("skip: {} がありません", path.display());
        return;
    }
    let clip = load_wav_16k(&path).unwrap();
    let mut audio = vec![0.0f32; 8000];
    audio.extend(&clip);
    audio.extend(vec![0.0f32; 24000]);
    let mut vad = SileroVad::new(0.5, 280).unwrap();
    let got: Vec<VadEvent> = audio
        .as_chunks::<FRAME>()
        .0
        .iter()
        .filter_map(|f| vad.process(f))
        .collect();
    let want = ev_list(&refs["resample48"]["events"]);
    eprintln!("rust: {got:?}\nsoxr: {want:?}");
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        let (a, b) = match (g, w) {
            (VadEvent::Start(a), VadEvent::Start(b)) | (VadEvent::End(a), VadEvent::End(b)) => {
                (*a, *b)
            }
            _ => panic!("種別不一致: {g:?} vs {w:?}"),
        };
        assert!((a - b).abs() <= 2 * FRAME as i64, "{g:?} vs {w:?}");
    }
}
