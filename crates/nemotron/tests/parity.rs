//! Python 参照(`NemotronOnnxAsr`)との転写一致と速度。
//!
//! 参照データ: 環境変数 `NEMOTRON_PARITY_DIR` の下に `ref.json`
//! (`{名前: {"text": ..., "sec": ...}}`)と `wav/<名前>.wav`(16kHz mono)。
//! モデルや参照が無い環境では eprintln! して何もしない。

use std::path::PathBuf;
use std::time::Instant;

use sttts_nemotron::{Nemotron, NemotronOptions};

fn load() -> Option<Nemotron> {
    if sttts_hub::find_snapshot(sttts_nemotron::DEFAULT_REPO, &["tokens.txt", "joiner.onnx"]).is_none() {
        eprintln!("Nemotron モデルが HF キャッシュに無い: skipping");
        return None;
    }
    let t = Instant::now();
    let mut o = NemotronOptions::default();
    if let Some(n) = std::env::var("NEMO_THREADS").ok().and_then(|s| s.parse().ok()) { o.num_threads = n; }
    let m = Nemotron::load(o, &|s| eprintln!("[progress] {s}")).expect("load");
    eprintln!("load: {:.2}s ({})", t.elapsed().as_secs_f32(), m.model_id());
    Some(m)
}

fn read_wav(p: &std::path::Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap();
    match r.spec().sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect(),
    }
}

#[test]
fn edge_cases_do_not_fail() {
    let Some(m) = load() else { return };
    assert_eq!(m.transcribe(&[]).unwrap(), "");
    for n in [1usize, 100, 799, 4039, 4040, 4041, 9000, 20000] {
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin() * 0.1).collect();
        m.transcribe(&v).unwrap_or_else(|e| panic!("len {n}: {e:#}"));
    }
    // 同じ入力は何度呼んでも同じ結果(状態が漏れない)
    let v: Vec<f32> = (0..30000).map(|i| ((i as f32) * 0.03).sin() * 0.2).collect();
    assert_eq!(m.transcribe(&v).unwrap(), m.transcribe(&v).unwrap());
}

#[test]
fn matches_python_reference() {
    let Some(dir) = std::env::var_os("NEMOTRON_PARITY_DIR").map(PathBuf::from) else {
        eprintln!("NEMOTRON_PARITY_DIR 未設定: skipping");
        return;
    };
    let Ok(text) = std::fs::read_to_string(dir.join("ref.json")) else {
        eprintln!("ref.json が無い: skipping");
        return;
    };
    let Some(m) = load() else { return };
    let refs: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut bad = 0;
    let (mut audio_s, mut run_s) = (0.0f64, 0.0f64);
    for (name, r) in refs.as_object().unwrap() {
        let pcm = read_wav(&dir.join("wav").join(format!("{name}.wav")));
        let t = Instant::now();
        let got = m.transcribe(&pcm).unwrap();
        let dt = t.elapsed().as_secs_f64();
        let want = r["text"].as_str().unwrap();
        let sec = pcm.len() as f64 / 16000.0;
        audio_s += sec;
        run_s += dt;
        let ok = got == want;
        eprintln!(
            "{} {name:30} {sec:6.2}s rust {dt:6.3}s (RTF {:.3}) py {:.3}s\n    py  : {want}\n    rust: {got}",
            if ok { "OK  " } else { "DIFF" },
            dt / sec,
            r["py_time"].as_f64().unwrap_or(0.0)
        );
        if !ok {
            bad += 1;
        }
    }
    eprintln!("total RTF {:.3} ({audio_s:.1}s audio in {run_s:.2}s)", run_s / audio_s);
    assert_eq!(bad, 0, "{bad} 件の転写が Python 参照と違う");
}

/// partial デコード(伸びていくバッファの全文再デコード)の速度
#[test]
fn partial_redecode_rtf() {
    let Some(dir) = std::env::var_os("NEMOTRON_PARITY_DIR").map(PathBuf::from) else { return };
    let p = dir.join("wav").join("long_concat.wav");
    if !p.exists() {
        return;
    }
    let Some(m) = load() else { return };
    let pcm = read_wav(&p);
    let step = 12800; // 800ms
    let (mut audio_s, mut run_s) = (0.0, 0.0);
    let mut n = step;
    while n <= pcm.len() {
        let t = Instant::now();
        m.transcribe(&pcm[..n]).unwrap();
        run_s += t.elapsed().as_secs_f64();
        audio_s += n as f64 / 16000.0;
        n += step;
    }
    eprintln!("partial re-decode: RTF {:.3} (累計 {audio_s:.1}s 分を {run_s:.2}s)", run_s / audio_s);
}

/// 続きから再開した結果が、最初からやり直した結果と同一であること
#[test]
fn incremental_resume_equals_fresh() {
    let Some(dir) = std::env::var_os("NEMOTRON_PARITY_DIR").map(PathBuf::from) else { return };
    let p = dir.join("wav").join("out_selfcheck_5b7255d9.wav");
    if !p.exists() {
        return;
    }
    let Some(m) = load() else { return };
    let pcm = read_wav(&p);
    let other = vec![0.01f32; 100]; // 再開状態を捨てるための別入力
    for n in (6400..=pcm.len()).step_by(12800).chain([pcm.len()]) {
        let inc = m.transcribe(&pcm[..n]).unwrap();
        m.transcribe(&other).unwrap();
        let fresh = m.transcribe(&pcm[..n]).unwrap();
        assert_eq!(inc, fresh, "len {n}");
        // 直前と同じ入力の再実行(再開位置がちょうど末尾)
        assert_eq!(m.transcribe(&pcm[..n]).unwrap(), fresh, "repeat len {n}");
    }
}

/// HF ダウンロードの動作確認(ネットワーク使用のため NEMOTRON_TEST_NETWORK=1 のときだけ)。
/// 小さいファイルだけを一時キャッシュへ取得し、hub 互換の構成になることを見る。
#[test]
fn download_small_files_into_temp_cache() {
    if std::env::var_os("NEMOTRON_TEST_NETWORK").is_none() {
        eprintln!("NEMOTRON_TEST_NETWORK 未設定: skipping");
        return;
    }
    let tmp = std::env::temp_dir().join(format!("nemotron-hub-test-{}", std::process::id()));
    // SAFETY: このテストバイナリで環境変数を触るのはこのテストだけ(他のテストは読むのみ)
    unsafe { std::env::set_var("HF_HUB_CACHE", &tmp) };
    let want = |f: &str| f == "tokens.txt" || f == "nemotron_onnx_config.json";
    let snap = sttts_hub::snapshot_download(sttts_nemotron::DEFAULT_REPO, &want, &|s| eprintln!("{s}")).unwrap();
    assert!(snap.join("tokens.txt").metadata().unwrap().len() > 100_000);
    assert!(snap.join("nemotron_onnx_config.json").exists());
    let found = sttts_hub::find_snapshot(sttts_nemotron::DEFAULT_REPO, &["tokens.txt"]).unwrap();
    assert_eq!(found, snap);
    let resolved = sttts_hub::materialize_snapshot(&snap).unwrap();
    assert!(resolved.join("tokens.txt").exists());
    let _ = std::fs::remove_dir_all(&tmp);
}
