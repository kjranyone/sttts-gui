//! 実モデル・実 GPU を使う確認。既定では実行しない(マイクには触れない)。
//!
//!   cargo test -p sttts-engine --release -- --ignored --test-threads=1 --nocapture
//!
//! モデルは HF キャッシュに無ければ自動でダウンロードされる。

use std::io::Cursor;
use std::path::PathBuf;

use serde_json::{Map, json};
use sttts_engine::tts::{IrodoriTts, TtsEngine, TtsRequest};

/// 参照音声に使える wav(声バンク、無ければ過去の合成出力)
fn first_voice() -> Option<String> {
    ["data/voices", "output"].iter().find_map(|d| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(d);
        std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("wav")))
            .map(|p| p.to_string_lossy().into_owned())
    })
}

#[test]
#[ignore = "実 GPU とモデルが要る"]
fn irodori_wrapper_synthesizes_on_gpu_with_ref_cache_and_sampling() {
    let tts = IrodoriTts::load("v4.1-small-mf", None, &|m| eprintln!("{m}")).expect("load");
    let none = Map::new();
    let req = |text: &'static str, refs: &'static [String], sampling: &Map<String, serde_json::Value>| {
        let t0 = std::time::Instant::now();
        let out = tts.synthesize(&TtsRequest { text, caption: None, ref_wavs: refs, seed: Some(1), sampling }).expect("synthesize");
        eprintln!("{text}: {} ms ({} ms audio) stages={:?}", t0.elapsed().as_millis(), out.duration_ms, out.stages);
        out
    };

    let a = req("こんにちは、テストです。", &[], &none);
    let r = hound::WavReader::new(Cursor::new(&a.wav)).unwrap();
    assert_eq!((r.spec().sample_rate, r.spec().channels), (a.sample_rate, 1));
    assert!(a.duration_ms > 800, "{}", a.duration_ms);

    // 同じ seed なら同じ音声(決定的)
    let b = req("こんにちは、テストです。", &[], &none);
    assert_eq!(a.wav, b.wav);

    // tts.sampling は反映される: duration_scale を大きくすると長くなる
    let slow = json!({"duration_scale": 1.3}).as_object().cloned().unwrap();
    let c = req("こんにちは、テストです。", &[], &slow);
    assert!(c.duration_ms > a.duration_ms, "{} !> {}", c.duration_ms, a.duration_ms);

    // 知らない項目は黙って捨てずエラー
    let bad = json!({"cfg_scale_text": 2.0}).as_object().cloned().unwrap();
    let err = tts.synthesize(&TtsRequest { text: "あ", caption: None, ref_wavs: &[], seed: Some(1), sampling: &bad }).unwrap_err();
    assert!(err.to_string().contains("cfg_scale_text"), "{err:#}");

    // 参照音声: 2 回目は符号化キャッシュに乗る
    if let Some(voice) = first_voice() {
        let refs: &'static [String] = Box::leak(vec![voice].into_boxed_slice());
        let first = req("参照音声ありで話します。", refs, &none);
        let second = req("参照音声ありで話します。", refs, &none);
        let ms = |o: &sttts_engine::tts::TtsOutput| o.stages.as_ref().and_then(|s| s.get("ref_latent_cache")).copied().unwrap_or(f64::NAN);
        eprintln!("ref latent: first {} ms, second {} ms", ms(&first), ms(&second));
        assert!(ms(&second) < ms(&first) * 0.2 + 5.0, "参照潜在がキャッシュされていない");
        assert_eq!(first.wav, second.wav);
    } else {
        eprintln!("data/voices に wav が無いため参照音声の確認は省略");
    }
}
