//! PyTorch(transformers, CPU, fp32)の参照との段階ごとの一致と、CPU(flex)での速度。
//!
//! 参照の生成: `cd backend && uv run --no-sync python scripts/dump_whisper_ref.py`
//! (→ `target/whisper-ref/{refs.safetensors,meta.json}`。場所は `WHISPER_REF_DIR` で変えられる)。
//! 参照かモデルが無い環境では `eprintln!` して戻る(CI を落とさない)。
//! encoder は 30 秒窓を毎回計算するので CPU では重い(1 窓あたり数十秒)。
//! `WHISPER_CASES=s1,s10` のようにケースを絞れる。GPU は `--features gpu` と `WHISPER_DEVICE=gpu`
//! (GPU の検証はメインが行う)。

use std::path::PathBuf;
use std::time::Instant;

use irodori::testing::{assert_close, to_vec};
use irodori::weights::Weights;
use sttts_whisper::mel::N_SAMPLES;
use sttts_whisper::{DEFAULT_REPO, MODEL_FILES, Whisper, WhisperOptions, decode};

fn ref_dir() -> PathBuf {
    std::env::var_os("WHISPER_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/whisper-ref"))
}

fn device() -> irodori::Device {
    #[cfg(feature = "gpu")]
    if std::env::var("WHISPER_DEVICE").as_deref() == Ok("gpu") {
        return irodori::gpu_device();
    }
    irodori::device::cpu_device()
}

fn cases() -> Vec<String> {
    match std::env::var("WHISPER_CASES") {
        Ok(s) => s.split(',').map(str::to_string).collect(),
        Err(_) => ["s1", "s10", "s25", "s35"].map(str::to_string).to_vec(),
    }
}

struct Env {
    refs: Weights,
    meta: serde_json::Value,
    whisper: Whisper,
}

fn setup() -> Option<Env> {
    let dir = ref_dir();
    let (Ok(refs), Ok(meta)) = (
        Weights::open(dir.join("refs.safetensors")),
        std::fs::read_to_string(dir.join("meta.json")).map(|s| serde_json::from_str::<serde_json::Value>(&s)),
    ) else {
        eprintln!("skip: 参照が {} にありません(tools/reference/dump_whisper_ref.py)", dir.display());
        return None;
    };
    let meta = meta.ok()?;
    let Some(snap) = sttts_hub::find_snapshot(DEFAULT_REPO, &MODEL_FILES) else {
        eprintln!("skip: {DEFAULT_REPO} が HF キャッシュにありません");
        return None;
    };
    let t = Instant::now();
    let whisper = Whisper::load_from(
        WhisperOptions { repo: DEFAULT_REPO.into(), language: "ja".into(), final_beam_size: 2, device: device() },
        &snap,
        &|m| eprintln!("  load: {m}"),
    )
    .expect("load");
    eprintln!("load: {:.1}s", t.elapsed().as_secs_f32());
    Some(Env { refs, meta, whisper })
}

fn f32s(w: &Weights, key: &str) -> Vec<f32> {
    w.f32_vec(key).unwrap_or_else(|e| panic!("{key}: {e}")).1
}

#[test]
fn mel_matches_transformers() {
    let Some(refs) = Weights::open(ref_dir().join("refs.safetensors")).ok() else {
        eprintln!("skip: 参照がありません");
        return;
    };
    let ex = sttts_whisper::mel::MelExtractor::new();
    for c in cases() {
        let audio = f32s(&refs, &format!("{c}.audio"));
        let got = ex.log_mel(&audio[..audio.len().min(N_SAMPLES)]);
        assert_close(&format!("{c} mel"), &got, &f32s(&refs, &format!("{c}.mel")), 1e-4);
    }
}

#[test]
fn encoder_decoder_and_transcripts() {
    let Some(env) = setup() else { return };
    let w = &env.whisper;
    let spec = w.spec();
    let eos = spec.eos;
    let strip = |mut v: Vec<i64>| {
        if v.last() == Some(&eos) {
            v.pop();
        }
        v
    };
    for c in cases() {
        let audio = f32s(&env.refs, &format!("{c}.audio"));
        let case = &env.meta["cases"][&c];
        let secs = audio.len() as f32 / 16000.0;
        let wins: Vec<&[f32]> = audio.chunks(N_SAMPLES).collect();

        // --- 先頭窓: encoder 出力 / decoder の最初の数ステップの logits(教師強制) ---
        let mel = w.mel().log_mel(wins[0]);
        let t = Instant::now();
        let enc = w.model().encode(&mel);
        let enc_v = to_vec(enc.clone());
        eprintln!("{c}: encoder {:.2}s", t.elapsed().as_secs_f32());
        // 32 層の f32 の足し込み順の違いが積み上がるので 1e-3 まで許容(ModernBERT と同様)
        assert_close(&format!("{c} encoder"), &enc_v, &f32s(&env.refs, &format!("{c}.enc")), 1e-3);

        let cross = w.model().cross_kv(&enc);
        let want = f32s(&env.refs, &format!("{c}.logits"));
        let vocab = sttts_whisper::model::VOCAB;
        let rows = want.len() / vocab;
        let ref_ids: Vec<i64> = case["greedy_ids0"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let mut cache = w.model().new_cache();
        let mut got = w.model().decode_step(&spec.prompt, &mut cache, &cross).unwrap();
        for step in 0..rows {
            let r = &want[step * vocab..(step + 1) * vocab];
            // 抑制トークン(HF の logits は抑制前)も含めて全語彙を比べる
            assert_close(&format!("{c} logits[{step}]"), &got, r, 1e-3);
            if step + 1 == rows {
                break;
            }
            got = w.model().decode_step(&[ref_ids[step]], &mut cache, &cross).unwrap();
        }

        // --- 転写(全窓。窓ごとに encoder は 1 回だけ) ---
        let (mut greedy, mut beam) = (String::new(), String::new());
        let (mut tg, mut tb) = (0f32, 0f32);
        for (k, win) in wins.iter().enumerate() {
            let cross = if k == 0 { w.model().cross_kv(&enc) } else { w.model().cross_kv(&w.model().encode(&w.mel().log_mel(win))) };
            let t = Instant::now();
            let g = strip(decode::generate(w.model(), spec, &cross, 1).unwrap());
            tg += t.elapsed().as_secs_f32();
            if k == 0 {
                assert_eq!(g, strip(ref_ids.clone()), "{c}: greedy のトークン列");
            }
            greedy += &w.decode_text(&g).unwrap();
            let t = Instant::now();
            beam += &w.decode_text(&decode::generate(w.model(), spec, &cross, 2).unwrap()).unwrap();
            tb += t.elapsed().as_secs_f32();
        }
        eprintln!("{c}: greedy={greedy:?} beam2={beam:?} (decode のみ {tg:.2}s / {tb:.2}s)");
        assert_eq!(greedy, case["greedy"].as_str().unwrap(), "{c}: greedy の転写");
        assert_eq!(beam, case["beam2"].as_str().unwrap(), "{c}: beam(2) の転写");

        // --- 公開 API(mel → encoder → decode を通しで)と RTF ---
        if c == "s10" || c == "s1" {
            let t = Instant::now();
            let text = w.transcribe(&audio, 2).unwrap();
            let el = t.elapsed().as_secs_f32();
            assert_eq!(text, case["beam2"].as_str().unwrap());
            eprintln!("{c}: transcribe(beam 2) {el:.2}s / {secs:.1}s 音声 → RTF {:.2}", el / secs);
        }
    }
}
