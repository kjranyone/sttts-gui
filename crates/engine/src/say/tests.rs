use std::cell::Cell;
use std::rc::Rc;

use serde_json::json;

use super::*;
use crate::config::{default_config, merge_config};
use crate::tts::MockTts;

fn cfg() -> Value {
    default_config()
}

fn line(text: &str) -> Line {
    Line { text: text.into(), ..Default::default() }
}

/// 一時ルートと、ロード回数を数える偽 TTS の作業場
fn studio(root: &Path, cfg: Value) -> (Studio, Rc<Cell<u32>>) {
    let loads = Rc::new(Cell::new(0));
    let counter = Rc::clone(&loads);
    let s = Studio::new(
        root.to_path_buf(),
        cfg,
        move || {
            counter.set(counter.get() + 1);
            Ok(Arc::new(MockTts::new("v4.1-small-mf", 0.0, 0.0)) as Arc<dyn TtsEngine>)
        },
        |_| {},
    );
    (s, loads)
}

fn add_voice(root: &Path, name: &str, audio: bool, file: Option<Value>) {
    let dir = voices_dir(root);
    std::fs::create_dir_all(&dir).unwrap();
    if audio {
        std::fs::write(dir.join(format!("{name}.wav")), crate::tts::beep_wav(0.5, Some(1))).unwrap();
    }
    if let Some(f) = file {
        std::fs::write(dir.join(format!("{name}.json")), f.to_string()).unwrap();
    }
}

#[test]
fn voice_caption_seed_and_sampling_layer_under_the_line() {
    let root = tempfile::tempdir().unwrap();
    add_voice(root.path(), "aoi", true, Some(json!({ "caption": "落ち着いた女性の声", "seed": 7, "sampling": { "duration_scale": 1.1 } })));
    let cfg = merge_config(&cfg(), &json!({ "tts": { "sampling": { "trim_tail": false } } }));

    let mut l = line(" こんにちは ");
    l.voice = Some("aoi".into());
    l.style = Some("囁くように".into());
    let spec = resolve(&cfg, root.path(), &l).unwrap();
    assert_eq!(spec.text, "こんにちは");
    assert_eq!(spec.caption.as_deref(), Some("落ち着いた女性の声。話し方は囁くように。"));
    assert_eq!(spec.voice_caption.as_deref(), Some("落ち着いた女性の声"));
    assert_eq!(spec.seed, Some(7));
    assert!(spec.ref_wav.as_deref().is_some_and(|p| p.ends_with("aoi.wav")));
    assert!(spec.ref_hash.is_some());
    assert_eq!(spec.sampling, *json!({ "trim_tail": false, "duration_scale": 1.1 }).as_object().unwrap());
    assert_eq!(spec.voice_sampling, *json!({ "duration_scale": 1.1 }).as_object().unwrap());

    // 行の指定が声より優先
    l.seed = Some(3);
    l.caption = Some("元気な少年".into());
    l.sampling = json!({ "duration_scale": 0.9 }).as_object().cloned().unwrap();
    let spec = resolve(&cfg, root.path(), &l).unwrap();
    assert_eq!(spec.seed, Some(3));
    assert_eq!(spec.caption.as_deref(), Some("元気な少年。話し方は囁くように。"));
    assert_eq!(spec.sampling["duration_scale"], json!(0.9));
}

#[test]
fn voice_may_be_only_a_settings_file_or_only_audio() {
    let root = tempfile::tempdir().unwrap();
    add_voice(root.path(), "narrator", false, Some(json!({ "caption": "低い男性の声", "seed": 42 })));
    add_voice(root.path(), "sakura", true, None);
    let names: Vec<String> = list_voices(root.path()).unwrap().into_iter().map(|v| v.name).collect();
    assert_eq!(names, ["narrator", "sakura"]);

    let n = load_voice(root.path(), "narrator").unwrap();
    assert_eq!((n.ref_wav, n.file.seed), (None, Some(42)));
    assert!(load_voice(root.path(), "sakura").unwrap().ref_wav.is_some());

    let err = load_voice(root.path(), "nobody").unwrap_err().to_string();
    assert!(err.contains("nobody") && err.contains("narrator, sakura"), "{err}");
    assert!(load_voice(root.path(), "../x").is_err());
}

#[test]
fn bad_sampling_is_rejected_before_any_synthesis() {
    let root = tempfile::tempdir().unwrap();
    let mut l = line("あ");
    l.sampling = json!({ "seed": 1 }).as_object().cloned().unwrap();
    assert!(resolve(&cfg(), root.path(), &l).unwrap_err().to_string().contains("seed"));
    l.sampling = json!({ "no_such_option": 1 }).as_object().cloned().unwrap();
    assert!(resolve(&cfg(), root.path(), &l).is_err());
    // 声の設定ファイルの未知の項目もエラー
    add_voice(root.path(), "v", false, Some(json!({ "captoin": "typo" })));
    assert!(load_voice(root.path(), "v").is_err());
}

#[test]
fn script_ids_default_to_line_numbers_and_must_be_unique() {
    let s = parse_script("{\"text\":\"a\"}\n\n{\"id\":\"intro\",\"text\":\"b\"}\n{\"text\":\"c\"}\n").unwrap();
    let ids: Vec<&str> = s.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["001", "intro", "004"]);
    assert!(parse_script("{\"id\":\"a\",\"text\":\"x\"}\n{\"id\":\"A\",\"text\":\"y\"}").is_err());
    assert!(parse_script("{\"id\":\"../a\",\"text\":\"x\"}").is_err());
    assert!(parse_script("{\"text\":\"x\",\"voise\":\"typo\"}").unwrap_err().to_string().contains('1'));
    assert!(parse_script("\n").is_err());
}

#[test]
fn render_is_incremental_and_skips_loading_when_nothing_changed() {
    let root = tempfile::tempdir().unwrap();
    add_voice(root.path(), "aoi", false, Some(json!({ "caption": "明るい声", "seed": 5 })));
    let script = root.path().join("ep1.jsonl");
    std::fs::write(&script, "{\"id\":\"a\",\"voice\":\"aoi\",\"text\":\"おはよう。\"}\n{\"id\":\"b\",\"text\":\"こんばんは。\"}\n").unwrap();

    let (s, loads) = studio(root.path(), cfg());
    let first = s.render(&script, None).unwrap();
    let out_dir = root.path().join("ep1");
    assert!(out_dir.join("a.wav").is_file() && out_dir.join("a.json").is_file() && out_dir.join("manifest.json").is_file());
    assert_eq!(first["lines"][0]["status"], "rendered");
    assert_eq!(first["lines"][0]["seed"], 5);
    let b_seed = first["lines"][1]["seed"].clone();
    assert_eq!(loads.get(), 1);

    // 何も変えなければ GPU(TTS のロード)に触れない。seed 未指定の行も気に入ったテイクが保たれる
    let (s, loads) = studio(root.path(), cfg());
    let again = s.render(&script, None).unwrap();
    assert_eq!(loads.get(), 0);
    assert_eq!(again["lines"][0]["status"], "unchanged");
    assert_eq!(again["lines"][1]["seed"], b_seed);

    // 直した行だけ撮り直す
    std::fs::write(&script, "{\"id\":\"a\",\"voice\":\"aoi\",\"text\":\"おはよう!\"}\n{\"id\":\"b\",\"text\":\"こんばんは。\"}\n").unwrap();
    let (s, _) = studio(root.path(), cfg());
    let edited = s.render(&script, None).unwrap();
    assert_eq!(edited["lines"][0]["status"], "rendered");
    assert_eq!(edited["lines"][1]["status"], "unchanged");

    // 声の設定を変えると、その声の行が撮り直しになる
    add_voice(root.path(), "aoi", false, Some(json!({ "caption": "明るい声", "seed": 6 })));
    let (s, _) = studio(root.path(), cfg());
    let revoiced = s.render(&script, None).unwrap();
    assert_eq!(revoiced["lines"][0]["status"], "rendered");
    assert_eq!(revoiced["lines"][0]["seed"], 6);
    assert_eq!(revoiced["lines"][1]["status"], "unchanged");
}

#[test]
fn render_reports_a_bad_line_before_synthesizing_anything() {
    let root = tempfile::tempdir().unwrap();
    let script = root.path().join("s.jsonl");
    std::fs::write(&script, "{\"id\":\"ok\",\"text\":\"a\"}\n{\"id\":\"bad\",\"voice\":\"ghost\",\"text\":\"b\"}\n").unwrap();
    let (s, loads) = studio(root.path(), cfg());
    let err = format!("{:#}", s.render(&script, None).unwrap_err());
    assert!(err.contains("bad") && err.contains("ghost"), "{err}");
    assert_eq!(loads.get(), 0);
    assert!(!root.path().join("s").join("ok.wav").exists());
}

#[test]
fn long_text_shares_one_seed_and_segments_tile_the_take() {
    let root = tempfile::tempdir().unwrap();
    let cfg = merge_config(&cfg(), &json!({ "pipeline": { "chunk_max_chars": 10 } }));
    let (s, _) = studio(root.path(), cfg);
    let wav = root.path().join("long.wav");
    let out = s.speak(&line("今日はいい天気ですね。散歩に行きましょう。帰りにパン屋へ寄りたいです。"), Some(&wav)).unwrap();
    let take = read_take(&wav).unwrap();
    assert!(take.segments.len() >= 2, "{:?}", take.segments);
    assert_eq!(take.segments[0].start_ms, 0);
    for w in take.segments.windows(2) {
        assert_eq!(w[0].end_ms, w[1].start_ms);
    }
    let reader = hound::WavReader::open(&wav).unwrap();
    let ms = u64::from(reader.duration()) * 1000 / u64::from(reader.spec().sample_rate);
    assert_eq!(ms, take.duration_ms);
    assert_eq!(out["duration_ms"], take.duration_ms);
    assert_eq!(out["seed"], take.seed);
}

#[test]
fn a_take_can_be_reproduced_with_like() {
    let root = tempfile::tempdir().unwrap();
    let (s, _) = studio(root.path(), cfg());
    let mut l = line("ありがとう");
    l.caption = Some("優しい声".into());
    l.style = Some("照れながら".into());
    let first = root.path().join("t1.wav");
    s.speak(&l, Some(&first)).unwrap();
    let take = read_take(&first).unwrap();

    let again = line_like(&take);
    assert_eq!(again.seed, Some(take.seed));
    let second = root.path().join("t2.wav");
    s.speak(&again, Some(&second)).unwrap();
    assert_eq!(std::fs::read(&first).unwrap(), std::fs::read(&second).unwrap());
    // .json を渡しても同じテイクを指す
    assert_eq!(read_take(&first.with_extension("json")).unwrap(), take);
}

#[test]
fn audition_then_save_promotes_the_take_to_a_reference_voice() {
    let root = tempfile::tempdir().unwrap();
    let (s, _) = studio(root.path(), cfg());
    let mut l = line("はじめまして");
    l.caption = Some("透明感のある少女の声".into());
    l.style = Some("緊張して".into());

    let out = s.audition(&l, 3, Some(&root.path().join("aud"))).unwrap();
    let takes = out["takes"].as_array().unwrap();
    assert_eq!(takes.len(), 3);
    let seeds: Vec<i64> = takes.iter().map(|t| t["seed"].as_i64().unwrap()).collect();
    assert!(seeds[0] != seeds[1] && seeds[1] != seeds[2] && seeds[0] != seeds[2], "{seeds:?}");
    assert!(s.audition(&Line { seed: Some(1), ..l.clone() }, 2, None).is_err());

    let chosen = PathBuf::from(takes[1]["out"].as_str().unwrap());
    let saved = s.save_voice("mio", &chosen).unwrap();
    assert_eq!(saved["seed"], seeds[1]);
    // 話し方(緊張して)は声に焼き込まず、声質の caption だけを引き継ぐ
    assert_eq!(saved["caption"], "透明感のある少女の声");
    let v = load_voice(root.path(), "mio").unwrap();
    assert_eq!(std::fs::read(v.ref_wav.unwrap()).unwrap(), std::fs::read(&chosen).unwrap());

    // 以後は声の名前だけで、参照音声 + seed + 声質で固定される
    let spec = resolve(&cfg(), root.path(), &Line { voice: Some("mio".into()), style: Some("嬉しそうに".into()), ..line("またね") }).unwrap();
    assert_eq!(spec.seed, Some(seeds[1]));
    assert!(spec.ref_wav.is_some());
    assert_eq!(spec.caption.as_deref(), Some("透明感のある少女の声。話し方は嬉しそうに。"));

    // 既存の声は上書きしない
    assert!(s.save_voice("mio", &chosen).is_err());
}
