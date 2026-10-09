use irodori::config::ModelConfig;
use irodori::duration::*;
use irodori::testing::{self, cpu};
use irodori::weights::Weights;
use irodori::Tensor;

fn meta() -> Option<serde_json::Value> {
    let s = std::fs::read_to_string(testing::ref_dir().join("meta.json")).ok()?;
    serde_json::from_str(&s).ok()
}

fn f32t<const D: usize>(r: &Weights, name: &str) -> Tensor<D> {
    r.tensor::<D>(name, &cpu()).unwrap()
}

fn mask_f<const D: usize>(r: &Weights, name: &str) -> Tensor<D> {
    let (shape, v) = r.i64_vec(name).unwrap();
    let v: Vec<f32> = v.into_iter().map(|x| x as f32).collect();
    Tensor::<D>::from_data(burn::tensor::TensorData::new(v, shape), &cpu())
}

#[test]
fn features_and_duration_match_reference() {
    let (Some(refs), Some(meta)) = (testing::refs(), meta()) else {
        eprintln!("reference outputs missing; skipped");
        return;
    };
    let Some(ckpt) = testing::checkpoint_dir() else {
        eprintln!("checkpoint missing; skipped");
        return;
    };
    let w = Weights::open(ckpt.join("model.safetensors")).unwrap();
    let cfg = ModelConfig::from_weights(&w).unwrap();
    let model = DurationPredictor::load(&w, &cfg, &cpu()).unwrap();

    let expected_frames = [("A", 72usize), ("B", 106), ("C", 100), ("D", 196)];
    for (case, want_frames) in expected_frames {
        let k = |s: &str| format!("{case}.{s}");
        // 特徴量
        let text = meta["cases"][case]["tokenizer_calls"][0]["texts"][0].as_str().unwrap();
        let (_, m) = refs.i64_vec(&k("tok_text.out1.0")).unwrap();
        let tokens = m.iter().filter(|&&x| x != 0).count();
        let (_, hs) = refs.i64_vec(&k("duration.in.has_speaker.0")).unwrap();
        let feats = build_duration_features(text, tokens, cfg.max_text_len, hs[0] != 0).unwrap();
        let (_, want) = refs.f32_vec(&k("duration.in.duration_features.0")).unwrap();
        let (d, _) = testing::max_abs_diff(&feats, &want);
        eprintln!("case {case}: features max|diff|={d:.3e}");
        assert!(d <= 1e-6, "case {case}: features differ: {feats:?} vs {want:?}");

        // モデル
        let inp = DurationInputs {
            text_state: f32t(&refs, &k("duration.in.text_state.0")),
            text_mask: mask_f(&refs, &k("duration.in.text_mask.0")),
            speaker_state: f32t(&refs, &k("duration.in.speaker_state.0")),
            has_speaker: mask_f(&refs, &k("duration.in.has_speaker.0")),
            caption_state: f32t(&refs, &k("duration.in.caption_state.0")),
            caption_mask: mask_f(&refs, &k("duration.in.caption_mask.0")),
            has_caption: mask_f(&refs, &k("duration.in.has_caption.0")),
        };
        let out = testing::to_vec(model.predict_log_frames(&inp));
        let (_, want) = refs.f32_vec(&k("duration.out.0")).unwrap();
        let (d, _) = testing::max_abs_diff(&out, &want);
        eprintln!("case {case}: duration out={out:?} want={want:?} max|diff|={d:.3e}");
        assert!(d <= 1e-4, "case {case}: duration differs by {d:e}");

        let frames = frames_from_log(out[0], 1.0, 0.5, 30.0, 48000, 1920);
        assert_eq!(frames, want_frames, "case {case}");
    }
}

#[test]
fn frames_from_log_clamps_and_rounds_half_even() {
    assert_eq!(frames_from_log((26.0f32).ln(), 1.0, 0.5, 30.0, 48000, 1920), 25);
    assert_eq!(frames_from_log(0.0, 1.0, 0.5, 30.0, 48000, 1920), 13); // min 0.5s = 12.5 -> 13
    assert_eq!(frames_from_log(20.0, 1.0, 0.5, 30.0, 48000, 1920), 750);
}

#[test]
fn emoji_count() {
    assert_eq!(count_annotation_emojis("a😮\u{200d}💨b😊"), 2);
}
