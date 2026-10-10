//! int8 weight-only の量子化版(v4.1 Small)を、PyTorch の参照(`dump_irodori_ref.py --int8`)と突き合わせる(CPU, fp32)。
//! 参照は int8 の重みを `qdata * scale` で fp32 に戻して fp32 で動かしたもの。Rust 版は重みを int8 のまま持ち、
//! 掛け算の直前に戻すので、量子化した層の読み込みと計算の両方を確かめる。CPU(flex)は遅いので `--release` で。

use std::time::Instant;

use burn::tensor::{Tensor, TensorData};
use irodori::pipeline::{SamplingRequest, Trace, Tts, TtsPaths};
use irodori::testing::{self, INT8_MODEL_REPO, INT8_MODEL_WEIGHTS, assert_close, device, to_vec};
use irodori::weights::Weights;

fn case_request(case: &str) -> SamplingRequest {
    let mut r = SamplingRequest { text: "こんにちは、よろしくお願いします。".into(), num_steps: Some(6), ..Default::default() };
    match case {
        "QA" => {
            r.seed = Some(20);
            r.no_ref = true;
            r.caption = Some("落ち着いた女性の声で、ゆっくり話す。".into());
        }
        "QC" => {
            r.seed = Some(21);
            r.ref_wav = Some(testing::int8_ref_dir().join("ref.wav"));
        }
        _ => panic!("unknown case {case}"),
    }
    r
}

#[test]
fn int8_checkpoint_keeps_linear_weights_quantized() {
    let Ok(paths) = TtsPaths::from_hf_cache(INT8_MODEL_REPO, INT8_MODEL_WEIGHTS) else {
        eprintln!("skip: {INT8_MODEL_REPO} が HF キャッシュにありません");
        return;
    };
    let w = Weights::open(&paths.model_weights).unwrap();
    assert!(w.is_quantized());
    // DiT・話者エンコーダ・ModernBERT の全結合は int8、それ以外(正規化・射影・長さ予測)は通常の重み
    assert!(w.is_int8("blocks.0.attention.wq.weight"));
    assert!(w.is_int8("speaker_encoder.blocks.0.mlp.w1.weight"));
    assert!(w.is_int8("pretrained_text_backbone.backbone.layers.0.attn.Wqkv.weight"));
    assert!(!w.is_int8("in_proj.weight") && w.contains("in_proj.weight"));
    // 元の名前で引くと qdata * scale に戻る
    let q = w.int8("blocks.0.attention.wq.weight").unwrap();
    let (shape, f) = w.f32_vec("blocks.0.attention.wq.weight").unwrap();
    assert_eq!(shape, q.shape.to_vec());
    let inp = q.shape[1];
    for i in [0, 1, inp + 3, q.values.len() - 1] {
        assert_eq!(f[i], f32::from(q.values[i]) * q.row_scales[i / inp]);
    }
}

#[test]
fn int8_end_to_end_matches_pytorch() {
    let Some(r) = testing::int8_refs() else {
        eprintln!("skip: int8 の参照出力がありません(tools/reference/dump_irodori_ref.py --int8)");
        return;
    };
    let Ok(paths) = TtsPaths::from_hf_cache(INT8_MODEL_REPO, INT8_MODEL_WEIGHTS) else {
        eprintln!("skip: {INT8_MODEL_REPO} が HF キャッシュにありません");
        return;
    };
    let dev = device();
    let tts = Tts::load(&paths, &dev).unwrap();
    for case in ["QA", "QC"] {
        let req = case_request(case);
        let (shape, data) = r.f32_vec(&format!("{case}.dit.in.x_t.0")).unwrap();
        let per = shape[1] * shape[2];
        let noise = Tensor::<3>::from_data(TensorData::new(data[..per].to_vec(), vec![1, shape[1], shape[2]]), &dev);
        let mut trace = Trace::default();
        let t0 = Instant::now();
        let out = tts.synthesize_traced(&req, Some(noise), Some(&mut trace)).unwrap();
        eprintln!("case {case}: {:.1}s; {:?}", t0.elapsed().as_secs_f64(), out.timings);

        let (_, want_dur) = r.f32_vec(&format!("{case}.duration.out.0")).unwrap();
        let got_dur = trace.duration_log_frames.unwrap();
        assert!((got_dur - want_dur[0]).abs() < 1e-3, "{case}: duration {got_dur} vs {}", want_dur[0]);
        let (_, want_text) = r.f32_vec(&format!("{case}.encode_conditions.out0.0")).unwrap();
        let got_text = to_vec(trace.text_state.clone().unwrap());
        assert_close(&format!("{case} text_state"), &got_text, &want_text[..got_text.len()], 1e-4);
        let (_, want_z) = r.f32_vec(&format!("{case}.codec_decode.in.0")).unwrap();
        assert_close(&format!("{case} latent"), &to_vec(trace.latent.clone().unwrap()), &want_z, 5e-3);
        let (_, want) = r.f32_vec(&format!("{case}.final_audio.0")).unwrap();
        assert_eq!(out.audio.len(), want.len(), "{case}: final length");
        assert_close(&format!("{case} final audio"), &out.audio, &want, 1e-2);
    }
}
