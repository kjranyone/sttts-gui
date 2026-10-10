//! v4 Large(T5Gemma 2 のテキストエンコーダ、話者は 4 フレームずつ)を PyTorch の参照と突き合わせる(CPU, fp32)。
//! 参照: `dump_irodori_ref.py --large`(ケース LA / LC)と `--large-int8`(QA / QC、int8 を fp32 に戻したもの)。
//! 重みが大きい(fp32 で 13GB)ので RAM に余裕のある環境で `--release --test-threads=1` で。

use std::path::Path;
use std::time::Instant;

use burn::tensor::{Device, Tensor, TensorData};
use irodori::pipeline::{MODEL_WEIGHTS, SamplingRequest, Trace, Tts, TtsPaths};
use irodori::t5gemma::T5Gemma2Encoder;
use irodori::testing::{self, LARGE_INT8_MODEL_REPO, LARGE_MODEL_REPO, assert_close, device, to_vec};
use irodori::weights::Weights;

const TEXT: &str = "こんにちは、よろしくお願いします。";
const CAPTION: &str = "落ち着いた女性の声で、ゆっくり話す。";

fn case_request(case: &str, ref_dir: &Path) -> SamplingRequest {
    let (seed, steps, caption, with_ref) = match case {
        "LA" => (30, 4, true, false),
        "LC" => (31, 4, false, true),
        "QA" => (20, 6, true, false),
        "QC" => (21, 6, false, true),
        _ => panic!("unknown case {case}"),
    };
    SamplingRequest {
        text: TEXT.into(),
        caption: caption.then(|| CAPTION.into()),
        no_ref: !with_ref,
        ref_wav: with_ref.then(|| ref_dir.join("ref.wav")),
        seed: Some(seed),
        num_steps: Some(steps),
        ..Default::default()
    }
}

fn end_to_end(r: &Weights, ref_dir: &Path, tts: &Tts, dev: &Device, cases: &[&str]) {
    for &case in cases {
        let req = case_request(case, ref_dir);
        let (shape, data) = r.f32_vec(&format!("{case}.dit.in.x_t.0")).unwrap();
        let per = shape[1] * shape[2];
        let noise = Tensor::<3>::from_data(TensorData::new(data[..per].to_vec(), vec![1, shape[1], shape[2]]), dev);
        let mut trace = Trace::default();
        let t0 = Instant::now();
        let out = tts.synthesize_traced(&req, Some(noise), Some(&mut trace)).unwrap();
        eprintln!("case {case}: {:.1}s; {:?}", t0.elapsed().as_secs_f64(), out.timings);

        let (_, want_text) = r.f32_vec(&format!("{case}.encode_conditions.out0.0")).unwrap();
        let got_text = to_vec(trace.text_state.clone().unwrap());
        assert_close(&format!("{case} text_state"), &got_text, &want_text[..got_text.len()], 1e-4);
        if let Some(sp) = trace.speaker_state.clone() {
            let (_, want) = r.f32_vec(&format!("{case}.encode_conditions.out2.0")).unwrap();
            assert_close(&format!("{case} speaker_state"), &to_vec(sp), &want, 1e-4);
        }
        let (_, want_dur) = r.f32_vec(&format!("{case}.duration.out.0")).unwrap();
        let got_dur = trace.duration_log_frames.unwrap();
        assert!((got_dur - want_dur[0]).abs() < 1e-3, "{case}: duration {got_dur} vs {}", want_dur[0]);
        let (_, want_z) = r.f32_vec(&format!("{case}.codec_decode.in.0")).unwrap();
        assert_close(&format!("{case} latent"), &to_vec(trace.latent.clone().unwrap()), &want_z, 5e-3);
        let (_, want) = r.f32_vec(&format!("{case}.final_audio.0")).unwrap();
        assert_eq!(out.audio.len(), want.len(), "{case}: final length");
        assert_close(&format!("{case} final audio"), &out.audio, &want, 1e-2);
    }
}

/// T5Gemma 2 エンコーダ単体(テキスト 256 トークン・キャプション 512 トークン。キャプションは窓付き注意の範囲が効く長さ)
#[test]
fn large_text_backbone_matches() {
    let Some((_, r)) = testing::named_refs("IRODORI_LARGE_REF_DIR", "irodori-ref-large") else {
        eprintln!("skip: Large の参照出力がありません(dump_irodori_ref.py --large)");
        return;
    };
    let Ok(paths) = TtsPaths::from_hf_cache(LARGE_MODEL_REPO, MODEL_WEIGHTS) else {
        eprintln!("skip: {LARGE_MODEL_REPO} が HF キャッシュにありません");
        return;
    };
    let dev = device();
    let w = Weights::open(&paths.model_weights).unwrap();
    assert!(T5Gemma2Encoder::is_t5gemma2(&w));
    let enc = T5Gemma2Encoder::load(&w, &dev).unwrap();
    for n in 0..2 {
        let (shape, ids) = r.i64_vec(&format!("LA.backbone.in.ids.{n}")).unwrap();
        let (_, mask) = r.i64_vec(&format!("LA.backbone.in.mask.{n}")).unwrap();
        let s = shape[1];
        let ids: Vec<Vec<i64>> = ids.chunks(s).map(<[i64]>::to_vec).collect();
        let mask: Vec<Vec<bool>> = mask.chunks(s).map(|m| m.iter().map(|&v| v != 0).collect()).collect();
        let t0 = Instant::now();
        let out = enc.forward(&ids, &mask);
        let el = t0.elapsed();
        let (_, want) = r.f32_vec(&format!("LA.backbone.out.{n}")).unwrap();
        assert_close(&format!("LA backbone.out.{n} (len {s}, {el:.2?})"), &to_vec(out), &want, 1e-4);
    }
}

#[test]
fn large_end_to_end_matches_pytorch() {
    let Some((dir, r)) = testing::named_refs("IRODORI_LARGE_REF_DIR", "irodori-ref-large") else {
        eprintln!("skip: Large の参照出力がありません(dump_irodori_ref.py --large)");
        return;
    };
    let Ok(paths) = TtsPaths::from_hf_cache(LARGE_MODEL_REPO, MODEL_WEIGHTS) else {
        eprintln!("skip: {LARGE_MODEL_REPO} が HF キャッシュにありません");
        return;
    };
    let dev = device();
    let tts = Tts::load(&paths, &dev).unwrap();
    end_to_end(&r, &dir, &tts, &dev, &["LA", "LC"]);
}

#[test]
fn large_int8_end_to_end_matches_pytorch() {
    let Some((dir, r)) = testing::named_refs("IRODORI_LARGE_INT8_REF_DIR", "irodori-ref-large-int8") else {
        eprintln!("skip: Large int8 の参照出力がありません(dump_irodori_ref.py --large-int8)");
        return;
    };
    let Ok(paths) = TtsPaths::from_hf_cache(LARGE_INT8_MODEL_REPO, testing::INT8_MODEL_WEIGHTS) else {
        eprintln!("skip: {LARGE_INT8_MODEL_REPO} が HF キャッシュにありません");
        return;
    };
    let dev = device();
    let w = Weights::open(&paths.model_weights).unwrap();
    assert!(w.is_int8("pretrained_text_backbone.backbone.layers.0.self_attn.q_proj.weight"));
    drop(w);
    let tts = Tts::load(&paths, &dev).unwrap();
    end_to_end(&r, &dir, &tts, &dev, &["QA", "QC"]);
}
