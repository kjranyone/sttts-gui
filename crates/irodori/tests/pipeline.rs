//! エンドツーエンド: テキスト → 音声。PyTorch の参照(`final_audio`)と、初期ノイズを揃えて突き合わせる。
//! 既定はケース A と C。`IRODORI_CASES=A,B,C,D` で変更。CPU(flex)は遅いので `--release` で。

use burn::tensor::{Tensor, TensorData};
use irodori::pipeline::{SamplingRequest, Trace, Tts, TtsPaths};
use irodori::testing::{self, assert_close, device, refs, to_vec};

fn case_request(case: &str) -> SamplingRequest {
    let long_text = "今日は朝から小雨が降っていましたが、午後にはすっかり晴れて、夕方の空がとてもきれいでした。";
    let text = "こんにちは、よろしくお願いします。";
    let seeds = [("A", 0u64), ("B", 1), ("C", 2), ("D", 3)];
    let seed = seeds.iter().find(|(c, _)| *c == case).unwrap().1;
    let mut r = SamplingRequest { seed: Some(seed), ..Default::default() };
    match case {
        "A" => {
            r.text = text.into();
            r.no_ref = true;
        }
        "B" => {
            r.text = text.into();
            r.caption = Some("落ち着いた女性の声で、ゆっくり話す。".into());
            r.no_ref = true;
        }
        "C" => {
            r.text = text.into();
            r.ref_wav = Some(testing::ref_dir().join("ref.wav"));
        }
        "D" => {
            r.text = long_text.into();
            r.no_ref = true;
        }
        _ => panic!("unknown case {case}"),
    }
    r
}

#[test]
fn end_to_end_matches_pytorch() {
    let Some(r) = refs() else {
        eprintln!("skip: 参照出力がありません(tools/reference/dump_irodori_ref.py)");
        return;
    };
    let Ok(paths) = TtsPaths::from_hf_cache(irodori::pipeline::MODEL_REPO) else {
        eprintln!("skip: モデルが HF キャッシュにありません");
        return;
    };
    let dev = device();
    let tts = Tts::load(&paths, &dev).unwrap();
    assert!(tts.has_watermark(), "透かしモデルが必要です");
    let cases = std::env::var("IRODORI_CASES").unwrap_or_else(|_| "A,C".into());
    for case in cases.split(',') {
        let req = case_request(case);
        let (shape, data) = r.f32_vec(&format!("{case}.dit.in.x_t.0")).unwrap();
        let noise = Tensor::<3>::from_data(TensorData::new(data, shape), &dev);
        let mut trace = Trace::default();
        let t0 = std::time::Instant::now();
        let out = tts.synthesize_traced(&req, Some(noise), Some(&mut trace)).unwrap();
        eprintln!("case {case}: {:.1}s total; timings {:?}", t0.elapsed().as_secs_f64(), out.timings);

        // 長さ(フレーム数)は一致する
        let (_, want_dur) = r.f32_vec(&format!("{case}.duration.out.0")).unwrap();
        let got_dur = trace.duration_log_frames.unwrap();
        assert!((got_dur - want_dur[0]).abs() < 1e-3, "{case}: duration {got_dur} vs {}", want_dur[0]);

        // 最終潜在(= codec_decode.in.0)
        let (_, want_z) = r.f32_vec(&format!("{case}.codec_decode.in.0")).unwrap();
        assert_close(&format!("{case} latent"), &to_vec(trace.latent.clone().unwrap()), &want_z, 2e-3);

        // 透かし前の音声(codec_decode.out.0 の先頭)と、最終音声
        let (_, dec) = r.f32_vec(&format!("{case}.codec_decode.out.0")).unwrap();
        let (_, wm_in) = r.f32_vec(&format!("{case}.watermark.in0.0")).unwrap();
        assert_eq!(trace.raw_audio.len(), wm_in.len(), "{case}: trimmed length");
        assert_close(&format!("{case} decoded"), &trace.raw_audio, &dec[..wm_in.len()], 5e-3);
        let (_, want) = r.f32_vec(&format!("{case}.final_audio.0")).unwrap();
        assert_eq!(out.audio.len(), want.len(), "{case}: final length");
        assert_close(&format!("{case} final audio"), &out.audio, &want, 5e-3);
    }
}
