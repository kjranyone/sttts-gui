//! RF(v4.1 Small、CFG あり)を PyTorch の参照出力(`dump_irodori_ref.py --rf`)と突き合わせる(CPU, fp32)。
//! ケースは RA〜RE(text CFG のみ / caption / speaker / alternating・sway・話者 K/V 強調・rescale・truncation / joint)。
//! CPU(flex)は遅いので `--release` で。

use std::time::Instant;

use burn::tensor::{Device, Tensor, TensorData};
use irodori::dit::Conditions;
use irodori::pipeline::{SamplingRequest, Trace, Tts, TtsPaths};
use irodori::sampler::{CfgGuidanceMode, TSchedule, rf_schedule};
use irodori::testing::{self, RF_MODEL_REPO, assert_close, device, to_vec};
use irodori::weights::Weights;

const TEXT: &str = "こんにちは、よろしくお願いします。";
const CAPTION: &str = "落ち着いた女性の声で、ゆっくり話す。";

fn case_request(case: &str) -> SamplingRequest {
    let mut r = SamplingRequest { text: TEXT.into(), no_ref: true, ..Default::default() };
    let with_ref = |r: &mut SamplingRequest| {
        r.no_ref = false;
        r.ref_wav = Some(testing::rf_ref_dir().join("ref.wav"));
    };
    match case {
        "RA" => {
            r.seed = Some(10);
            r.num_steps = Some(8);
        }
        "RB" => {
            r.seed = Some(11);
            r.num_steps = Some(8);
            r.caption = Some(CAPTION.into());
        }
        "RC" => {
            r.seed = Some(12);
            r.num_steps = Some(8);
            with_ref(&mut r);
        }
        "RD" => {
            r.seed = Some(13);
            r.num_steps = Some(6);
            r.caption = Some(CAPTION.into());
            with_ref(&mut r);
            r.cfg_guidance_mode = CfgGuidanceMode::Alternating;
            r.t_schedule_sway = true;
            r.sway_coeff = -0.5;
            r.speaker_kv_scale = Some(1.3);
            r.speaker_kv_min_t = Some(0.7);
            r.speaker_kv_max_layers = Some(4);
            r.rescale_k = Some(1.1);
            r.rescale_sigma = Some(0.5);
            r.truncation_factor = Some(0.9);
        }
        "RE" => {
            r.seed = Some(14);
            r.num_steps = Some(6);
            r.caption = Some(CAPTION.into());
            r.cfg_guidance_mode = CfgGuidanceMode::Joint;
            r.cfg_scale = Some(2.0);
            r.cfg_min_t = 0.3;
        }
        _ => panic!("unknown case {case}"),
    }
    r
}

fn setup() -> Option<(Weights, TtsPaths, Device)> {
    let Some(r) = testing::rf_refs() else {
        eprintln!("skip: RF の参照出力がありません(tools/reference/dump_irodori_ref.py --rf)");
        return None;
    };
    let Ok(paths) = TtsPaths::from_hf_cache(RF_MODEL_REPO) else {
        eprintln!("skip: {RF_MODEL_REPO} が HF キャッシュにありません");
        return None;
    };
    Some((r, paths, device()))
}

fn t3(r: &Weights, key: &str, dev: &Device) -> Tensor<3> {
    r.tensor::<3>(key, dev).unwrap()
}

fn mask(r: &Weights, key: &str, dev: &Device) -> Tensor<2> {
    let (shape, v) = r.i64_vec(key).unwrap();
    let data: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<2>::from_data(TensorData::new(data, shape), dev)
}

/// 原典の時刻列(linspace と sway)と一致する。参照の DiT 呼び出しの t を順に並べ、重複(alternating は 1 ステップ 2 回呼ぶ)を除いて比べる
#[test]
fn rf_schedule_matches_reference() {
    let Some((r, _, _)) = setup() else { return };
    for (case, steps, schedule) in [("RA", 8, TSchedule::Linear), ("RD", 6, TSchedule::Sway(-0.5))] {
        let mut seen: Vec<f32> = Vec::new();
        for n in 0.. {
            let Ok((_, t)) = r.f32_vec(&format!("{case}.dit.in.t.{n}")) else { break };
            if seen.last() != Some(&t[0]) {
                seen.push(t[0]);
            }
        }
        let want = rf_schedule(steps, schedule).unwrap();
        assert_eq!(seen.len(), steps, "{case}: {seen:?}");
        for (got, want) in seen.iter().zip(&want) {
            assert!((got - want).abs() < 1e-6, "{case}: schedule {seen:?} vs {want:?}");
        }
    }
}

/// DiT 1 回の forward(条件あり・なしをまとめたバッチを含む)が一致する。入力は参照から注入する
#[test]
fn rf_dit_forward_matches() {
    let Some((r, paths, dev)) = setup() else { return };
    let w = Weights::open(paths.model_dir.join("model.safetensors")).unwrap();
    let cfg = irodori::config::ModelConfig::from_weights(&w).unwrap();
    assert!(!cfg.is_meanflow());
    let dit = irodori::dit::Dit::load(&w, &cfg, &dev).unwrap();
    for case in ["RA", "RB", "RC", "RE"] {
        for n in [0usize, 1, 5] {
            let k = |name: &str| format!("{case}.dit.in.{name}.{n}");
            if !r.contains(&k("x_t")) {
                continue;
            }
            let cond = Conditions {
                text_state: t3(&r, &k("text_state"), &dev),
                text_mask: mask(&r, &k("text_mask"), &dev),
                speaker_state: Some(t3(&r, &k("speaker_state"), &dev)),
                speaker_mask: Some(mask(&r, &k("speaker_mask"), &dev)),
                caption_state: r.contains(&k("caption_state")).then(|| t3(&r, &k("caption_state"), &dev)),
                caption_mask: r.contains(&k("caption_mask")).then(|| mask(&r, &k("caption_mask"), &dev)),
            };
            let x_t = t3(&r, &k("x_t"), &dev);
            let t = r.tensor::<1>(&k("t"), &dev).unwrap();
            let t0 = Instant::now();
            let out = dit.forward_with_encoded_conditions(x_t, t, None, &cond, None).unwrap();
            let el = t0.elapsed();
            let want = to_vec(t3(&r, &format!("{case}.dit.out.{n}"), &dev));
            assert_close(&format!("{case} dit.out.{n} batch={} ({el:.2?})", cond.batch()), &to_vec(out), &want, 1e-4);
        }
    }
}

/// テキスト → 音声(初期ノイズを注入)。全ステップの CFG を通した最終潜在と最終音声が一致する
#[test]
fn rf_end_to_end_matches_pytorch() {
    let Some((r, paths, dev)) = setup() else { return };
    let tts = Tts::load(&paths, &dev).unwrap();
    assert!(!tts.is_meanflow());
    let cases = std::env::var("IRODORI_RF_CASES").unwrap_or_else(|_| "RA,RB,RC,RD,RE".into());
    for case in cases.split(',') {
        let req = case_request(case);
        // dit.in.x_t.0 は CFG でバッチを重ねたもの。先頭が初期ノイズ(truncation 後)
        let (shape, data) = r.f32_vec(&format!("{case}.dit.in.x_t.0")).unwrap();
        let per = shape[1] * shape[2];
        let f = req.truncation_factor.unwrap_or(1.0) as f32;
        let noise: Vec<f32> = data[..per].iter().map(|v| v / f).collect();
        let noise = Tensor::<3>::from_data(TensorData::new(noise, vec![1, shape[1], shape[2]]), &dev);
        let mut trace = Trace::default();
        let t0 = Instant::now();
        let out = tts.synthesize_traced(&req, Some(noise), Some(&mut trace)).unwrap();
        eprintln!("case {case}: {:.1}s; {:?}", t0.elapsed().as_secs_f64(), out.timings);

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
