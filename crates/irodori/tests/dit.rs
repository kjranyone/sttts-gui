//! DiT / 話者エンコーダ / サンプラを PyTorch の参照出力と突き合わせる(CPU, fp32)。

use std::time::Instant;

use burn::tensor::{Device, Tensor, TensorData};
use irodori::config::ModelConfig;
use irodori::dit::{Conditions, Dit};
use irodori::sampler::{find_flattening_point, sample_euler_meanflow, unpatchify_latent};
use irodori::testing::{self, assert_close, cpu, to_vec};
use irodori::weights::Weights;

struct Ctx {
    refs: Weights,
    dit: Dit,
    dev: Device,
}

fn setup() -> Option<Ctx> {
    let Some(refs) = testing::refs() else {
        eprintln!("skip: reference dump not found (run backend/scripts/dump_irodori_ref.py)");
        return None;
    };
    let Some(dir) = testing::checkpoint_dir() else {
        eprintln!("skip: checkpoint not found");
        return None;
    };
    let dev = cpu();
    let w = Weights::open(dir.join("model.safetensors")).unwrap();
    let cfg = ModelConfig::from_weights(&w).unwrap();
    let t0 = Instant::now();
    let dit = Dit::load(&w, &cfg, &dev).unwrap();
    eprintln!("Dit::load: {:.2?}", t0.elapsed());
    Some(Ctx { refs, dit, dev })
}

fn t3(c: &Ctx, key: &str) -> Tensor<3> {
    c.refs.tensor::<3>(key, &c.dev).unwrap()
}

fn mask(c: &Ctx, key: &str) -> Tensor<2> {
    let (shape, v) = c.refs.i64_vec(key).unwrap();
    let data: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<2>::from_data(TensorData::new(data, shape), &c.dev)
}

fn vec1(c: &Ctx, key: &str) -> Tensor<1> {
    c.refs.tensor::<1>(key, &c.dev).unwrap()
}

/// `dit.in.*.n` から条件を組む
fn dit_conditions(c: &Ctx, case: &str, n: usize) -> Conditions {
    Conditions {
        text_state: t3(c, &format!("{case}.dit.in.text_state.{n}")),
        text_mask: mask(c, &format!("{case}.dit.in.text_mask.{n}")),
        speaker_state: Some(t3(c, &format!("{case}.dit.in.speaker_state.{n}"))),
        speaker_mask: Some(mask(c, &format!("{case}.dit.in.speaker_mask.{n}"))),
        caption_state: Some(t3(c, &format!("{case}.dit.in.caption_state.{n}"))),
        caption_mask: Some(mask(c, &format!("{case}.dit.in.caption_mask.{n}"))),
    }
}

/// sampler 用の条件(`encode_conditions.out*.1`)
fn sampler_conditions(c: &Ctx, case: &str) -> Conditions {
    let k = |i: usize| format!("{case}.encode_conditions.out{i}.1");
    Conditions {
        text_state: t3(c, &k(0)),
        text_mask: mask(c, &k(1)),
        speaker_state: Some(t3(c, &k(2))),
        speaker_mask: Some(mask(c, &k(3))),
        caption_state: Some(t3(c, &k(4))),
        caption_mask: Some(mask(c, &k(5))),
    }
}

#[test]
fn speaker_encoder_matches() {
    let Some(c) = setup() else { return };
    // ケース C: 参照潜在あり
    let latent = t3(&c, "C.codec_encode.out.0");
    let [b, t, _] = latent.dims();
    let ones = Tensor::<2>::ones([b, t], &c.dev);
    let t0 = Instant::now();
    let (st, mk) = c.dit.encode_speaker(Some((latent, ones)), 1).unwrap();
    eprintln!("encode_speaker (ref 75 frames): {:.2?}", t0.elapsed());
    for n in 0..2 {
        assert_close(&format!("C speaker_state.{n}"), &to_vec(st.clone()), &to_vec(t3(&c, &format!("C.encode_conditions.out2.{n}"))), 1e-4);
        assert_close(&format!("C speaker_mask.{n}"), &to_vec(mk.clone()), &to_vec(mask(&c, &format!("C.encode_conditions.out3.{n}"))), 0.0);
    }
    // ケース A: 参照なし
    let (st, mk) = c.dit.encode_speaker(None, 1).unwrap();
    assert_close("A speaker_state", &to_vec(st), &to_vec(t3(&c, "A.encode_conditions.out2.0")), 1e-4);
    assert_close("A speaker_mask", &to_vec(mk), &to_vec(mask(&c, "A.encode_conditions.out3.0")), 0.0);
}

#[test]
fn dit_forward_matches() {
    let Some(c) = setup() else { return };
    for case in ["A", "B", "C", "D"] {
        for n in 0..4 {
            let cond = dit_conditions(&c, case, n);
            let x_t = t3(&c, &format!("{case}.dit.in.x_t.{n}"));
            let t = vec1(&c, &format!("{case}.dit.in.t.{n}"));
            let dt = vec1(&c, &format!("{case}.dit.in.delta_t.{n}"));
            let t0 = Instant::now();
            let out = c.dit.forward_with_encoded_conditions(x_t, t, dt, &cond, None).unwrap();
            let el = t0.elapsed();
            let want = to_vec(t3(&c, &format!("{case}.dit.out.{n}")));
            assert_close(&format!("{case} dit.out.{n} ({el:.2?})"), &to_vec(out), &want, 1e-4);
        }
    }
}

#[test]
fn sampler_matches() {
    let Some(c) = setup() else { return };
    let dim = c.dit.cfg.patched_latent_dim();
    for case in ["A", "B", "C", "D"] {
        let cond = sampler_conditions(&c, case);
        let noise = t3(&c, &format!("{case}.dit.in.x_t.0"));
        let seq = noise.dims()[1];
        let t0 = Instant::now();
        let out = sample_euler_meanflow(&c.dit, &cond, seq, 4, Some(noise), 0).unwrap();
        eprintln!("{case}: sample 4 steps seq={seq}: {:.2?}", t0.elapsed());
        let want = t3(&c, &format!("{case}.codec_decode.in.0"));
        // codec_decode.in は unpatchify 済み(latent_patch_size = 1 なので同形)
        let patch = c.dit.cfg.latent_patch_size;
        let out = unpatchify_latent(out, patch, c.dit.cfg.latent_dim);
        assert_eq!(out.dims(), want.dims());
        assert_close(&format!("{case} final latent"), &to_vec(out), &to_vec(want), 1e-3);
        let _ = dim;
    }
}

/// 各ステップの x_t を、参照の x_t.n から 1 ステップずつ進めて照合する(誤差が累積しない形)
#[test]
fn sampler_steps_match() {
    let Some(c) = setup() else { return };
    let cond = sampler_conditions(&c, "A");
    let kv = c.dit.build_context_kv_cache(&cond).unwrap();
    let sched = irodori::sampler::meanflow_schedule(4);
    assert_eq!(sched, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
    for n in 0..3 {
        let x = t3(&c, &format!("A.dit.in.x_t.{n}"));
        let b = x.dims()[0];
        let t = Tensor::<1>::from_data(TensorData::new(vec![sched[n]; b], vec![b]), &c.dev);
        let d = Tensor::<1>::from_data(TensorData::new(vec![sched[n] - sched[n + 1]; b], vec![b]), &c.dev);
        let v = c.dit.forward_with_encoded_conditions(x.clone(), t, d, &cond, Some(&kv)).unwrap();
        let next = x.add(v.mul_scalar(sched[n + 1] - sched[n]));
        assert_close(&format!("A x_t.{} from step", n + 1), &to_vec(next), &to_vec(t3(&c, &format!("A.dit.in.x_t.{}", n + 1))), 1e-4);
    }
}

#[test]
fn flattening_point() {
    let Some(c) = setup() else { return };
    // 参照(PyTorch)での値: (case, window, std, mean) -> index
    let cases = [
        ("A", 20, 0.05, 0.1, 72),
        ("B", 20, 0.05, 0.1, 106),
        ("A", 5, 1.0, 1.0, 0),
        ("B", 5, 1.0, 1.0, 18),
        ("C", 5, 1.0, 1.0, 15),
        ("D", 5, 1.0, 1.0, 11),
    ];
    for (case, w, s, m, want) in cases {
        let z = t3(&c, &format!("{case}.codec_decode.in.0"));
        let [_, t, d] = z.dims();
        let got = find_flattening_point(&to_vec(z), t, d, w, s, m);
        assert_eq!(got, want, "case {case} window {w}");
    }
    // 合成: 前半ノイズ、後半ゼロ → ゼロ区間の手前(窓が全部ゼロになる最初の位置)
    let (t, d) = (60, 4);
    let mut v = vec![0f32; t * d];
    for (i, x) in v.iter_mut().enumerate().take(30 * d) {
        *x = if i % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + (i % 3) as f32);
    }
    assert_eq!(find_flattening_point(&v, t, d, 20, 0.05, 0.1), 30);
}

#[test]
fn schedule_and_noise() {
    let dev = cpu();
    let a = irodori::sampler::initial_noise(1, 8, 32, 7, &dev);
    let b = irodori::sampler::initial_noise(1, 8, 32, 7, &dev);
    assert_eq!(to_vec(a.clone()), to_vec(b));
    let v = to_vec(a);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    assert!(mean.abs() < 0.2);
}
