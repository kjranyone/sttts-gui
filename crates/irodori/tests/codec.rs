use std::time::Instant;

use irodori::codec::{self, DacVae};
use irodori::pth::Pth;
use irodori::testing::{assert_close, device, max_abs_diff, refs, to_vec};
use irodori::{Device, Tensor};

fn weights_path() -> Option<std::path::PathBuf> {
    let p = codec::default_weights_path();
    if p.is_none() {
        eprintln!("DACVAE weights.pth not found in the HF cache: skipping");
    }
    p
}

fn load_model(dev: &Device) -> Option<DacVae> {
    let p = weights_path()?;
    let t = Instant::now();
    let m = DacVae::load(p, dev).expect("load DACVAE");
    eprintln!("DACVAE load: {:.2}s", t.elapsed().as_secs_f32());
    Some(m)
}

#[test]
fn pth_reads_state_dict_and_folds_weight_norm() {
    let Some(path) = weights_path() else { return };
    let mut p = Pth::load(&path).unwrap();
    assert_eq!(p.tensors.len(), 317);
    assert_eq!(p.metadata["kwargs"]["sample_rate"], 48000);
    assert_eq!(p.get("decoder.model.0.weight_v").unwrap().shape, vec![1536, 1024, 7]);

    // 畳み込み前の g と v から手計算した値と一致すること
    let g = p.get("decoder.model.0.weight_g").unwrap().data[5] as f64;
    let v = p.get("decoder.model.0.weight_v").unwrap().clone();
    let per = 1024 * 7;
    let row = &v.data[5 * per..6 * per];
    let norm = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

    p.fold_weight_norm().unwrap();
    assert!(!p.contains("decoder.model.0.weight_g"));
    let w = p.get("decoder.model.0.weight").unwrap();
    assert_eq!(w.shape, vec![1536, 1024, 7]);
    let want = (row[100] as f64 * g / norm) as f32;
    assert!((w.data[5 * per + 100] - want).abs() < 1e-6);
    // 畳み込み後の行ノルムは g に等しい
    let wn = w.data[5 * per..6 * per].iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    assert!((wn - g).abs() < 1e-4 * g.abs().max(1.0), "{wn} vs {g}");
    // ConvTranspose(dim 0 = 入力チャネル)も畳み込まれる
    assert_eq!(p.get("decoder.model.1.block.1.weight").unwrap().shape, vec![1536, 768, 24]);
    // 透かし枝で使わない通常の重みは残る
    assert!(p.contains("decoder.model.1.block.3.weight"));
}

#[test]
fn resample_and_loudness_sanity() {
    // 24k → 48k の 1 kHz 正弦波:中央部は解析解と一致
    let n = 24000;
    let x: Vec<f32> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 24000.0).sin() as f32).collect();
    let y = codec::resample(&x, 24000, 48000);
    assert_eq!(y.len(), 48000);
    let mut worst = 0f32;
    for (i, &v) in y.iter().enumerate().skip(200).take(47600) {
        let want = (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin() as f32;
        worst = worst.max((v - want).abs());
    }
    eprintln!("resample sine: max err {worst:.3e}");
    assert!(worst < 5e-3);

    // 48 kHz のフルスケール 997 Hz 正弦波は約 -3.01 LUFS(モノ)
    let s: Vec<f32> = (0..48000 * 5)
        .map(|i| (2.0 * std::f64::consts::PI * 997.0 * i as f64 / 48000.0).sin() as f32)
        .collect();
    let l = codec::integrated_loudness(&s, 48000);
    eprintln!("loudness of full-scale 997 Hz sine: {l:.3} LUFS");
    assert!((l - -3.01).abs() < 0.05, "{l}");
    // 正規化後は目標に一致
    let mut q = s.clone();
    codec::normalize_loudness(&mut q, 48000, -16.0);
    let l2 = codec::integrated_loudness(&q, 48000);
    assert!((l2 + 16.0).abs() < 0.01, "{l2}");
}

#[test]
fn decode_matches_reference() {
    let Some(refs) = refs() else {
        eprintln!("reference outputs not found: skipping");
        return;
    };
    let dev = device();
    let Some(model) = load_model(&dev) else { return };
    assert_eq!(model.sample_rate, 48000);
    assert_eq!(model.hop, 1920);
    assert_eq!(model.latent_dim, 32);

    for case in ["A", "C", "D"] {
        let z: Tensor<3> = refs.tensor(&format!("{case}.codec_decode.in.0"), &dev).unwrap();
        let want = to_vec(refs.tensor::<3>(&format!("{case}.codec_decode.out.0"), &dev).unwrap());
        let [_, t, _] = z.dims();

        let t0 = Instant::now();
        let out = model.decode_latent(z.clone());
        let got = to_vec(out);
        let dt = t0.elapsed().as_secs_f32();
        let (d, m) = max_abs_diff(&got, &want);
        eprintln!("decode {case}: T={t} {:.2}s audio, {dt:.2}s CPU, max|diff|={d:.3e} (ref max {m:.3}, rel {:.2e})", got.len() as f32 / 48000.0, d / m);
        assert_close(&format!("decode {case}"), &got, &want, 1e-4);

        // 窓分割(重なり付き)版と全体版の差
        let configs: &[(usize, usize)] = if case == "A" { &[(25, 8), (25, 4), (50, 8)] } else { &[(25, 8)] };
        for &(window, ctx) in configs {
            let t0 = Instant::now();
            let w = to_vec(model.decode_latent_windowed(z.clone(), window, ctx));
            let dt = t0.elapsed().as_secs_f32();
            let (dw, _) = max_abs_diff(&w, &got);
            eprintln!("  windowed(window={window}, ctx={ctx}) {case}: {dt:.2}s, max|diff vs full|={dw:.3e} (rel {:.2e})", dw / m);
            if ctx >= 8 {
                assert!(dw <= 1e-4 * m, "windowed({window},{ctx}) {case}: {dw:e}");
            }
        }
    }
}

#[test]
fn encode_matches_reference() {
    let Some(refs) = refs() else {
        eprintln!("reference outputs not found: skipping");
        return;
    };
    let dev = device();
    let Some(model) = load_model(&dev) else { return };

    let wav: Tensor<3> = refs.tensor("C.codec_encode.in.0", &dev).unwrap();
    let want = to_vec(refs.tensor::<3>("C.codec_encode.out.0", &dev).unwrap());
    let t0 = Instant::now();
    let z = model.encode_waveform(wav, 24000, Some(codec::DEFAULT_NORMALIZE_DB), true).unwrap();
    assert_eq!(z.dims(), [1, 75, 32]);
    let got = to_vec(z);
    let dt = t0.elapsed().as_secs_f32();
    let (d, m) = max_abs_diff(&got, &want);
    eprintln!("encode C: {dt:.2}s CPU, max|diff|={d:.3e} (ref max {m:.3}, rel {:.2e})", d / m);
    assert_close("encode C", &got, &want, 1e-4);
}
