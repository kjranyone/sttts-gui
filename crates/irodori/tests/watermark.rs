use irodori::testing::{cpu, hf_snapshot, refs};
use irodori::watermark::{IRODORI_PAYLOAD, Watermarker};

fn sdr(orig: &[f32], recon: &[f32]) -> f64 {
    let p: f64 = orig.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / orig.len() as f64;
    let e: f64 = orig.iter().zip(recon).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / orig.len() as f64;
    10.0 * (p / e).log10()
}

fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64 - ma, y as f64 - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

#[test]
fn watermark_matches_reference() {
    let Some(r) = refs() else {
        eprintln!("no refs; skipping");
        return;
    };
    let Some(snap) = hf_snapshot("models--sony--silentcipher") else {
        eprintln!("no silentcipher weights; skipping");
        return;
    };
    let wm = Watermarker::load(snap.join("44_1_khz/73999_iteration"), &cpu()).unwrap();
    let mut checked = 0;
    for case in ["A", "B", "C", "D"] {
        let (kin, kout) = (format!("{case}.watermark.in0.0"), format!("{case}.watermark.out0.0"));
        if !r.contains(&kin) {
            continue;
        }
        let (shape, input) = r.f32_vec(&kin).unwrap();
        let (_, want) = r.f32_vec(&kout).unwrap();
        let t0 = std::time::Instant::now();
        let got = wm.encode(&input, 48000, &IRODORI_PAYLOAD).unwrap();
        eprintln!("case {case}: shape {shape:?}, encode {:.2}s", t0.elapsed().as_secs_f64());
        assert_eq!(got.len(), want.len());
        let gd: Vec<f32> = got.iter().zip(&input).map(|(a, b)| a - b).collect();
        let wd: Vec<f32> = want.iter().zip(&input).map(|(a, b)| a - b).collect();
        let c = corr(&gd, &wd);
        let (sg, sw) = (sdr(&input, &got), sdr(&input, &want));
        let in_max = input.iter().fold(0f32, |m, v| m.max(v.abs()));
        let max_err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("case {case}: corr={c:.6} sdr got={sg:.3} want={sw:.3} max_err={max_err:.3e} (1e-3*in_max={:.3e})", 1e-3 * in_max);
        assert!(c >= 0.99, "watermark residual correlation {c}");
        assert!((sg - sw).abs() <= 1.0, "sdr {sg} vs {sw}");
        assert!(max_err <= 1e-3 * in_max, "max err {max_err}");
        checked += 1;
    }
    assert!(checked > 0, "no watermark cases in refs");
}
