//! テキスト・キャプション条件(normalize / tokenizer / ModernBERT / projector)の PyTorch 参照との一致。

use std::time::Instant;

use irodori::condition::{TextConditioner, caption_inputs};
use irodori::testing::{self, assert_close, to_vec};
use irodori::text::normalize_text;
use irodori::tokenizer::Tokenizer;
use irodori::weights::Weights;

#[test]
fn normalize_text_matches_python() {
    let data = include_str!("data/normalize_cases.tsv");
    let mut n = 0;
    for line in data.lines().filter(|l| !l.is_empty()) {
        let (a, b) = line.split_once('\t').unwrap();
        let input: String = serde_json::from_str(a).unwrap();
        let want: String = serde_json::from_str(b).unwrap();
        assert_eq!(normalize_text(&input), want, "input {input:?}");
        n += 1;
    }
    assert!(n >= 20);
}

fn rows_i64(refs: &Weights, key: &str) -> Vec<Vec<i64>> {
    let (shape, v) = refs.i64_vec(key).unwrap();
    v.chunks(shape[1]).map(<[i64]>::to_vec).collect()
}

fn rows_bool(refs: &Weights, key: &str) -> Vec<Vec<bool>> {
    let (shape, v) = refs.i64_vec(key).unwrap();
    v.chunks(shape[1]).map(|c| c.iter().map(|&x| x != 0).collect()).collect()
}

const TEXT_A: &str = "こんにちは、よろしくお願いします。";
const CAPTION_B: &str = "落ち着いた女性の声で、ゆっくり話す。";
const TEXT_D: &str = "今日は朝から小雨が降っていましたが、午後にはすっかり晴れて、夕方の空がとてもきれいでした。";

#[test]
fn text_and_caption_conditions_match_pytorch() {
    let Some(refs) = testing::refs() else {
        eprintln!("no reference outputs (run tools/reference/dump_irodori_ref.py); skipping");
        return;
    };
    let Some(dir) = testing::checkpoint_dir() else {
        eprintln!("no checkpoint in HF cache; skipping");
        return;
    };
    let tok = Tokenizer::load(dir.join("tokenizer")).unwrap();
    let w = Weights::open(dir.join("model.safetensors")).unwrap();
    let dev = testing::device();
    let t0 = Instant::now();
    let cond = TextConditioner::load(&w, &dev).unwrap();
    eprintln!("load: {:.1}s", t0.elapsed().as_secs_f32());
    let (max_text, max_cap) = (cond.cfg.max_text_len, cond.max_caption_len());

    // トークナイザ: ID とマスクが完全一致(テキスト A/D、キャプション A=空 / B)
    for (case, text) in [("A", TEXT_A), ("D", TEXT_D)] {
        let norm = normalize_text(text);
        let (ids, mask) = tok.batch_encode(&[norm.trim().to_string()], max_text, cond.cfg.text_add_bos).unwrap();
        assert_eq!(ids, rows_i64(&refs, &format!("{case}.tok_text.out0.0")), "{case} text ids");
        assert_eq!(mask, rows_bool(&refs, &format!("{case}.tok_text.out1.0")), "{case} text mask");
    }
    for (case, cap) in [("A", ""), ("B", CAPTION_B)] {
        let (ids, mask) = tok.batch_encode(&[cap.to_string()], max_cap, cond.caption_add_bos()).unwrap();
        assert_eq!(ids, rows_i64(&refs, &format!("{case}.tok_caption.out0.0")), "{case} caption ids");
        assert_eq!(mask, rows_bool(&refs, &format!("{case}.tok_caption.out1.0")), "{case} caption mask");
    }

    // ModernBERT 単体(参照の入力 ID/マスクを使う)。text: A.0 / D.0、caption: B.1
    for (key, n) in [("A", 0), ("B", 1), ("D", 0)] {
        let ids = rows_i64(&refs, &format!("{key}.backbone.in.ids.{n}"));
        let mask = rows_bool(&refs, &format!("{key}.backbone.in.mask.{n}"));
        let t = Instant::now();
        let out = cond.backbone().forward(&ids, &mask);
        let dt = t.elapsed().as_secs_f32();
        let want = refs.f32_vec(&format!("{key}.backbone.out.{n}")).unwrap().1;
        eprintln!("backbone {key}.{n}: {dt:.1}s ({} tokens)", ids[0].len());
        // 25 層でも実測 1e-6 級(最大絶対値比 <1e-6)なので 1e-4 で固定
        assert_close(&format!("backbone.out {key}.{n}"), &to_vec(out), &want, 1e-4);
    }

    // encode_text / encode_caption(トークナイザ出力から。空キャプションはマスク全 false)
    for (case, text, cap) in [("A", TEXT_A, ""), ("B", TEXT_A, CAPTION_B), ("D", TEXT_D, "")] {
        let norm = normalize_text(text);
        let (ids, mask) = tok.batch_encode(&[norm.trim().to_string()], max_text, cond.cfg.text_add_bos).unwrap();
        let text_state = cond.encode_text(&ids, &mask);
        let want = refs.f32_vec(&format!("{case}.encode_conditions.out0.0")).unwrap().1;
        assert_close(&format!("encode_conditions.out0 {case}"), &to_vec(text_state), &want, 1e-4);
        assert_eq!(mask, rows_bool(&refs, &format!("{case}.encode_conditions.out1.0")), "{case} out1");

        let (cids, cmask) =
            caption_inputs(&tok, Some(cap), 1, max_cap, cond.caption_add_bos()).unwrap();
        let cap_state = cond.encode_caption(&cids, &cmask).unwrap();
        let want = refs.f32_vec(&format!("{case}.encode_conditions.out4.0")).unwrap().1;
        assert_close(&format!("encode_conditions.out4 {case}"), &to_vec(cap_state), &want, 1e-4);
        assert_eq!(cmask, rows_bool(&refs, &format!("{case}.encode_conditions.out5.0")), "{case} out5");
    }
}
