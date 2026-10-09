//! 自己回帰デコード(greedy / beam search)。
//!
//! beam search は transformers の `_beam_search`(`length_penalty=1.0`、`early_stopping=False`)と
//! 同じ規則: 候補は全ビームから累積対数確率の上位 `2 * beam` 個、うち EOS で終わったものは先頭
//! `beam` 個に入っていれば終了済みに加え(スコア = 累積 / 生成長)、残りの先頭 `beam` 個で継続する。
//! 終了済みが `beam` 個そろい、かつ継続中の最良スコアが終了済みの最低スコア以下になったら止める。
//! 対数確率は suppress 適用前の log_softmax(HF と同じく、抑制したトークンの分は再正規化しない)。

use anyhow::Result;

use crate::model::{CrossKv, MAX_TARGET, Model, SelfCache};

/// デコードの特殊トークンと抑制リスト
pub struct DecodeSpec {
    pub prompt: Vec<i64>,
    pub eos: i64,
    pub suppress: Vec<usize>,
    /// 最初に生成するトークンだけ追加で抑制する(空白・EOS)
    pub begin_suppress: Vec<usize>,
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let lse = max + sum.ln();
    logits.iter().map(|&x| x - lse).collect()
}

fn suppress(lp: &mut [f32], ids: &[usize]) {
    for &i in ids {
        if let Some(v) = lp.get_mut(i) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// 生成したトークン列(EOS とプロンプトを含まない)を返す。
pub fn generate(model: &Model, spec: &DecodeSpec, cross: &CrossKv, beam_size: usize) -> Result<Vec<i64>> {
    let mut cache = model.new_cache();
    let logits = model.decode_step(&spec.prompt, &mut cache, cross)?;
    if beam_size <= 1 {
        greedy(model, spec, cross, cache, logits)
    } else {
        beam_search(model, spec, cross, cache, logits, beam_size)
    }
}

fn greedy(
    model: &Model,
    spec: &DecodeSpec,
    cross: &CrossKv,
    mut cache: SelfCache,
    mut logits: Vec<f32>,
) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    loop {
        let mut lp = log_softmax(&logits);
        suppress(&mut lp, &spec.suppress);
        if out.is_empty() {
            suppress(&mut lp, &spec.begin_suppress);
        }
        let tok = lp.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0
            as i64;
        if tok == spec.eos || spec.prompt.len() + out.len() + 1 >= MAX_TARGET {
            break;
        }
        out.push(tok);
        logits = model.decode_step(&[tok], &mut cache, cross)?;
    }
    Ok(out)
}

struct Running {
    tokens: Vec<i64>,
    score: f32,
    cache: SelfCache,
    /// 次トークンの logits(この系列の最後まで処理した結果)
    logits: Vec<f32>,
}

fn beam_search(
    model: &Model,
    spec: &DecodeSpec,
    cross: &CrossKv,
    cache: SelfCache,
    logits: Vec<f32>,
    beams: usize,
) -> Result<Vec<i64>> {
    let k = 2 * beams;
    let mut running = vec![Running { tokens: vec![], score: 0.0, cache, logits }];
    // (スコア, トークン列)。スコアの降順
    let mut finished: Vec<(f32, Vec<i64>)> = Vec::new();
    loop {
        // 全ビームの候補から上位 k
        let mut cands: Vec<(f32, usize, usize)> = Vec::new();
        for (bi, b) in running.iter().enumerate() {
            let mut lp = log_softmax(&b.logits);
            suppress(&mut lp, &spec.suppress);
            if b.tokens.is_empty() {
                suppress(&mut lp, &spec.begin_suppress);
            }
            if lp.iter().all(|v| *v == f32::NEG_INFINITY) {
                continue;
            }
            // 各ビームの上位 k だけ残せば全体の上位 k に足りる
            let mut idx: Vec<usize> = (0..lp.len()).filter(|&i| lp[i] > f32::NEG_INFINITY).collect();
            let take = k.min(idx.len());
            idx.select_nth_unstable_by(take.saturating_sub(1), |&a, &b| lp[b].total_cmp(&lp[a]));
            cands.extend(idx[..take].iter().map(|&t| (b.score + lp[t], bi, t)));
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        cands.truncate(k);

        let cur_len = spec.prompt.len() + running[0].tokens.len();
        let at_limit = cur_len + 1 >= MAX_TARGET;
        let hits: Vec<bool> = cands.iter().map(|c| c.2 as i64 == spec.eos || at_limit).collect();

        // 終了済みの更新(先頭 beams 個に入った終了候補だけ)
        let gen_len = (running[0].tokens.len() + 1) as f32;
        for (i, c) in cands.iter().enumerate().take(beams) {
            if hits[i] {
                let mut toks = running[c.1].tokens.clone();
                if c.2 as i64 != spec.eos {
                    toks.push(c.2 as i64);
                }
                finished.push((c.0 / gen_len, toks));
            }
        }
        finished.sort_by(|a, b| b.0.total_cmp(&a.0));
        finished.truncate(beams);

        if hits.iter().all(|&h| h) {
            break;
        }
        // 継続: 終了しなかった候補の上位 beams 個
        let next: Vec<_> = cands.iter().zip(&hits).filter(|(_, h)| !**h).map(|(c, _)| *c).take(beams).collect();
        let mut new_running = Vec::with_capacity(next.len());
        for (score, bi, tok) in next {
            let parent = &running[bi];
            let mut cache = parent.cache.clone();
            let logits = model.decode_step(&[tok as i64], &mut cache, cross)?;
            let mut tokens = parent.tokens.clone();
            tokens.push(tok as i64);
            new_running.push(Running { tokens, score, cache, logits });
        }
        running = new_running;

        // 打ち切り: 終了済みが beams 個そろい、継続中の最良がそれ以下
        if finished.len() >= beams {
            let best = running[0].score / running[0].tokens.len() as f32;
            let worst = finished.last().map_or(f32::NEG_INFINITY, |f| f.0);
            if best <= worst {
                break;
            }
        }
    }
    Ok(finished.into_iter().next().map(|f| f.1).unwrap_or_default())
}
