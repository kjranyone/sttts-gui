//! キャッシュ対応ストリーミング RNNT ASR(`vendor/nemotron_onnx_streaming.py` の移植)。
//!
//! HF processor のスケジュールを再現する: 先頭チャンクは `1 + 8*r` メルフレーム、
//! 以降は `8*(r+1)` フレーム(`r` は `chunk_ms` で決まる右コンテキスト)。
//! デコードは `ParakeetRNNTGenerationMixin` と同じ貪欲法で、ブランクかシンボル上限で
//! エンコーダフレームを進め、予測ネットワークの状態は非ブランクを消費したときだけ更新する。

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorRef};

use crate::mel::{HOP_LENGTH, LogMel, N_FFT, N_MELS, WIN_LENGTH};

const SUBSAMPLING: usize = 8;
/// sliding_window (57) - 1(サブサンプル後のエンコーダフレーム単位)
const LEFT_CONTEXT: usize = 56;
const DECODER_LAYERS: usize = 2;

/// chunk_ms → 右コンテキスト(ルックアヘッド)
fn lookahead(chunk_ms: u32) -> Option<usize> {
    Some(match chunk_ms {
        80 => 0,
        160 => 1,
        320 => 3,
        560 => 6,
        1120 => 13,
        _ => return None,
    })
}

/// ONNX グラフの探索(サブディレクトリ → フラット、精度付き → fp32 の順)
fn resolve_graph(dir: &Path, stem: &str, precision: &str) -> Result<std::path::PathBuf> {
    let suffix = if precision == "fp32" { String::new() } else { format!("_{precision}") };
    let mut cands = vec![
        dir.join(stem).join(format!("{stem}{suffix}.onnx")),
        dir.join(format!("{stem}{suffix}.onnx")),
    ];
    if !suffix.is_empty() {
        cands.push(dir.join(stem).join(format!("{stem}.onnx")));
        cands.push(dir.join(format!("{stem}.onnx")));
    }
    cands
        .into_iter()
        .find(|p| p.exists())
        .with_context(|| format!("ONNX グラフが見つかりません: {stem} ({})", dir.display()))
}

/// `tokens.txt`(id<TAB>piece)による id → テキスト。SentencePiece 形式で、
/// `▁` が語頭、`<0xNN>` はバイトフォールバック。
struct PieceDecoder {
    pieces: Vec<String>,
}

fn is_lang_tag(p: &str) -> bool {
    // ^<[a-z]{2,3}-[A-Z]{2,3}>$
    let Some(inner) = p.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else { return false };
    let Some((a, b)) = inner.split_once('-') else { return false };
    (2..=3).contains(&a.len())
        && a.bytes().all(|c| c.is_ascii_lowercase())
        && (2..=3).contains(&b.len())
        && b.bytes().all(|c| c.is_ascii_uppercase())
}

fn byte_token(p: &str) -> Option<u8> {
    let h = p.strip_prefix("<0x")?.strip_suffix('>')?;
    if h.len() != 2 {
        return None;
    }
    u8::from_str_radix(h, 16).ok()
}

impl PieceDecoder {
    fn load(path: &Path, vocab: usize) -> Self {
        let mut pieces = vec![String::new(); vocab];
        if let Ok(text) = std::fs::read_to_string(path) {
            for line in text.lines() {
                let (idx, piece) = line.split_once('\t').unwrap_or((line, ""));
                if let Ok(i) = idx.trim().parse::<usize>()
                    && i < vocab
                {
                    pieces[i] = piece.to_string();
                }
            }
        }
        Self { pieces }
    }

    fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        let mut buf: Vec<u8> = Vec::new();
        let flush = |buf: &mut Vec<u8>, out: &mut String| {
            if !buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(buf));
                buf.clear();
            }
        };
        for &id in ids {
            let Some(piece) = self.pieces.get(id).filter(|p| !p.is_empty()) else { continue };
            if let Some(b) = byte_token(piece) {
                buf.push(b);
                continue;
            }
            flush(&mut buf, &mut out);
            if piece.starts_with('<') && piece.ends_with('>') {
                continue;
            }
            out.push_str(piece);
        }
        flush(&mut buf, &mut out);
        out.replace('\u{2581}', " ").trim().to_string()
    }
}

/// エンコーダ 1 本(先頭用/定常用)。出力 i(i≥1)が書き戻すキャッシュ番号を持つ。
struct Encoder {
    session: Session,
    out_to_cache: Vec<usize>,
}

/// 1 発話ぶんのストリーミング状態
#[derive(Clone)]
struct State {
    caches: Vec<Vec<f32>>,
    cache_valid: usize,
    dec_h: Vec<f32>,
    dec_c: Vec<f32>,
    dec_out: Vec<f32>,
    dec_out_shape: Vec<i64>,
    tokens: Vec<usize>,
    mel_idx: usize,
}

pub struct Engine {
    enc_first: Encoder,
    enc_steady: Encoder,
    decoder: Session,
    joiner: Session,
    extractor: LogMel,
    pieces: PieceDecoder,
    cache_names: Vec<String>,
    cache_shapes: Vec<Vec<i64>>,
    blank_id: usize,
    max_symbols: usize,
    dec_hidden: usize,
    prompt_id: i64,
    enc_frames_per_chunk: usize,
    mel_frames_first: usize,
    mel_frames_steady: usize,
    samples_first: usize,
    samples_steady: usize,
    resume: Option<Resume>,
}

/// 直前の `transcribe` が完全なチャンクまで処理した時点の状態。
/// partial は伸びていくバッファを繰り返し渡されるので、先頭が同じなら続きから再開する
/// (結果は最初からやり直した場合と同一。完全チャンクは後続の音声に依存しない)。
struct Resume {
    /// 処理済みチャンクが参照した音声(先頭からここまで)
    head: Vec<f32>,
    state: State,
}

fn ort_err<T: std::fmt::Display>(e: T) -> anyhow::Error {
    anyhow!("{e}")
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    Session::builder()
        .map_err(ort_err)?
        .with_intra_threads(threads)
        .map_err(ort_err)?
        .with_inter_threads(1)
        .map_err(ort_err)?
        .commit_from_file(path)
        .map_err(ort_err)
        .with_context(|| format!("ONNX 読込: {}", path.display()))
}

fn encoder(path: &Path, threads: usize, cache_index: &HashMap<String, usize>) -> Result<Encoder> {
    let session = session(path, threads)?;
    let out_to_cache = session
        .outputs()
        .iter()
        .skip(1)
        .map(|o| {
            // "k_cache_out_3" → "k_cache_3"
            let name = o.name().replace("_out_", "_");
            cache_index.get(&name).copied().with_context(|| format!("未知のキャッシュ出力: {}", o.name()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Encoder { session, out_to_cache })
}

/// 言語キー("ja" 等)を Nemotron の locale 辞書に合わせる
pub fn normalize_language(lang: &str) -> String {
    let l = lang.trim();
    if l.is_empty() || l == "auto" {
        return "auto".into();
    }
    if l.contains('-') {
        return l.into();
    }
    match l {
        "ja" => "ja-JP",
        "en" => "en-US",
        "zh" => "zh-CN",
        "ko" => "ko-KR",
        _ => l,
    }
    .into()
}

impl Engine {
    pub fn new(model_dir: &Path, language: &str, chunk_ms: u32, precision: &str, num_threads: usize) -> Result<Self> {
        let r = lookahead(chunk_ms).with_context(|| format!("chunk_ms は 80/160/320/560/1120 のいずれか: {chunk_ms}"))?;
        let enc_frames_per_chunk = r + 1;
        let mel_frames_first = 1 + SUBSAMPLING * r;
        let mel_frames_steady = SUBSAMPLING * (r + 1);

        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("nemotron_onnx_config.json")).context("nemotron_onnx_config.json")?,
        )?;
        let vocab = meta["vocab_size"].as_u64().context("vocab_size")? as usize;
        let blank_id = meta["blank_token_id"].as_u64().context("blank_token_id")? as usize;
        let max_symbols = meta["max_symbols_per_step"].as_u64().context("max_symbols_per_step")? as usize;
        let dec_hidden = meta["decoder_hidden_size"].as_u64().unwrap_or(640) as usize;
        let prompt_id = meta["prompt_dictionary"][language]
            .as_i64()
            .with_context(|| format!("未知の言語 {language:?}(locale: ja-JP、bare: de、または auto)"))?;

        // serde_json は既定でキー順がソートされる。キャッシュ名の並びに意味は無い(名前で渡す)。
        let shapes = meta["cache_shapes"].as_object().context("cache_shapes")?;
        let mut cache_names = Vec::new();
        let mut cache_shapes = Vec::new();
        for (k, v) in shapes {
            cache_names.push(k.clone());
            cache_shapes.push(v.as_array().context("cache shape")?.iter().map(|d| d.as_i64().unwrap_or(0)).collect());
        }
        let cache_index: HashMap<String, usize> = cache_names.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();

        let stem = format!("encoder_{chunk_ms}ms");
        let enc_first = encoder(&resolve_graph(model_dir, &format!("{stem}_first"), precision)?, num_threads, &cache_index)?;
        let enc_steady = encoder(&resolve_graph(model_dir, &stem, precision)?, num_threads, &cache_index)?;
        let decoder = session(&resolve_graph(model_dir, "decoder", precision)?, num_threads)?;
        let joiner = session(&resolve_graph(model_dir, "joiner", precision)?, num_threads)?;

        Ok(Self {
            enc_first,
            enc_steady,
            decoder,
            joiner,
            extractor: LogMel::new(),
            pieces: PieceDecoder::load(&model_dir.join("tokens.txt"), vocab),
            cache_names,
            cache_shapes,
            blank_id,
            max_symbols,
            dec_hidden,
            prompt_id,
            enc_frames_per_chunk,
            mel_frames_first,
            mel_frames_steady,
            samples_first: (mel_frames_first - 1) * HOP_LENGTH + WIN_LENGTH / 2,
            samples_steady: mel_frames_steady * HOP_LENGTH + WIN_LENGTH,
            resume: None,
        })
    }

    fn fresh_state(&mut self) -> Result<State> {
        let caches = self.cache_shapes.iter().map(|s| vec![0f32; s.iter().product::<i64>() as usize]).collect();
        let n = DECODER_LAYERS * self.dec_hidden;
        let mut st = State {
            caches,
            cache_valid: 0,
            dec_h: vec![0.0; n],
            dec_c: vec![0.0; n],
            dec_out: Vec::new(),
            dec_out_shape: Vec::new(),
            tokens: Vec::new(),
            mel_idx: 0,
        };
        self.run_decoder(&mut st, self.blank_id)?;
        Ok(st)
    }

    /// 予測ネットワークを 1 トークン進めて状態を確定する
    fn run_decoder(&mut self, st: &mut State, token: usize) -> Result<()> {
        let hs = [DECODER_LAYERS as i64, 1, self.dec_hidden as i64];
        let outs = self
            .decoder
            .run(vec![
                (Cow::Borrowed("token"), SessionInputValue::from(Tensor::from_array(([1i64, 1], vec![token as i64])).map_err(ort_err)?)),
                (Cow::Borrowed("h_in"), SessionInputValue::from(TensorRef::from_array_view((hs, &*st.dec_h)).map_err(ort_err)?)),
                (Cow::Borrowed("c_in"), SessionInputValue::from(TensorRef::from_array_view((hs, &*st.dec_c)).map_err(ort_err)?)),
            ])
            .map_err(ort_err)?;
        let (shape, out) = outs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        st.dec_out.clear();
        st.dec_out.extend_from_slice(out);
        st.dec_out_shape = shape.iter().copied().collect();
        st.dec_h.copy_from_slice(outs[1].try_extract_tensor::<f32>().map_err(ort_err)?.1);
        st.dec_c.copy_from_slice(outs[2].try_extract_tensor::<f32>().map_err(ort_err)?.1);
        Ok(())
    }

    /// 1 チャンクぶんのエンコーダ出力 `enc`(`frames × hidden`)の先頭 `num_frames` を貪欲デコードする
    fn greedy_decode(&mut self, st: &mut State, enc: &[f32], num_frames: usize) -> Result<()> {
        let h = self.dec_hidden;
        for t in 0..num_frames {
            let frame = &enc[t * h..(t + 1) * h];
            let mut symbols = 0;
            loop {
                let token = {
                    let outs = self
                        .joiner
                        .run(vec![
                            (Cow::Borrowed("encoder_frame"), SessionInputValue::from(TensorRef::from_array_view(([1i64, h as i64], frame)).map_err(ort_err)?)),
                            (
                                Cow::Borrowed("decoder_out"),
                                SessionInputValue::from(
                                    TensorRef::from_array_view((st.dec_out_shape.clone(), &*st.dec_out)).map_err(ort_err)?,
                                ),
                            ),
                        ])
                        .map_err(ort_err)?;
                    let (_, logits) = outs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
                    let mut best = 0;
                    for (i, &v) in logits.iter().enumerate() {
                        if v > logits[best] {
                            best = i;
                        }
                    }
                    best
                };
                if token == self.blank_id {
                    break; // フレームを進める(予測状態はそのまま)
                }
                // 言語タグ(auto モードの <xx-XX>)は常に捨てる
                if !self.pieces.pieces.get(token).is_some_and(|p| is_lang_tag(p)) {
                    st.tokens.push(token);
                }
                self.run_decoder(st, token)?;
                symbols += 1;
                if symbols >= self.max_symbols {
                    break; // 強制的に進める
                }
            }
        }
        Ok(())
    }

    /// `features`: `mel_frames × N_MELS`。戻り値は `(frames × hidden)` のエンコーダ出力。
    fn run_encoder(&mut self, st: &mut State, features: &[f32], first: bool) -> Result<Vec<f32>> {
        let expected = if first { self.mel_frames_first } else { self.mel_frames_steady };
        if features.len() != expected * N_MELS {
            bail!("チャンクのメルフレーム数が不正: {} (期待 {expected})", features.len() / N_MELS);
        }
        let mut mask = vec![0f32; LEFT_CONTEXT + self.enc_frames_per_chunk];
        let invalid = LEFT_CONTEXT - st.cache_valid;
        mask[..invalid].fill(-1e9);
        let prompt = [self.prompt_id];

        let mut inputs: Vec<(Cow<str>, SessionInputValue)> = Vec::with_capacity(3 + self.cache_names.len());
        inputs.push((
            Cow::Borrowed("input_features"),
            SessionInputValue::from(TensorRef::from_array_view(([1i64, expected as i64, N_MELS as i64], features)).map_err(ort_err)?),
        ));
        inputs.push((Cow::Borrowed("prompt_ids"), SessionInputValue::from(TensorRef::from_array_view(([1i64], &prompt[..])).map_err(ort_err)?)));
        inputs.push((
            Cow::Borrowed("cache_mask"),
            SessionInputValue::from(TensorRef::from_array_view(([1i64, 1, 1, mask.len() as i64], &*mask)).map_err(ort_err)?),
        ));
        for ((name, shape), data) in self.cache_names.iter().zip(&self.cache_shapes).zip(&st.caches) {
            inputs.push((
                Cow::Borrowed(name.as_str()),
                SessionInputValue::from(TensorRef::from_array_view((shape.clone(), &**data)).map_err(ort_err)?),
            ));
        }
        let enc = if first { &mut self.enc_first } else { &mut self.enc_steady };
        let outs = enc.session.run(inputs).map_err(ort_err)?;
        for (i, &ci) in enc.out_to_cache.iter().enumerate() {
            let (_, v) = outs[i + 1].try_extract_tensor::<f32>().map_err(ort_err)?;
            st.caches[ci].copy_from_slice(v);
        }
        let result = outs[0].try_extract_tensor::<f32>().map_err(ort_err)?.1.to_vec();
        st.cache_valid = (st.cache_valid + self.enc_frames_per_chunk).min(LEFT_CONTEXT);
        Ok(result)
    }

    /// 末尾 `valid` フレームより後ろを 0 にし、`want` フレームへゼロ詰め/切り詰めする
    fn fit_features(mut feats: Vec<f32>, frames: usize, valid: Option<usize>, want: usize) -> Vec<f32> {
        if let Some(v) = valid
            && v < frames
        {
            feats[v * N_MELS..].fill(0.0);
        }
        feats.resize(want * N_MELS, 0.0);
        feats
    }

    fn first_chunk(&mut self, st: &mut State, pcm: &[f32], valid_mel: Option<usize>) -> Result<()> {
        let (feats, frames) = self.extractor.compute(pcm, true);
        let feats = Self::fit_features(feats, frames, valid_mel, self.mel_frames_first);
        let enc = self.run_encoder(st, &feats, true)?;
        st.mel_idx = self.mel_frames_first;
        let decode_frames = match valid_mel {
            None => self.enc_frames_per_chunk,
            Some(0) => 0,
            Some(v) => self.enc_frames_per_chunk.min(v.div_ceil(SUBSAMPLING).max(1)),
        };
        self.greedy_decode(st, &enc, decode_frames)
    }

    fn steady_chunk(&mut self, st: &mut State, window: &[f32], valid_mel: Option<usize>) -> Result<()> {
        let (feats, frames) = self.extractor.compute(window, false);
        let feats = Self::fit_features(feats, frames, valid_mel, self.mel_frames_steady);
        let enc = self.run_encoder(st, &feats, false)?;
        let decode_frames = match valid_mel {
            None => self.enc_frames_per_chunk,
            Some(v) => self.enc_frames_per_chunk.min(v.div_ceil(SUBSAMPLING).max(1)),
        };
        st.mel_idx += self.mel_frames_steady;
        self.greedy_decode(st, &enc, decode_frames)
    }

    /// 発話全体 → テキスト(Python の reset → accept_waveform → finish)
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let total = audio.len();
        if total == 0 {
            self.resume = None;
            return Ok(String::new());
        }
        let start_of = |mel_idx: usize| mel_idx * HOP_LENGTH - N_FFT / 2;
        if total < self.samples_first {
            // 先頭チャンクに満たない短い音声: ゼロ詰めして 1 チャンクだけ処理する
            self.resume = None;
            let mut st = self.fresh_state()?;
            let valid_mel = self.mel_frames_first.min(total / HOP_LENGTH);
            let mut padded = audio.to_vec();
            padded.resize(self.samples_first, 0.0);
            self.first_chunk(&mut st, &padded, Some(valid_mel))?;
            return Ok(self.pieces.decode(&st.tokens));
        }

        // 直前の呼び出しと先頭が同じなら、その時点の状態から再開する
        let resumed = match self.resume.take() {
            Some(r) if audio.len() >= r.head.len() && audio[..r.head.len()] == r.head[..] => Some(r.state),
            _ => None,
        };
        let mut st = match resumed {
            Some(st) => st,
            None => {
                let mut st = self.fresh_state()?;
                self.first_chunk(&mut st, &audio[..self.samples_first], None)?;
                st
            }
        };
        loop {
            let start = start_of(st.mel_idx);
            if start + self.samples_steady > total {
                break;
            }
            self.steady_chunk(&mut st, &audio[start..start + self.samples_steady], None)?;
        }
        // ここまでが後続の音声に依存しない部分。次回の再開用に保存する
        let head_len = if st.mel_idx == self.mel_frames_first {
            self.samples_first
        } else {
            start_of(st.mel_idx - self.mel_frames_steady) + self.samples_steady
        };
        self.resume = Some(Resume { head: audio[..head_len].to_vec(), state: st.clone() });

        // 末尾: 実音声が n_fft 以上残っていれば、ゼロ詰めした最終チャンクを処理する
        let start = start_of(st.mel_idx);
        let real = total.saturating_sub(start);
        let valid_mel = if real >= N_FFT { (real - N_FFT) / HOP_LENGTH + 1 } else { 0 };
        if valid_mel > 0 {
            let mut window = audio[start..total.min(start + self.samples_steady)].to_vec();
            window.resize(self.samples_steady, 0.0);
            self.steady_chunk(&mut st, &window, Some(valid_mel))?;
        }
        Ok(self.pieces.decode(&st.tokens))
    }
}
