//! テキスト/キャプション用トークナイザ(`tokenizer/tokenizer.json`, Unigram + Metaspace)。
//! 原典 `irodori_tts/tokenizer.py` の `PretrainedTextTokenizer.batch_encode` 相当。

use std::path::Path;

use anyhow::{Result, anyhow, bail};

/// `[batch][max_length]` のトークン ID とマスク
pub type Encoded = (Vec<Vec<i64>>, Vec<Vec<bool>>);

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: Option<i64>,
    pad_id: i64,
}

impl Tokenizer {
    /// `tokenizer.json` のパス、またはそれを含むディレクトリ(チェックポイントの `tokenizer/`)
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() { path.join("tokenizer.json") } else { path.to_path_buf() };
        let inner = tokenizers::Tokenizer::from_file(&file)
            .map_err(|e| anyhow!("load tokenizer {}: {e}", file.display()))?;
        let bos_id = inner.token_to_id("<s>").map(i64::from);
        // 原典は pad_token が無ければ eos を使う。このチェックポイントには `<pad>` がある。
        let pad_id = inner
            .token_to_id("<pad>")
            .or_else(|| inner.token_to_id("</s>"))
            .ok_or_else(|| anyhow!("tokenizer has neither <pad> nor </s>"))?;
        Ok(Self { inner, bos_id, pad_id: i64::from(pad_id) })
    }

    pub fn bos_id(&self) -> Option<i64> {
        self.bos_id
    }

    pub fn pad_id(&self) -> i64 {
        self.pad_id
    }

    /// `add_special_tokens=False` のトークン ID。
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i64::from(i)).collect())
    }

    /// 先頭に BOS(`add_bos`)を付け、`max_length` まで右パディング(長すぎれば右を切る)。
    /// 戻り値は `[batch][max_length]` の ID とマスク。
    pub fn batch_encode(
        &self,
        texts: &[String],
        max_length: usize,
        add_bos: bool,
    ) -> Result<Encoded> {
        if texts.is_empty() {
            bail!("texts must contain at least one item");
        }
        if max_length == 0 {
            bail!("max_length must be > 0");
        }
        let bos = if add_bos {
            Some(self.bos_id.ok_or_else(|| anyhow!("tokenizer has no <s> but add_bos=true"))?)
        } else {
            None
        };
        if let (Some(b), 1) = (bos, max_length) {
            return Ok((vec![vec![b]; texts.len()], vec![vec![true]; texts.len()]));
        }
        let body_max = if bos.is_some() { max_length - 1 } else { max_length };
        let mut ids = Vec::with_capacity(texts.len());
        let mut masks = Vec::with_capacity(texts.len());
        for t in texts {
            let mut body = self.encode(t)?;
            body.truncate(body_max);
            let mut row = Vec::with_capacity(max_length);
            let mut mask = Vec::with_capacity(max_length);
            if let Some(b) = bos {
                row.push(b);
                mask.push(true);
            }
            mask.extend(std::iter::repeat_n(true, body.len()));
            row.extend(body);
            row.resize(max_length, self.pad_id);
            mask.resize(max_length, false);
            ids.push(row);
            masks.push(mask);
        }
        Ok((ids, masks))
    }
}
