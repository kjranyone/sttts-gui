//! ストリーミング・リサンプラ(rubato の FFT 同期リサンプラを固定長チャンクで包む)。

use anyhow::{Result, anyhow};
use rubato::{FftFixedIn, Resampler};

/// 任意長の入力を受け、固定比でリサンプルした出力を返す。
///
/// 内部で入力を `chunk` サンプルずつに切って処理する(余りは次回へ持ち越し)。
/// 先頭のフィルタ遅延(`output_delay`)は捨てて、入出力の時間軸を揃える。
pub struct StreamResampler {
    inner: Option<FftFixedIn<f32>>,
    chunk: usize,
    pending: Vec<f32>,
    skip: usize,
}

impl StreamResampler {
    /// `chunk` は入力側のチャンク長(サンプル)。入力レートと出力レートが同じなら素通し。
    pub fn new(in_rate: u32, out_rate: u32, chunk: usize) -> Result<Self> {
        if in_rate == 0 || out_rate == 0 || chunk == 0 {
            return Err(anyhow!(
                "リサンプラの引数が不正です: {in_rate}->{out_rate}, chunk={chunk}"
            ));
        }
        let (inner, skip, chunk) = if in_rate == out_rate {
            (None, 0, chunk)
        } else {
            let r = FftFixedIn::<f32>::new(in_rate as usize, out_rate as usize, chunk, 2, 1)
                .map_err(|e| anyhow!("リサンプラ生成失敗: {e}"))?;
            let d = r.output_delay();
            // rubato は比に合わせて入力チャンク長を丸めることがある(44.1k など)。実際の長さを使う
            let c = r.input_frames_next();
            (Some(r), d, c)
        };
        Ok(Self {
            inner,
            chunk,
            pending: Vec::new(),
            skip,
        })
    }

    /// 入力を追加し、出せる分のリサンプル結果を `out` に追記する。
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        let Some(r) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            return Ok(());
        };
        self.pending.extend_from_slice(input);
        while self.pending.len() >= self.chunk {
            let block: Vec<f32> = self.pending.drain(..self.chunk).collect();
            let res = r
                .process(&[block], None)
                .map_err(|e| anyhow!("リサンプル失敗: {e}"))?;
            Self::push_out(&mut self.skip, &res[0], out);
        }
        Ok(())
    }

    fn push_out(skip: &mut usize, res: &[f32], out: &mut Vec<f32>) {
        let s = (*skip).min(res.len());
        *skip -= s;
        out.extend_from_slice(&res[s..]);
    }
}

/// 一括リサンプル(オフライン用)。入力長に比例した出力長になるよう終端を切る。
pub fn resample_all(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>> {
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }
    let chunk = 1024usize;
    let mut rs = StreamResampler::new(in_rate, out_rate, chunk)?;
    let mut out = Vec::with_capacity(input.len() * out_rate as usize / in_rate as usize + chunk);
    rs.process(input, &mut out)?;
    // 余り + フィルタ遅延ぶんの 0 を足して出し切る
    let tail = vec![0.0f32; 16384];
    rs.process(&tail, &mut out)?;
    let want = (input.len() as u64 * out_rate as u64 / in_rate as u64) as usize;
    out.truncate(want);
    Ok(out)
}
