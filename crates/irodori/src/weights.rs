//! safetensors の読み込み(mmap)。dtype は F32 / F16 / BF16 を f32 へ変換して返す。
//!
//! torchao で量子化したチェックポイント(`Aratako/Irodori-TTS-*-Quantized`)も読む。量子化した重みは
//! `<親>._weight_qdata`(I8 `[out, in]`)と `<親>._weight_scale`(行ごとの scale `[out, 1]`)に分かれて入っており、
//! 元の名前(`<親>.weight`)で引くと `qdata * scale` に戻した f32 を返す。int8 のまま使うときは [`Weights::int8`]。
//! 対応は int8 weight-only(W8A16)のみ。活性も量子化する版や float8 / int4 はエラーにする。

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use memmap2::Mmap;
use serde_json::Value;

#[derive(Debug, Clone)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct Weights {
    mmap: Mmap,
    base: usize,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
    /// 量子化した重み: 元の名前 → (qdata の名前, scale の名前)
    quantized: HashMap<String, (String, String)>,
}

/// int8 weight-only の重み(torchao `Int8Tensor`)。`[out, in]` の値と、行ごとの scale
pub struct Int8Weight {
    pub shape: [usize; 2],
    pub values: Vec<i8>,
    pub row_scales: Vec<f32>,
}

/// 対応する量子化(`irodori_quantization_json` の `quantization_type`)
const SUPPORTED_QUANTIZATION: &str = "int8_weight_only";

impl Weights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            bail!("not a safetensors file: {}", path.display());
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        if n > mmap.len() - 8 {
            bail!("truncated safetensors header in {} (header {} bytes, file {} bytes)", path.display(), n, mmap.len());
        }
        let header: Value = serde_json::from_slice(&mmap[8..8 + n]).context("safetensors header")?;
        let data_len = mmap.len() - 8 - n;
        let obj = header.as_object().ok_or_else(|| anyhow!("bad header"))?;
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (mk, mv) in m {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = v["dtype"].as_str().ok_or_else(|| anyhow!("dtype of {k}"))?.to_string();
            let shape = v["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("shape of {k}"))?
                .iter()
                .map(|x| x.as_u64().map(|d| d as usize).ok_or_else(|| anyhow!("bad shape of {k}")))
                .collect::<Result<Vec<_>>>()?;
            let off = v["data_offsets"].as_array().ok_or_else(|| anyhow!("offsets of {k}"))?;
            let (start, end) = match (off.first().and_then(Value::as_u64), off.get(1).and_then(Value::as_u64)) {
                (Some(s), Some(e)) => (s as usize, e as usize),
                _ => bail!("bad data_offsets of {k}"),
            };
            // 途中までしかダウンロードされていないファイルで、範囲外を読んで落ちないようにする
            if start > end || end > data_len {
                bail!("tensor {k} is out of range in {} (file truncated?)", path.display());
            }
            entries.insert(k.clone(), Entry { dtype, shape, start, end });
        }
        let quantized = quantized_tensors(&metadata, &entries).with_context(|| format!("{}", path.display()))?;
        Ok(Self { mmap, base: 8 + n, entries, metadata, quantized })
    }

    /// torchao で量子化したチェックポイントか
    pub fn is_quantized(&self) -> bool {
        !self.quantized.is_empty()
    }

    /// 量子化した重みか(元の名前で引く)
    pub fn is_int8(&self, name: &str) -> bool {
        self.quantized.contains_key(name)
    }

    /// int8 の重みをそのまま返す(GPU に int8 のまま置くため)
    pub fn int8(&self, name: &str) -> Result<Int8Weight> {
        let (q, sc) = self.quantized.get(name).ok_or_else(|| anyhow!("not an int8 tensor: {name}"))?;
        let e = self.entry(q)?;
        let [out, inp] = e.shape[..] else { bail!("{name}: int8 weight must be 2-D, got {:?}", e.shape) };
        let values: Vec<i8> = self.mmap[self.base + e.start..self.base + e.end].iter().map(|&b| b as i8).collect();
        let (sshape, row_scales) = self.f32_vec(sc)?;
        ensure!(sshape == [out, 1], "{name}: scale shape {sshape:?} != [{out}, 1] (only per-row int8 is supported)");
        ensure!(values.len() == out * inp, "size mismatch for {name}");
        Ok(Int8Weight { shape: [out, inp], values, row_scales })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name) || self.quantized.contains_key(name)
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        match self.quantized.get(name) {
            Some((q, _)) => Ok(&self.entry(q)?.shape),
            None => Ok(&self.entry(name)?.shape),
        }
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or_else(|| anyhow!("tensor not found: {name}"))
    }

    /// (shape, f32 値) を返す。量子化した重みは `qdata * scale` に戻す。
    pub fn f32_vec(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        if self.quantized.contains_key(name) {
            let q = self.int8(name)?;
            let [_, inp] = q.shape;
            let data = q.values.iter().enumerate().map(|(i, &v)| f32::from(v) * q.row_scales[i / inp]).collect();
            return Ok((q.shape.to_vec(), data));
        }
        let e = self.entry(name)?;
        let bytes = &self.mmap[self.base + e.start..self.base + e.end];
        let numel: usize = e.shape.iter().product();
        let out: Vec<f32> = match e.dtype.as_str() {
            "F32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            other => bail!("unsupported dtype {other} for {name}"),
        };
        if out.len() != numel {
            bail!("size mismatch for {name}");
        }
        Ok((e.shape.clone(), out))
    }

    /// 名前でテンソルを取得(次元数 `D` を検査する)。
    pub fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>> {
        let (shape, data) = self.f32_vec(name)?;
        if shape.len() != D {
            bail!("{name}: expected {D} dims, got shape {shape:?}");
        }
        Ok(Tensor::<D>::from_data(TensorData::new(data, shape), device))
    }

    /// 整数テンソル(I64 / I32)を i64 の平坦ベクトルで返す(トークン ID など)。
    pub fn i64_vec(&self, name: &str) -> Result<(Vec<usize>, Vec<i64>)> {
        let e = self.entry(name)?;
        let bytes = &self.mmap[self.base + e.start..self.base + e.end];
        let out: Vec<i64> = match e.dtype.as_str() {
            "I64" => bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            "I32" => bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            "BOOL" => bytes.iter().map(|&b| b as i64).collect(),
            other => bail!("unsupported int dtype {other} for {name}"),
        };
        Ok((e.shape.clone(), out))
    }
}

/// メタデータから量子化した重みを集める。対応外の量子化はここでエラーにする(黙って誤った値で動かさない)。
fn quantized_tensors(metadata: &HashMap<String, String>, entries: &HashMap<String, Entry>) -> Result<HashMap<String, (String, String)>> {
    let Some(raw) = metadata.get("irodori_quantization_json") else { return Ok(HashMap::new()) };
    let q: Value = serde_json::from_str(raw).context("irodori_quantization_json")?;
    let (version, backend, kind) = (q["format_version"].as_u64(), q["backend"].as_str(), q["quantization_type"].as_str().unwrap_or(""));
    ensure!(version == Some(1) && backend == Some("torchao"), "unsupported quantization format: {raw}");
    ensure!(
        kind == SUPPORTED_QUANTIZATION,
        "unsupported quantization {kind:?} (the Rust Irodori supports {SUPPORTED_QUANTIZATION}, i.e. the int8-weight-only checkpoints)"
    );
    let mut out = HashMap::new();
    for (name, v) in metadata {
        // テンソルごとの記述は JSON(`{"_type": "Tensor" | "Int8Tensor" | ...}`)。それ以外のメタデータは飛ばす
        let Ok(desc) = serde_json::from_str::<Value>(v) else { continue };
        match desc.get("_type").and_then(Value::as_str) {
            Some("Int8Tensor") => {}
            Some("Tensor") | None => continue,
            Some(other) => bail!("unsupported quantized tensor {name}: {other}"),
        }
        ensure!(desc["_data"]["act_quant_kwargs"].is_null(), "{name}: activation quantization is not supported");
        let (parent, attr) = name.rsplit_once('.').ok_or_else(|| anyhow!("bad quantized tensor name {name}"))?;
        let (qd, sc) = (format!("{parent}._{attr}_qdata"), format!("{parent}._{attr}_scale"));
        ensure!(entries.get(&qd).is_some_and(|e| e.dtype == "I8"), "{name}: missing int8 data {qd}");
        ensure!(entries.contains_key(&sc), "{name}: missing scale {sc}");
        out.insert(name.clone(), (qd, sc));
    }
    ensure!(!out.is_empty(), "quantized checkpoint without quantized tensors");
    Ok(out)
}
