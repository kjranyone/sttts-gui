//! safetensors の読み込み(mmap)。dtype は F32 / F16 / BF16 を f32 へ変換して返す。

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
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
}

impl Weights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            bail!("not a safetensors file: {}", path.display());
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&mmap[8..8 + n]).context("safetensors header")?;
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
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"].as_array().ok_or_else(|| anyhow!("offsets of {k}"))?;
            entries.insert(
                k.clone(),
                Entry {
                    dtype,
                    shape,
                    start: off[0].as_u64().unwrap_or(0) as usize,
                    end: off[1].as_u64().unwrap_or(0) as usize,
                },
            );
        }
        Ok(Self { mmap, base: 8 + n, entries, metadata })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.entry(name)?.shape)
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or_else(|| anyhow!("tensor not found: {name}"))
    }

    /// (shape, f32 値) を返す。
    pub fn f32_vec(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
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
