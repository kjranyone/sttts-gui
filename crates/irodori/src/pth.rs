//! PyTorch `.pth`(zip + pickle)の state_dict を読む最小実装。Python / torch には依存しない。
//!
//! - pickle は protocol 2〜5 の必要最小限のオペコードだけを解釈する小さな VM。
//!   `torch._utils._rebuild_tensor_v2` / `collections.OrderedDict` / storage の persistent_id を扱う。
//! - テンソルの数値は dtype に関わらず f32 へ変換して保持する(Float/Double/Half/BFloat16/整数)。
//! - `weight_norm`(`weight_g` + `weight_v`、または parametrizations の `original0/1`)は
//!   [`Pth::fold_weight_norm`] で重みへ畳み込める。

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{Device, Tensor, TensorData};

/// f32 に変換済みの生テンソル
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl RawTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// 読み込んだ `.pth`
#[derive(Debug, Default)]
pub struct Pth {
    /// state_dict(名前順)
    pub tensors: BTreeMap<String, RawTensor>,
    /// トップレベルの `metadata`(無ければ `Null`)。pickle の dict/list/数値/文字列を JSON に写したもの
    pub metadata: serde_json::Value,
}

impl Pth {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load(path)
    }

    pub fn get(&self, name: &str) -> Result<&RawTensor> {
        self.tensors.get(name).ok_or_else(|| anyhow!("tensor not found: {name}"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// 名前でテンソルを取得(次元数 `D` を検査する)。
    pub fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>> {
        let t = self.get(name)?;
        if t.shape.len() != D {
            bail!("{name}: expected {D} dims, got shape {:?}", t.shape);
        }
        Ok(Tensor::<D>::from_data(TensorData::new(t.data.clone(), t.shape.clone()), device))
    }

    /// `weight_g` / `weight_v`(旧 `torch.nn.utils.weight_norm`)と
    /// `parametrizations.weight.original0/1`(新)を `weight = g * v / ||v||` へ畳み込む。
    /// ノルムは dim 0 以外の全次元で取る(`weight_g` の形が `[N,1,..]` の場合。
    /// ConvTranspose1d も dim 0 = 入力チャネルで同じ)。
    pub fn fold_weight_norm(&mut self) -> Result<()> {
        let mut pairs: Vec<(String, String, String)> = Vec::new(); // (g, v, out)
        for k in self.tensors.keys() {
            if let Some(base) = k.strip_suffix(".weight_g") {
                pairs.push((k.clone(), format!("{base}.weight_v"), format!("{base}.weight")));
            } else if let Some(base) = k.strip_suffix(".parametrizations.weight.original0") {
                pairs.push((
                    k.clone(),
                    format!("{base}.parametrizations.weight.original1"),
                    format!("{base}.weight"),
                ));
            }
        }
        for (gk, vk, out) in pairs {
            let g = self.tensors.remove(&gk).unwrap();
            let v = self.tensors.remove(&vk).with_context(|| format!("{vk} (pair of {gk})"))?;
            let n = v.shape[0];
            if g.numel() != n {
                bail!("weight_norm over dim 0 only: g shape {:?} vs v shape {:?}", g.shape, v.shape);
            }
            let per = v.numel() / n.max(1);
            let mut w = vec![0f32; v.data.len()];
            for i in 0..n {
                let row = &v.data[i * per..(i + 1) * per];
                let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                let s = (g.data[i] as f64 / norm.max(1e-12)) as f32;
                for (o, &x) in w[i * per..(i + 1) * per].iter_mut().zip(row) {
                    *o = x * s;
                }
            }
            self.tensors.insert(out, RawTensor { shape: v.shape, data: w });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// pickle VM
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum DType {
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DType {
    fn from_storage_name(n: &str) -> Option<Self> {
        Some(match n {
            "FloatStorage" => Self::F32,
            "DoubleStorage" => Self::F64,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::BF16,
            "LongStorage" => Self::I64,
            "IntStorage" => Self::I32,
            "ShortStorage" => Self::I16,
            "CharStorage" => Self::I8,
            "ByteStorage" => Self::U8,
            "BoolStorage" => Self::Bool,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    fn read(self, b: &[u8]) -> f32 {
        match self {
            Self::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            Self::F64 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::F16 => half::f16::from_le_bytes(b.try_into().unwrap()).to_f32(),
            Self::BF16 => half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32(),
            Self::I64 => i64::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::I16 => i16::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::I8 => b[0] as i8 as f32,
            Self::U8 | Self::Bool => b[0] as f32,
        }
    }
}

#[derive(Debug, Clone)]
struct TensorRef {
    key: String,
    dtype: DType,
    offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

#[derive(Debug, Clone)]
enum Obj {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes,
    Tuple(Vec<Obj>),
    List(Vec<Obj>),
    Dict(Vec<(Obj, Obj)>),
    Global(String, String),
    Storage { key: String, dtype: DType },
    Tensor(TensorRef),
    /// 解釈しないオブジェクト(クラス名のみ保持)
    Opaque,
    Mark,
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.i + n > self.b.len() {
            bail!("pickle: unexpected end");
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn line(&mut self) -> Result<String> {
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i] != b'\n' {
            self.i += 1;
        }
        let s = String::from_utf8_lossy(&self.b[start..self.i]).into_owned();
        self.i += 1;
        Ok(s)
    }
}

fn pop_mark(stack: &mut Vec<Obj>) -> Result<Vec<Obj>> {
    let pos = stack.iter().rposition(|o| matches!(o, Obj::Mark)).ok_or_else(|| anyhow!("pickle: no mark"))?;
    let items = stack.split_off(pos + 1);
    stack.pop();
    Ok(items)
}

fn as_usize_vec(o: &Obj) -> Result<Vec<usize>> {
    match o {
        Obj::Tuple(v) | Obj::List(v) => v
            .iter()
            .map(|x| match x {
                Obj::Int(i) if *i >= 0 => Ok(*i as usize),
                other => Err(anyhow!("expected non-negative int, got {other:?}")),
            })
            .collect(),
        other => Err(anyhow!("expected tuple of ints, got {other:?}")),
    }
}

fn reduce(callable: Obj, args: Obj) -> Result<Obj> {
    let args = match args {
        Obj::Tuple(v) => v,
        other => vec![other],
    };
    let Obj::Global(module, name) = &callable else {
        return Ok(Obj::Opaque);
    };
    match (module.as_str(), name.as_str()) {
        ("collections", "OrderedDict") => Ok(Obj::Dict(Vec::new())),
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            // (storage, storage_offset, size, stride, ...)
            let Some(Obj::Storage { key, dtype }) = args.first() else {
                bail!("_rebuild_tensor: first arg is not a storage: {:?}", args.first());
            };
            let offset = match args.get(1) {
                Some(Obj::Int(i)) => *i as usize,
                _ => 0,
            };
            let size = as_usize_vec(args.get(2).ok_or_else(|| anyhow!("missing size"))?)?;
            let stride = as_usize_vec(args.get(3).ok_or_else(|| anyhow!("missing stride"))?)?;
            Ok(Obj::Tensor(TensorRef { key: key.clone(), dtype: *dtype, offset, size, stride }))
        }
        ("torch._utils", "_rebuild_parameter") => args.into_iter().next().ok_or_else(|| anyhow!("empty args")),
        _ => Ok(Obj::Opaque),
    }
}

fn run_pickle(data: &[u8]) -> Result<Obj> {
    let mut r = Reader { b: data, i: 0 };
    let mut stack: Vec<Obj> = Vec::new();
    let mut memo: HashMap<usize, Obj> = HashMap::new();

    macro_rules! pop {
        () => {
            stack.pop().ok_or_else(|| anyhow!("pickle: stack underflow"))?
        };
    }

    loop {
        let op = r.u8()?;
        match op {
            0x80 => {
                r.u8()?; // PROTO
            }
            0x95 => {
                r.u64()?; // FRAME
            }
            b'.' => return stack.pop().ok_or_else(|| anyhow!("pickle: empty stack at STOP")),
            b'N' => stack.push(Obj::None),
            0x88 => stack.push(Obj::Bool(true)),
            0x89 => stack.push(Obj::Bool(false)),
            b'K' => {
                let v = r.u8()?;
                stack.push(Obj::Int(v as i64));
            }
            b'M' => {
                let v = r.u16()?;
                stack.push(Obj::Int(v as i64));
            }
            b'J' => {
                let v = r.u32()? as i32;
                stack.push(Obj::Int(v as i64));
            }
            0x8a | 0x8b => {
                let n = if op == 0x8a { r.u8()? as usize } else { r.u32()? as usize };
                let b = r.take(n)?;
                let mut v: i128 = 0;
                for (k, &x) in b.iter().enumerate().take(16) {
                    v |= (x as i128) << (8 * k);
                }
                if n > 0 && n < 16 && b[n - 1] & 0x80 != 0 {
                    v -= 1i128 << (8 * n);
                }
                stack.push(Obj::Int(v as i64));
            }
            b'G' => {
                let b: [u8; 8] = r.take(8)?.try_into().unwrap();
                stack.push(Obj::Float(f64::from_be_bytes(b)));
            }
            b'X' => {
                let n = r.u32()? as usize;
                stack.push(Obj::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            0x8c => {
                let n = r.u8()? as usize;
                stack.push(Obj::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            0x8d => {
                let n = r.u64()? as usize;
                stack.push(Obj::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            b'U' => {
                let n = r.u8()? as usize;
                stack.push(Obj::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            b'T' => {
                let n = r.u32()? as usize;
                stack.push(Obj::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            b'C' => {
                let n = r.u8()? as usize;
                r.take(n)?;
                stack.push(Obj::Bytes);
            }
            b'B' => {
                let n = r.u32()? as usize;
                r.take(n)?;
                stack.push(Obj::Bytes);
            }
            b')' => stack.push(Obj::Tuple(Vec::new())),
            b']' => stack.push(Obj::List(Vec::new())),
            b'}' => stack.push(Obj::Dict(Vec::new())),
            b'(' => stack.push(Obj::Mark),
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Obj::Tuple(items));
            }
            0x85 => {
                let a = pop!();
                stack.push(Obj::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop!();
                let a = pop!();
                stack.push(Obj::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop!();
                let b = pop!();
                let a = pop!();
                stack.push(Obj::Tuple(vec![a, b, c]));
            }
            b'l' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Obj::List(items));
            }
            b'd' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Obj::Dict(items.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect()));
            }
            b'a' => {
                let v = pop!();
                match stack.last_mut() {
                    Some(Obj::List(l)) => l.push(v),
                    other => bail!("APPEND on {other:?}"),
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Obj::List(l)) => l.extend(items),
                    other => bail!("APPENDS on {other:?}"),
                }
            }
            b's' => {
                let v = pop!();
                let k = pop!();
                match stack.last_mut() {
                    Some(Obj::Dict(d)) => d.push((k, v)),
                    other => bail!("SETITEM on {other:?}"),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Obj::Dict(d)) => d.extend(items.chunks(2).map(|c| (c[0].clone(), c[1].clone()))),
                    other => bail!("SETITEMS on {other:?}"),
                }
            }
            b'c' => {
                let m = r.line()?;
                let n = r.line()?;
                stack.push(Obj::Global(m, n));
            }
            0x93 => {
                let n = pop!();
                let m = pop!();
                match (m, n) {
                    (Obj::Str(m), Obj::Str(n)) => stack.push(Obj::Global(m, n)),
                    other => bail!("STACK_GLOBAL on {other:?}"),
                }
            }
            b'q' => {
                let k = r.u8()? as usize;
                memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("BINPUT on empty stack"))?);
            }
            b'r' => {
                let k = r.u32()? as usize;
                memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("LONG_BINPUT on empty stack"))?);
            }
            0x94 => {
                let k = memo.len();
                memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("MEMOIZE on empty stack"))?);
            }
            b'h' => {
                let k = r.u8()? as usize;
                stack.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("BINGET {k}: not in memo"))?);
            }
            b'j' => {
                let k = r.u32()? as usize;
                stack.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("LONG_BINGET {k}: not in memo"))?);
            }
            b'0' => {
                pop!();
            }
            b'1' => {
                pop_mark(&mut stack)?;
            }
            b'2' => {
                let t = stack.last().cloned().ok_or_else(|| anyhow!("DUP on empty stack"))?;
                stack.push(t);
            }
            b'R' => {
                let args = pop!();
                let callable = pop!();
                stack.push(reduce(callable, args)?);
            }
            0x81 => {
                // NEWOBJ(cls, args)
                let _args = pop!();
                pop!(); // cls
                stack.push(Obj::Opaque);
            }
            0x92 => {
                // NEWOBJ_EX(cls, args, kwargs)
                let _kw = pop!();
                let _args = pop!();
                pop!(); // cls
                stack.push(Obj::Opaque);
            }
            b'b' => {
                pop!(); // BUILD: state は使わない(OrderedDict の `_metadata` など)
            }
            b'Q' => {
                // BINPERSID: ('storage', storage_type, key, location, numel)
                let pid = pop!();
                let Obj::Tuple(v) = pid else { bail!("persistent id is not a tuple: {pid:?}") };
                match (v.first(), v.get(1), v.get(2)) {
                    (Some(Obj::Str(tag)), Some(Obj::Global(_, ty)), Some(Obj::Str(key))) if tag == "storage" => {
                        let dtype = DType::from_storage_name(ty)
                            .ok_or_else(|| anyhow!("unsupported storage type {ty}"))?;
                        stack.push(Obj::Storage { key: key.clone(), dtype });
                    }
                    _ => bail!("unsupported persistent id: {v:?}"),
                }
            }
            other => bail!("pickle: unsupported opcode 0x{other:02x} at {}", r.i - 1),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// zip / テンソルの実体化
// ---------------------------------------------------------------------------------------------

fn to_json(o: &Obj) -> serde_json::Value {
    use serde_json::Value;
    match o {
        Obj::None => Value::Null,
        Obj::Bool(b) => Value::Bool(*b),
        Obj::Int(i) => Value::from(*i),
        Obj::Float(f) => Value::from(*f),
        Obj::Str(s) => Value::String(s.clone()),
        Obj::Tuple(v) | Obj::List(v) => Value::Array(v.iter().map(to_json).collect()),
        Obj::Dict(d) => Value::Object(
            d.iter()
                .filter_map(|(k, v)| match k {
                    Obj::Str(s) => Some((s.clone(), to_json(v))),
                    _ => None,
                })
                .collect(),
        ),
        _ => Value::Null,
    }
}

fn dict_get<'a>(d: &'a [(Obj, Obj)], key: &str) -> Option<&'a Obj> {
    d.iter().find(|(k, _)| matches!(k, Obj::Str(s) if s == key)).map(|(_, v)| v)
}

fn materialize(t: &TensorRef, storage: &[u8]) -> Result<RawTensor> {
    let numel: usize = t.size.iter().product();
    let es = t.dtype.size();
    let contiguous = {
        let mut expect = 1usize;
        let mut ok = true;
        for (&s, &st) in t.size.iter().zip(&t.stride).rev() {
            if s != 1 && st != expect {
                ok = false;
            }
            expect *= s;
        }
        ok
    };
    let mut data = Vec::with_capacity(numel);
    if contiguous {
        let start = t.offset * es;
        let end = start + numel * es;
        if end > storage.len() {
            bail!("tensor out of storage bounds ({end} > {})", storage.len());
        }
        data.extend(storage[start..end].chunks_exact(es).map(|c| t.dtype.read(c)));
    } else {
        let nd = t.size.len();
        let mut idx = vec![0usize; nd];
        for _ in 0..numel {
            let pos: usize = t.offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
            let b = storage.get(pos * es..(pos + 1) * es).ok_or_else(|| anyhow!("strided read out of bounds"))?;
            data.push(t.dtype.read(b));
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < t.size[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }
    Ok(RawTensor { shape: t.size.clone(), data })
}

/// `.pth` を読み込む。トップレベルが `{"state_dict": ..., "metadata": ...}` でも、state_dict そのものでもよい。
pub fn load(path: impl AsRef<Path>) -> Result<Pth> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file)).context("not a zip-format .pth")?;

    let pkl_name = (0..zip.len())
        .filter_map(|i| zip.by_index_raw(i).ok().map(|f| f.name().to_string()))
        .find(|n| n == "data.pkl" || n.ends_with("/data.pkl"))
        .ok_or_else(|| anyhow!("data.pkl not found in {}", path.display()))?;
    let prefix = pkl_name.strip_suffix("data.pkl").unwrap().to_string();
    let mut pkl = Vec::new();
    zip.by_name(&pkl_name)?.read_to_end(&mut pkl)?;

    let root = run_pickle(&pkl).context("pickle")?;
    let Obj::Dict(top) = &root else { bail!("top-level object is not a dict: {root:?}") };
    let (sd, metadata) = match dict_get(top, "state_dict") {
        Some(Obj::Dict(sd)) => (sd.as_slice(), dict_get(top, "metadata").map(to_json).unwrap_or_default()),
        _ => (top.as_slice(), serde_json::Value::Null),
    };

    let mut out = Pth { tensors: BTreeMap::new(), metadata };
    let mut cache: HashMap<String, Vec<u8>> = HashMap::new();
    for (k, v) in sd {
        let (Obj::Str(name), Obj::Tensor(t)) = (k, v) else { continue };
        if !cache.contains_key(&t.key) {
            let mut buf = Vec::new();
            zip.by_name(&format!("{prefix}data/{}", t.key))
                .with_context(|| format!("storage {} of {name}", t.key))?
                .read_to_end(&mut buf)?;
            cache.insert(t.key.clone(), buf);
        }
        out.tensors.insert(name.clone(), materialize(t, &cache[&t.key]).with_context(|| name.clone())?);
    }
    Ok(out)
}
