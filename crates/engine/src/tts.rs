//! TTS エンジン。実体は `irodori`(純 Rust の Irodori-TTS)。テスト・GUI 開発用にビープを返すモックも持つ。
//!
//! Irodori の項目を塞がない: `tts.sampling`(項目名は Irodori と同じ)で `SamplingRequest` の全項目を指定できる。
//! 発話ごとにアプリが決める項目(`text` / `caption` / `ref_*` / `no_ref` / `seed`)だけは上書きさせず、黙って捨てずエラーにする。

use std::collections::{BTreeMap, HashMap};
use std::f32::consts::PI;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};

use crate::util::lock;

/// アプリが発話ごとに決める SamplingRequest 項目(`tts.sampling` では上書きさせない)
pub const RESERVED_SAMPLING_KEYS: [&str; 9] =
    ["text", "caption", "ref_wav", "ref_wavs", "ref_latent", "ref_latents", "ref_embed", "no_ref", "seed"];

/// 1 回の合成の指定
pub struct TtsRequest<'a> {
    pub text: &'a str,
    pub caption: Option<&'a str>,
    pub ref_wavs: &'a [String],
    pub seed: Option<u64>,
    /// 既定(`tts.sampling`)に発話ごとの上書き(duration_scale 等)を重ねたもの
    pub sampling: &'a Map<String, Value>,
}

#[derive(Debug)]
pub struct TtsOutput {
    /// 16bit PCM の WAV
    pub wav: Vec<u8>,
    pub sample_rate: u32,
    pub duration_ms: u64,
    pub gen_ms: u64,
    pub used_seed: Option<i64>,
    /// 段階別時間(ms)
    pub stages: Option<BTreeMap<String, f64>>,
    /// Irodori からの通知(透かしが使えない、参照音声をトリムした 等)。GUI のログへ流す
    pub messages: Vec<String>,
}

pub trait TtsEngine: Send + Sync {
    fn model_id(&self) -> &str;
    fn synthesize(&self, req: &TtsRequest) -> Result<TtsOutput>;
    /// 初回カーネルのコンパイルなどを先払いする(失敗しても致命的ではない)
    fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// `tts.sampling` を検証する。予約キーはエラー(黙って捨てると効かない原因が追えない)。
pub fn check_sampling_overrides(sampling: &Map<String, Value>) -> Result<()> {
    let mut reserved: Vec<&str> = RESERVED_SAMPLING_KEYS.iter().copied().filter(|k| sampling.contains_key(*k)).collect();
    reserved.sort_unstable();
    if !reserved.is_empty() {
        bail!("tts.sampling に指定できないキー(発話ごとにアプリが決定): {reserved:?}");
    }
    Ok(())
}

/// `tts.sampling` の値の型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingKind {
    Int,
    Float,
    Bool,
}

/// `tts.sampling` で指定できる Irodori の項目(GUI はこの一覧から編集欄を作る)。
/// [`apply_sampling`] が受け付ける項目と一致させる(テストで検査)。
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingField {
    /// Irodori の `SamplingRequest` の項目名
    pub key: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    pub kind: SamplingKind,
    /// Irodori の既定値(`SamplingRequest::default()`)。null は「なし」
    pub default: Value,
    /// null を明示指定できる項目で、null の意味(例: 正規化しない)。既定が null の項目は None
    pub null_label: Option<&'static str>,
}

/// `tts.sampling` で指定できる全項目と Irodori の既定値
pub fn sampling_fields() -> Vec<SamplingField> {
    let d = irodori::pipeline::SamplingRequest::default();
    let f = |key, label, help, kind, default: Value| SamplingField { key, label, help, kind, default, null_label: None };
    use SamplingKind::*;
    vec![
        f("num_steps", "ステップ数", "多いほど丁寧だが遅い(MeanFlow は少数で足りる)", Int, d.num_steps.into()),
        f("duration_scale", "長さの倍率", "1 より大きいとゆっくり、小さいと速く読む", Float, d.duration_scale.into()),
        f("seconds", "長さ(秒、固定)", "指定すると長さ予測を使わずこの長さで合成する", Float, d.seconds.into()),
        f("min_seconds", "最短(秒)", "予測した長さの下限", Float, d.min_seconds.into()),
        f("max_seconds", "最長(秒)", "予測した長さの上限", Float, d.max_seconds.into()),
        f("max_ref_seconds", "参照音声の上限(秒)", "長い参照音声はこの長さで切る(空 = 切らない)", Float, d.max_ref_seconds.into()),
        SamplingField {
            null_label: Some("正規化しない"),
            ..f("ref_normalize_db", "参照音声の音量(LUFS)", "参照音声をこの音量にそろえてから使う", Float, d.ref_normalize_db.map(f32_value).into())
        },
        f("ref_ensure_max", "参照音声の音割れ防止", "正規化しないとき、ピークが 1.0 を超えたら縮める", Bool, d.ref_ensure_max.into()),
        f("trim_tail", "末尾の無音を削る", "生成音声の末尾にある無音・ノイズを切る", Bool, d.trim_tail.into()),
        f("tail_window_size", "末尾判定の窓", "末尾判定に使う潜在フレーム数", Int, d.tail_window_size.into()),
        f("tail_std_threshold", "末尾判定の std 閾値", "これより小さい揺れを無音とみなす", Float, f32_value(d.tail_std_threshold).into()),
        f("tail_mean_threshold", "末尾判定の mean 閾値", "これより小さい平均を無音とみなす", Float, f32_value(d.tail_mean_threshold).into()),
        f("watermark", "透かし(SilentCipher)", "生成音声に聞こえない透かしを入れる(使えるときのみ)", Bool, d.watermark.into()),
    ]
}

/// f32 の既定値を JSON へ(0.05f32 → 0.05。f64 へ広げたときの端数を見せない)
fn f32_value(x: f32) -> f64 {
    (f64::from(x) * 1e6).round() / 1e6
}

// ---------------------------------------------------------------- Irodori

/// 実エンジン
pub struct IrodoriTts {
    model_id: String,
    tts: irodori::pipeline::Tts,
    num_steps: Option<usize>,
    /// 参照音声 → 符号化済み潜在のキャッシュ(声は繰り返し使われる。符号化は数百 ms〜かかる)
    ref_cache: Mutex<HashMap<RefKey, irodori::Tensor<3>>>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct RefKey {
    path: PathBuf,
    len: u64,
    mtime_ns: u128,
    max_ref_seconds_bits: Option<u64>,
    normalize_db_bits: Option<u32>,
    ensure_max: bool,
}

/// モデルエイリアス → HF リポジトリ。Rust 版が扱えるのは MeanFlow の v4.1 Small のみ。
pub const MODEL_ALIASES: [(&str, &str); 1] = [("v4.1-small-mf", "Aratako/Irodori-TTS-v4.1-Small-MF")];

const REF_CACHE_MAX: usize = 8;

impl IrodoriTts {
    pub fn load(model_id: &str, num_steps: Option<usize>, progress: &dyn Fn(&str)) -> Result<Self> {
        if !MODEL_ALIASES.iter().any(|(alias, _)| *alias == model_id) {
            bail!(
                "未対応の TTS モデル: {model_id}(Rust 版 Irodori が扱えるのは MeanFlow の {} のみ)",
                MODEL_ALIASES.iter().map(|(a, _)| *a).collect::<Vec<_>>().join(", ")
            );
        }
        progress("TTS モデル取得中");
        let paths = irodori::pipeline::TtsPaths::ensure_downloaded(progress)?;
        progress("TTS モデル構築中(初回は GPU のカーネル準備に時間がかかります)");
        let _gpu_load = crate::util::gpu_load_guard(); // 重い GPU ロードは直列化する
        let tts = irodori::pipeline::Tts::load(&paths, &crate::engines::gpu_device()?)?;
        progress(&format!("ロード完了: {model_id}"));
        Ok(Self { model_id: model_id.to_string(), tts, num_steps, ref_cache: Mutex::new(HashMap::new()) })
    }

    fn ref_latent(&self, path: &Path, req: &irodori::pipeline::SamplingRequest, messages: &mut Vec<String>) -> Result<irodori::Tensor<3>> {
        let meta = std::fs::metadata(path).with_context(|| format!("参照音声を開けません: {}", path.display()))?;
        let mtime_ns = meta.modified().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        let key = RefKey {
            path: path.to_path_buf(),
            len: meta.len(),
            mtime_ns,
            max_ref_seconds_bits: req.max_ref_seconds.map(f64::to_bits),
            normalize_db_bits: req.ref_normalize_db.map(f32::to_bits),
            ensure_max: req.ref_ensure_max,
        };
        if let Some(l) = lock(&self.ref_cache).get(&key) {
            return Ok(l.clone());
        }
        let latent = self.tts.encode_reference(path, req, messages)?;
        let mut cache = lock(&self.ref_cache);
        if cache.len() >= REF_CACHE_MAX {
            cache.clear();
        }
        cache.insert(key, latent.clone());
        Ok(latent)
    }
}

/// `tts.sampling` の項目を Irodori の `SamplingRequest` へ反映する。知らない項目はエラーにする。
pub fn apply_sampling(req: &mut irodori::pipeline::SamplingRequest, sampling: &Map<String, Value>) -> Result<()> {
    check_sampling_overrides(sampling)?;
    let f64_of = |k: &str, v: &Value| v.as_f64().ok_or_else(|| anyhow!("tts.sampling.{k} は数値で指定してください: {v}"));
    let usize_of = |k: &str, v: &Value| v.as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("tts.sampling.{k} は正の整数で指定してください: {v}"));
    let bool_of = |k: &str, v: &Value| v.as_bool().ok_or_else(|| anyhow!("tts.sampling.{k} は true / false で指定してください: {v}"));
    for (k, v) in sampling {
        match k.as_str() {
            "num_steps" if !v.is_null() => req.num_steps = usize_of(k, v)?,
            "duration_scale" => req.duration_scale = f64_of(k, v)?,
            "seconds" => req.seconds = if v.is_null() { None } else { Some(f64_of(k, v)?) },
            "min_seconds" => req.min_seconds = f64_of(k, v)?,
            "max_seconds" => req.max_seconds = f64_of(k, v)?,
            "max_ref_seconds" => req.max_ref_seconds = if v.is_null() { None } else { Some(f64_of(k, v)?) },
            "ref_normalize_db" => req.ref_normalize_db = if v.is_null() { None } else { Some(f64_of(k, v)? as f32) },
            "ref_ensure_max" => req.ref_ensure_max = bool_of(k, v)?,
            "trim_tail" => req.trim_tail = bool_of(k, v)?,
            "tail_window_size" => req.tail_window_size = usize_of(k, v)?,
            "tail_std_threshold" => req.tail_std_threshold = f64_of(k, v)? as f32,
            "tail_mean_threshold" => req.tail_mean_threshold = f64_of(k, v)? as f32,
            "watermark" => req.watermark = bool_of(k, v)?,
            "num_steps" => {}
            other => bail!(
                "tts.sampling の項目 {other:?} は Rust 版 Irodori が対応していません(MeanFlow で意味のある項目: {})",
                sampling_fields().iter().map(|f| f.key).collect::<Vec<_>>().join(", ")
            ),
        }
    }
    Ok(())
}

impl TtsEngine for IrodoriTts {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn warmup(&self) -> Result<()> {
        self.tts.warmup()
    }

    fn synthesize(&self, req: &TtsRequest) -> Result<TtsOutput> {
        let t0 = Instant::now();
        let mut sr = irodori::pipeline::SamplingRequest {
            text: req.text.to_string(),
            caption: req.caption.map(str::to_string),
            seed: req.seed,
            ..Default::default()
        };
        if let Some(n) = self.num_steps {
            sr.num_steps = n;
        }
        apply_sampling(&mut sr, req.sampling)?;
        let mut stages: BTreeMap<String, f64> = BTreeMap::new();
        let mut messages = Vec::new();
        // Irodori は参照音声を 1 本の話者条件として使う(複数指定のときは先頭)
        match req.ref_wavs.first() {
            Some(path) => {
                let r0 = Instant::now();
                sr.ref_latent = Some(self.ref_latent(Path::new(path), &sr, &mut messages)?);
                sr.ref_wav = Some(PathBuf::from(path));
                stages.insert("ref_latent_cache".into(), round1(r0.elapsed().as_secs_f64() * 1000.0));
            }
            None => sr.no_ref = true,
        }
        let out = self.tts.synthesize(&sr)?;
        let gen_ms = t0.elapsed().as_millis() as u64;
        for (name, sec) in &out.timings {
            stages.insert(name.clone(), round1(sec * 1000.0));
        }
        let wav = wav_bytes(&out.audio, out.sample_rate)?;
        let duration_ms = (out.audio.len() as f64 / f64::from(out.sample_rate) * 1000.0) as u64;
        messages.extend(out.messages);
        Ok(TtsOutput { wav, sample_rate: out.sample_rate, duration_ms, gen_ms, used_seed: Some(out.used_seed as i64), stages: Some(stages), messages })
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// モノラル f32 → 16bit PCM WAV
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut buf = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut w = hound::WavWriter::new(&mut buf, spec)?;
        for &s in samples {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
    }
    Ok(buf.into_inner())
}

// ---------------------------------------------------------------- モック

pub const MOCK_SAMPLE_RATE: u32 = 48000;

/// テキスト長に比例した長さのビープを返す TTS 代替。
pub struct MockTts {
    model_id: String,
    delay: Duration,
    rtf: f64,
}

impl MockTts {
    /// `delay_ms`: 固定の合成時間、`rtf`: 音声長に比例する合成時間(ベンチで実機相当を模倣)
    pub fn new(model_id: &str, delay_ms: f64, rtf: f64) -> Self {
        Self { model_id: model_id.to_string(), delay: Duration::from_secs_f64(delay_ms.max(0.0) / 1000.0), rtf }
    }
}

/// 短いビープ(フェード付き)の WAV。`seed` があれば音程を変える。
pub fn beep_wav(duration_s: f64, seed: Option<u64>) -> Vec<u8> {
    let freq = seed.map_or(660.0, |s| 440.0 + (s % 7) as f32 * 60.0);
    let n = ((f64::from(MOCK_SAMPLE_RATE) * duration_s) as usize).max(1);
    let fade = (MOCK_SAMPLE_RATE / 40) as usize; // 25ms
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let mut amp = 0.35f32;
            if i < fade {
                amp *= i as f32 / fade as f32;
            } else if n - i < fade {
                amp *= (n - i) as f32 / fade as f32;
            }
            amp * (2.0 * PI * freq * i as f32 / MOCK_SAMPLE_RATE as f32).sin()
        })
        .collect();
    wav_bytes(&samples, MOCK_SAMPLE_RATE).unwrap_or_default()
}

impl TtsEngine for MockTts {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn synthesize(&self, req: &TtsRequest) -> Result<TtsOutput> {
        let t0 = Instant::now();
        // 読了時間風: 文字数×90ms + 400ms、上限8秒
        let duration = (0.4 + 0.09 * req.text.chars().count() as f64).min(8.0);
        let wav = beep_wav(duration, req.seed);
        let target = self.delay + Duration::from_secs_f64(self.rtf * duration);
        if let Some(rest) = target.checked_sub(t0.elapsed()) {
            std::thread::sleep(rest);
        }
        Ok(TtsOutput {
            wav,
            sample_rate: MOCK_SAMPLE_RATE,
            duration_ms: (duration * 1000.0) as u64,
            gen_ms: t0.elapsed().as_millis() as u64,
            used_seed: Some(req.seed.unwrap_or(0) as i64),
            stages: None,
            messages: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn sampling_rejects_app_owned_keys() {
        for key in RESERVED_SAMPLING_KEYS {
            let err = check_sampling_overrides(&map(json!({ key: 1 }))).unwrap_err().to_string();
            assert!(err.contains(key), "{key}: {err}");
        }
        assert!(check_sampling_overrides(&Map::new()).is_ok());
    }

    #[test]
    fn sampling_passes_irodori_options() {
        let mut req = irodori::pipeline::SamplingRequest::default();
        apply_sampling(
            &mut req,
            &map(json!({"duration_scale": 1.1, "num_steps": 8, "seconds": 3.0, "trim_tail": false, "max_ref_seconds": null})),
        )
        .unwrap();
        assert_eq!((req.duration_scale, req.num_steps, req.seconds, req.trim_tail, req.max_ref_seconds), (1.1, 8, Some(3.0), false, None));
    }

    #[test]
    fn unknown_sampling_key_is_an_error_not_silently_dropped() {
        let mut req = irodori::pipeline::SamplingRequest::default();
        let err = apply_sampling(&mut req, &map(json!({"cfg_scale_text": 2.0}))).unwrap_err().to_string();
        assert!(err.contains("cfg_scale_text"), "{err}");
        let err = apply_sampling(&mut req, &map(json!({"duration_scale": "fast"}))).unwrap_err().to_string();
        assert!(err.contains("duration_scale"), "{err}");
    }

    /// GUI の編集欄はこの一覧から作る。全項目が apply_sampling に通り、既定値を入れても何も変わらないこと
    #[test]
    fn sampling_fields_match_apply_sampling() {
        let fields = sampling_fields();
        let defaults: Map<String, Value> = fields.iter().map(|f| (f.key.to_string(), f.default.clone())).collect();
        let mut req = irodori::pipeline::SamplingRequest::default();
        apply_sampling(&mut req, &defaults).unwrap();
        let d = irodori::pipeline::SamplingRequest::default();
        assert_eq!(
            (req.num_steps, req.duration_scale, req.seconds, req.max_ref_seconds, req.ref_normalize_db, req.watermark),
            (d.num_steps, d.duration_scale, d.seconds, d.max_ref_seconds, d.ref_normalize_db, d.watermark)
        );
        assert_eq!((req.tail_std_threshold, req.tail_mean_threshold), (d.tail_std_threshold, d.tail_mean_threshold));
        for f in &fields {
            let value = match f.kind {
                SamplingKind::Int => json!(7),
                SamplingKind::Float => json!(1.5),
                SamplingKind::Bool => json!(!f.default.as_bool().unwrap()),
            };
            apply_sampling(&mut req, &map(json!({ f.key: value }))).unwrap_or_else(|e| panic!("{}: {e}", f.key));
            if f.null_label.is_some() {
                apply_sampling(&mut req, &map(json!({ f.key: null }))).unwrap_or_else(|e| panic!("{}: {e}", f.key));
            }
            assert!(!RESERVED_SAMPLING_KEYS.contains(&f.key), "{}", f.key);
        }
        assert_eq!(req.ref_normalize_db, None);
    }

    #[test]
    fn mock_tts_rtf_simulation() {
        let tts = MockTts::new("m", 0.0, 0.1);
        let text = "あ".repeat(20); // 0.4 + 1.8 = 2.2 秒 → 合成 ≈ 220ms
        let r = tts.synthesize(&TtsRequest { text: &text, caption: None, ref_wavs: &[], seed: None, sampling: &Map::new() }).unwrap();
        assert!((180..400).contains(&r.gen_ms), "{}", r.gen_ms);
        assert!(r.sample_rate > 0 && !r.wav.is_empty());
    }

    #[test]
    fn wav_roundtrip() {
        let wav = wav_bytes(&[0.0, 0.5, -0.5, 1.5], 16000).unwrap();
        let r = hound::WavReader::new(Cursor::new(wav)).unwrap();
        assert_eq!((r.spec().sample_rate, r.spec().channels, r.len()), (16000, 1, 4));
    }
}
