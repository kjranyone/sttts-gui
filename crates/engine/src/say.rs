//! ヘッドレス合成(`sttts-say` の本体)。エージェントや動画制作のワークフローから、
//! セリフを「キャラクター(声)」と「話し方」で WAV にする。マイクは開かない。
//!
//! キャラクターを固定するための 3 つの単位:
//!
//! - **声**: `data/voices/<name>.{wav,flac}`(参照音声。GUI の声バンクと同じ)と
//!   `data/voices/<name>.json`([`VoiceFile`]: 声質の caption・seed・sampling)。どちらか片方だけでもよい。
//! - **テイク**: 出力 WAV と隣の `<stem>.json`([`Take`])。再現に要る指定と実際に使った seed を全部持つ。
//!   テイクから同じ指定で撮り直せ(`--like`)、声として登録できる([`Studio::save_voice`])。
//!   登録はテイクの音声を参照音声に昇格させる。参照音声なしの声は seed と caption だけで決まり、
//!   文が変わると声質が揺れるので、気に入ったテイクを参照音声にするのが最も強い固定になる。
//! - **台本**: JSONL(1 行 = 1 テイク、[`Line`])。指定が前回のテイクと同じ行は合成し直さない(増分レンダリング)。
//!   直した行だけが撮り直しになり、seed を決めていない行も気に入ったテイクが保たれる。
//!
//! 合成の指定は GUI と同じ規則で決まる: `tts.sampling`(backend.json)の上に声・行の sampling を重ね、
//! 予約キー・未知のキーはエラー。caption は声質に話し方を重ねる([`compose_caption`])。

use std::cell::OnceCell;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sttts_i18n::trf;

use crate::chunker::{ChunkOptions, split_chunks};
use crate::config::{get, get_i64};
use crate::performance::compose_caption;
use crate::tts::{TtsEngine, TtsRequest, apply_sampling};

const VOICE_AUDIO_EXT: &[&str] = &["wav", "flac"];

/// 声の設定(`data/voices/<name>.json`)。参照音声は同名の wav/flac。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VoiceFile {
    /// 声質の指示(Irodori の caption)。行の `style` はこの後ろに重なる
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    /// 固定 seed。参照音声なしの声ではこれと caption が声質を決める
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// この声だけの sampling(`tts.sampling` の上に重なる。項目名は Irodori と同じ)
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub sampling: Map<String, Value>,
    /// 登録元のテイク(記録のみ)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Voice {
    pub name: String,
    pub ref_wav: Option<PathBuf>,
    #[serde(flatten)]
    pub file: VoiceFile,
}

/// 1 テイクの指定(台本の 1 行、`speak` の引数)。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Line {
    /// 台本での出力名(`<id>.wav`)。省略時は行番号(001, 002, ...)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub text: String,
    /// 声(`data/voices` の名前)。省略時は backend.json の `voice`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// この行の話し方(例: 「囁くように、少し照れて」)。声質の caption の後ろに重なる
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// 声質の caption を置き換える
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    /// 声の seed より優先
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// 声の sampling の上に重なる(duration_scale 等)
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub sampling: Map<String, Value>,
}

/// 解決済みの合成指定。同じ Spec なら同じテイクとみなす(増分レンダリングの比較キー)。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Spec {
    pub text: String,
    pub voice: Option<String>,
    /// 話し方を重ねる前の声質(テイクを声として登録するときに使う)
    pub voice_caption: Option<String>,
    /// Irodori へ渡す caption
    pub caption: Option<String>,
    pub ref_wav: Option<String>,
    /// 参照音声の中身のハッシュ(同名の声を差し替えたら撮り直す)
    pub ref_hash: Option<String>,
    /// 決めてある seed(None = 合成時にランダム)
    pub seed: Option<i64>,
    /// 声と行の sampling(`tts.sampling` を除く。声として登録するときに引き継ぐ)
    pub voice_sampling: Map<String, Value>,
    /// 実際に使う sampling(`tts.sampling` + 声 + 行)
    pub sampling: Map<String, Value>,
    pub model: String,
    pub num_steps: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// テイクの記録(`<stem>.json`)。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Take {
    /// 指定されたとおりの行(`--like` で撮り直す元)
    pub line: Line,
    pub spec: Spec,
    /// 実際に使った seed(全チャンク共通)
    pub seed: i64,
    pub sample_rate: u32,
    pub duration_ms: u64,
    /// チャンクごとの区間(字幕・口パクのタイミングに使える)
    pub segments: Vec<Segment>,
}

pub fn voices_dir(root: &Path) -> PathBuf {
    root.join("data").join("voices")
}

/// テイクの記録ファイル(WAV の隣、拡張子 .json)
pub fn take_record_path(wav: &Path) -> PathBuf {
    wav.with_extension("json")
}

/// ファイル名に使う名前(声・台本の id)の検証。パスの区切りや Windows で使えない文字を拒否する。
fn check_name(kind: &str, name: &str) -> Result<()> {
    let bad = name.trim().is_empty()
        || name != name.trim()
        || name.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control())
        || name.starts_with('.');
    if bad {
        bail!(
            "{}",
            trf!(
                "Invalid {kind} name: {name:?} (it becomes a file name; avoid / \\ : * ? \" < > | and a leading dot)",
                "{kind} の名前が不正です: {name:?}(ファイル名になります。/ \\ : * ? \" < > | と先頭のドットは使えません)",
                "{kind} 名称无效:{name:?}(将作为文件名;不能使用 / \\ : * ? \" < > | 和开头的点)"
            )
        );
    }
    Ok(())
}

fn voice_audio(dir: &Path, name: &str) -> Option<PathBuf> {
    VOICE_AUDIO_EXT.iter().map(|e| dir.join(format!("{name}.{e}"))).find(|p| p.is_file())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    serde_json::from_str(&text).with_context(|| path.display().to_string())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, text + "\n").with_context(|| path.display().to_string())
}

/// 書き出しは一時ファイル → 置き換え(途中で止まっても壊れたテイクを残さない)
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| tmp.display().to_string())?;
    std::fs::rename(&tmp, path).with_context(|| path.display().to_string())
}

pub fn load_voice(root: &Path, name: &str) -> Result<Voice> {
    check_name("voice", name)?;
    let dir = voices_dir(root);
    let ref_wav = voice_audio(&dir, name);
    let json_path = dir.join(format!("{name}.json"));
    let file = if json_path.is_file() { read_json(&json_path)? } else { VoiceFile::default() };
    if ref_wav.is_none() && !json_path.is_file() {
        let known = list_voices(root)?.into_iter().map(|v| v.name).collect::<Vec<_>>().join(", ");
        bail!(
            "{}",
            trf!(
                "Unknown voice: {name} (available: {known})",
                "声が見つかりません: {name}(登録済み: {known})",
                "找不到声音:{name}(已登记:{known})"
            )
        );
    }
    Ok(Voice { name: name.to_string(), ref_wav, file })
}

/// 登録済みの声(参照音声か設定ファイルのある名前)を名前順に。
pub fn list_voices(root: &Path) -> Result<Vec<Voice>> {
    let mut names: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(voices_dir(root)) {
        for e in entries.flatten() {
            let p = e.path();
            let ext = p.extension().and_then(|x| x.to_str()).map(str::to_ascii_lowercase);
            let is_voice = ext.is_some_and(|x| x == "json" || VOICE_AUDIO_EXT.contains(&x.as_str()));
            if let Some(stem) = p.file_stem().and_then(|s| s.to_str()).filter(|_| is_voice && p.is_file()) {
                names.push(stem.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    names.iter().filter(|n| check_name("voice", n).is_ok()).map(|n| load_voice(root, n)).collect()
}

/// 参照音声の中身のハッシュ(FNV-1a 64bit。Rust のバージョンで変わらない)
fn file_hash(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| {
        let path = path.display();
        trf!("Cannot open the reference audio: {path}", "参照音声を開けません: {path}", "无法打开参考音频:{path}")
    })?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    Ok(format!("{h:016x}"))
}

fn random_seed() -> i64 {
    i64::from(rand::rng().random::<u32>() >> 1)
}

/// 行を合成指定へ解決する(GPU に触れずに、指定の誤りをここで全部見つける)。
pub fn resolve(cfg: &Value, root: &Path, line: &Line) -> Result<Spec> {
    let text = line.text.trim().to_string();
    if text.is_empty() {
        bail!("{}", trf!("The text is empty", "セリフが空です", "台词为空"));
    }
    let voice = match &line.voice {
        Some(name) => load_voice(root, name)?,
        // 声を指定しないときは backend.json の `voice`(GUI の自動発話の既定と同じ)
        None => Voice {
            name: String::new(),
            ref_wav: get(cfg, "voice", "ref_wavs").as_array().and_then(|a| a.first()).and_then(Value::as_str).map(PathBuf::from),
            file: VoiceFile {
                caption: get(cfg, "voice", "caption").as_str().map(str::to_string),
                seed: get(cfg, "voice", "seed").as_i64(),
                ..Default::default()
            },
        },
    };
    let voice_caption = line.caption.clone().or(voice.file.caption.clone()).filter(|c| !c.trim().is_empty());
    let caption = compose_caption(voice_caption.as_deref(), line.style.as_deref());
    let ref_hash = voice.ref_wav.as_deref().map(file_hash).transpose()?;

    let mut voice_sampling = voice.file.sampling.clone();
    voice_sampling.extend(line.sampling.clone());
    let mut sampling = get(cfg, "tts", "sampling").as_object().cloned().unwrap_or_default();
    sampling.extend(voice_sampling.clone());
    // 予約キー・未知のキー・型の誤りを合成前に検出する
    apply_sampling(&mut irodori::pipeline::SamplingRequest::default(), &sampling)?;

    Ok(Spec {
        text,
        voice: line.voice.clone(),
        voice_caption,
        caption,
        ref_wav: voice.ref_wav.map(|p| p.to_string_lossy().into_owned()),
        ref_hash,
        seed: line.seed.or(voice.file.seed),
        voice_sampling,
        sampling,
        model: get(cfg, "tts", "model").as_str().unwrap_or("v4.1-small-mf").to_string(),
        num_steps: get(cfg, "tts", "num_steps").clone(),
    })
}

/// 台本(JSONL)を読む。空行は飛ばす。id の省略は行番号、重複はエラー。
pub fn parse_script(text: &str) -> Result<Vec<(String, Line)>> {
    let mut out: Vec<(String, Line)> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let raw = raw.trim().trim_start_matches('\u{feff}');
        if raw.is_empty() {
            continue;
        }
        let n = i + 1;
        let line: Line = serde_json::from_str(raw).with_context(|| trf!("script line {n}", "台本の {n} 行目", "脚本第 {n} 行"))?;
        let id = line.id.clone().unwrap_or_else(|| format!("{n:03}"));
        check_name("id", &id)?;
        if out.iter().any(|(other, _)| other.eq_ignore_ascii_case(&id)) {
            bail!("{}", trf!("Duplicate id in the script: {id}", "台本の id が重複しています: {id}", "脚本中的 id 重复:{id}"));
        }
        out.push((id, line));
    }
    if out.is_empty() {
        bail!("{}", trf!("The script has no lines", "台本に行がありません", "脚本中没有台词"));
    }
    Ok(out)
}

/// テイクの記録を読む(`take` は WAV か .json のどちらでもよい)
pub fn read_take(take: &Path) -> Result<Take> {
    read_json(&take_record_path(take))
}

/// テイクと同じ指定の行(seed は実際に使ったものに固定)。`--like` の土台。
pub fn line_like(take: &Take) -> Line {
    Line { id: None, seed: Some(take.seed), ..take.line.clone() }
}

fn summary(wav: &Path, take: &Take, status: &str) -> Value {
    json!({
        "out": wav,
        "record": take_record_path(wav),
        "status": status,
        "voice": take.spec.voice,
        "caption": take.spec.caption,
        "seed": take.seed,
        "duration_ms": take.duration_ms,
        "segments": take.segments,
    })
}

/// 合成の作業場。TTS は最初に合成が要るときに 1 回だけロードする(全行が最新なら GPU に触れない)。
pub struct Studio {
    root: PathBuf,
    cfg: Value,
    loader: Box<dyn Fn() -> Result<Arc<dyn TtsEngine>>>,
    engine: OnceCell<Arc<dyn TtsEngine>>,
    log: Box<dyn Fn(&str)>,
}

impl Studio {
    pub fn new(root: PathBuf, cfg: Value, loader: impl Fn() -> Result<Arc<dyn TtsEngine>> + 'static, log: impl Fn(&str) + 'static) -> Self {
        Self { root, cfg, loader: Box::new(loader), engine: OnceCell::new(), log: Box::new(log) }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn engine(&self) -> Result<Arc<dyn TtsEngine>> {
        if let Some(e) = self.engine.get() {
            return Ok(Arc::clone(e));
        }
        let e = (self.loader)()?;
        let _ = self.engine.set(Arc::clone(&e));
        Ok(e)
    }

    /// テイクのチャンク分割。ファイル出力では初音を急がないので、先頭の短縮はせず
    /// `pipeline.chunk_max_chars` 程度まで文を束ねる(長すぎる文だけ分ける)。
    fn chunks(&self, text: &str) -> Vec<String> {
        let max = get_i64(&self.cfg, "pipeline", "chunk_max_chars", 80).max(0) as usize;
        if max == 0 {
            return vec![text.to_string()];
        }
        let opt = ChunkOptions { min_chars: max, first_min_chars: max, max_chars: max, first_mora_min: 0.0, first_mora_max: 0.0 };
        split_chunks(text, &opt)
    }

    /// 1 テイクを合成して `wav` へ書く(記録 `<stem>.json` も)。
    pub fn take(&self, line: &Line, spec: &Spec, wav: &Path) -> Result<Take> {
        let engine = self.engine()?;
        // 未指定ならここで 1 回だけ決めて全チャンクで共有する(チャンクごとに声質が変わらないように)
        let seed = spec.seed.unwrap_or_else(random_seed);
        let ref_wavs: Vec<String> = spec.ref_wav.iter().cloned().collect();
        let mut samples: Vec<i16> = Vec::new();
        let mut sample_rate = 0u32;
        let mut segments = Vec::new();
        for chunk in self.chunks(&spec.text) {
            let out = engine.synthesize(&TtsRequest {
                text: &chunk,
                caption: spec.caption.as_deref(),
                ref_wavs: &ref_wavs,
                seed: Some(seed as u64),
                sampling: &spec.sampling,
            })?;
            for m in &out.messages {
                (self.log)(m);
            }
            let mut reader = hound::WavReader::new(Cursor::new(out.wav))?;
            let sr = reader.spec().sample_rate;
            if sample_rate != 0 && sr != sample_rate {
                bail!("sample rate changed between chunks: {sample_rate} -> {sr}");
            }
            sample_rate = sr;
            let start = samples.len();
            for s in reader.samples::<i16>() {
                samples.push(s?);
            }
            let ms = |n: usize| (n as u64 * 1000) / u64::from(sr.max(1));
            segments.push(Segment { text: chunk, start_ms: ms(start), end_ms: ms(samples.len()) });
        }
        let mut buf = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
        {
            let wspec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::new(&mut buf, wspec)?;
            for s in &samples {
                w.write_sample(*s)?;
            }
            w.finalize()?;
        }
        if let Some(dir) = wav.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
        }
        let take = Take {
            line: line.clone(),
            spec: spec.clone(),
            seed,
            sample_rate,
            duration_ms: segments.last().map_or(0, |s| s.end_ms),
            segments,
        };
        // 記録を先に消し、音声 → 記録の順に書く(途中で止まったら「最新でない」扱いになる)
        let _ = std::fs::remove_file(take_record_path(wav));
        write_atomic(wav, &buf.into_inner())?;
        write_json(&take_record_path(wav), &take)?;
        Ok(take)
    }

    /// 1 行を合成する。`out` 省略時は `output/say/say_<時刻>_<seed>.wav`。
    pub fn speak(&self, line: &Line, out: Option<&Path>) -> Result<Value> {
        let spec = resolve(&self.cfg, &self.root, line)?;
        let wav = match out {
            Some(p) => p.to_path_buf(),
            None => {
                let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                let dir = self.root.join("output").join("say");
                (0u32..).map(|n| dir.join(format!("say_{secs}_{n:02}.wav"))).find(|p| !p.exists()).unwrap_or_default()
            }
        };
        let take = self.take(line, &spec, &wav)?;
        Ok(summary(&wav, &take, "rendered"))
    }

    /// 台本を `out_dir/<id>.wav` へ合成する。指定が前回と同じ行は合成し直さない。
    /// 結果は標準出力用の値として返し、`out_dir/manifest.json` にも残す。
    pub fn render(&self, script: &Path, out_dir: Option<&Path>) -> Result<Value> {
        let text = std::fs::read_to_string(script).with_context(|| script.display().to_string())?;
        let lines = parse_script(&text)?;
        // 指定の誤りは合成を始める前に全行ぶん見つける
        let specs = lines
            .iter()
            .map(|(id, line)| resolve(&self.cfg, &self.root, line).with_context(|| format!("id {id}")))
            .collect::<Result<Vec<_>>>()?;
        let out_dir = match out_dir {
            Some(d) => d.to_path_buf(),
            None => {
                let stem = script.file_stem().and_then(|s| s.to_str()).unwrap_or("script");
                script.parent().unwrap_or(Path::new(".")).join(stem)
            }
        };
        let total = lines.len();
        let mut results = Vec::new();
        for (i, ((id, line), spec)) in lines.iter().zip(&specs).enumerate() {
            let wav = out_dir.join(format!("{id}.wav"));
            let previous = wav.is_file().then(|| read_take(&wav).ok()).flatten().filter(|t| t.spec == *spec);
            let n = i + 1;
            let (take, status) = match previous {
                Some(t) => (t, "unchanged"),
                None => {
                    (self.log)(&trf!("[{n}/{total}] {id}: synthesizing", "[{n}/{total}] {id}: 合成中", "[{n}/{total}] {id}:正在合成"));
                    (self.take(line, spec, &wav).with_context(|| format!("id {id}"))?, "rendered")
                }
            };
            let mut s = summary(&wav, &take, status);
            s["id"] = json!(id);
            s["text"] = json!(take.spec.text);
            results.push(s);
        }
        let manifest = json!({ "script": script, "out_dir": out_dir, "lines": results });
        write_json(&out_dir.join("manifest.json"), &manifest)?;
        Ok(manifest)
    }

    /// seed だけを変えた候補を `count` 本作る(気に入った候補を `save_voice` で声にする)。
    pub fn audition(&self, line: &Line, count: usize, out_dir: Option<&Path>) -> Result<Value> {
        if line.seed.is_some() {
            bail!(
                "{}",
                trf!(
                    "audition tries random seeds; do not pass a seed",
                    "audition は seed を変えて候補を作ります。seed は指定しないでください",
                    "audition 会尝试随机 seed,请不要指定 seed"
                )
            );
        }
        let mut spec = resolve(&self.cfg, &self.root, line)?;
        let out_dir = match out_dir {
            Some(d) => d.to_path_buf(),
            None => {
                let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                self.root.join("output").join("say").join(format!("audition_{secs}"))
            }
        };
        let mut takes = Vec::new();
        let mut used: Vec<i64> = Vec::new();
        for i in 0..count.max(1) {
            let seed = std::iter::repeat_with(random_seed).find(|s| !used.contains(s)).unwrap_or_default();
            used.push(seed);
            spec.seed = Some(seed);
            let n = i + 1;
            (self.log)(&trf!("[{n}/{count}] seed {seed}", "[{n}/{count}] seed {seed}", "[{n}/{count}] seed {seed}"));
            let wav = out_dir.join(format!("take_{n:02}_seed{seed}.wav"));
            let line = Line { seed: Some(seed), ..line.clone() };
            let take = self.take(&line, &spec, &wav)?;
            takes.push(summary(&wav, &take, "rendered"));
        }
        Ok(json!({ "out_dir": out_dir, "takes": takes }))
    }

    /// テイクを声 `name` として登録する。テイクの音声を参照音声にし、声質の caption・seed・sampling を引き継ぐ。
    /// 既にある声は上書きしない(キャラクターを黙って変えない)。
    pub fn save_voice(&self, name: &str, take_path: &Path) -> Result<Value> {
        check_name("voice", name)?;
        let take = read_take(take_path)?;
        let wav = take_path.with_extension("wav");
        if !wav.is_file() {
            let path = wav.display();
            bail!("{}", trf!("Take audio not found: {path}", "テイクの音声が見つかりません: {path}", "找不到录音音频:{path}"));
        }
        let dir = voices_dir(&self.root);
        if voice_audio(&dir, name).is_some() || dir.join(format!("{name}.json")).exists() {
            bail!(
                "{}",
                trf!(
                    "Voice already exists: {name} (choose another name, or delete it first)",
                    "同名の声が既にあります: {name}(別の名前にするか、先に削除してください)",
                    "已存在同名声音:{name}(请换个名称,或先删除它)"
                )
            );
        }
        std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
        let file = VoiceFile {
            caption: take.spec.voice_caption.clone(),
            seed: Some(take.seed),
            sampling: take.spec.voice_sampling.clone(),
            source: Some(wav.to_string_lossy().into_owned()),
        };
        let ref_wav = dir.join(format!("{name}.wav"));
        std::fs::copy(&wav, &ref_wav).with_context(|| ref_wav.display().to_string())?;
        write_json(&dir.join(format!("{name}.json")), &file)?;
        Ok(serde_json::to_value(Voice { name: name.to_string(), ref_wav: Some(ref_wav), file })?)
    }
}

#[cfg(test)]
mod tests;
