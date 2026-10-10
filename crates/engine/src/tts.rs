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
use sttts_i18n::{tr, trf};

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
        bail!(
            "{}",
            trf!(
                "Keys not allowed in tts.sampling (the app sets them for each utterance): {reserved:?}",
                "tts.sampling に指定できないキー(発話ごとにアプリが決定): {reserved:?}",
                "tts.sampling 中不能指定的键(由应用按每次发话决定):{reserved:?}"
            )
        );
    }
    Ok(())
}

/// `tts.sampling` の値の型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingKind {
    Int,
    Float,
    Bool,
    /// 決まった文字列のどれか
    Choice(&'static [&'static str]),
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

/// `tts.sampling` で指定できる全項目と Irodori の既定値。項目名・説明は今の表示言語。
pub fn sampling_fields() -> Vec<SamplingField> {
    let d = irodori::pipeline::SamplingRequest::default();
    let f = |key, label, help, kind, default: Value| SamplingField { key, label, help, kind, default, null_label: None };
    use SamplingKind::*;
    vec![
        f(
            "num_steps",
            tr!("Steps", "ステップ数", "步数"),
            tr!(
                "More is more careful but slower (empty = the model's default: 4 for MeanFlow, 40 for RF)",
                "多いほど丁寧だが遅い(空 = モデルの既定: MeanFlow は 4、RF は 40)",
                "越多越精细但越慢(空 = 模型默认:MeanFlow 为 4,RF 为 40)"
            ),
            Int,
            d.num_steps.into(),
        ),
        f(
            "duration_scale",
            tr!("Length scale", "長さの倍率", "时长倍率"),
            tr!("Above 1 reads slower, below 1 faster", "1 より大きいとゆっくり、小さいと速く読む", "大于 1 读得慢,小于 1 读得快"),
            Float,
            d.duration_scale.into(),
        ),
        f(
            "seconds",
            tr!("Length (s, fixed)", "長さ(秒、固定)", "时长(秒,固定)"),
            tr!(
                "If set, synthesizes at this length instead of predicting it",
                "指定すると長さ予測を使わずこの長さで合成する",
                "指定后不使用时长预测,按此时长合成"
            ),
            Float,
            d.seconds.into(),
        ),
        f(
            "min_seconds",
            tr!("Min (s)", "最短(秒)", "最短(秒)"),
            tr!("Lower bound of the predicted length", "予測した長さの下限", "预测时长的下限"),
            Float,
            d.min_seconds.into(),
        ),
        f(
            "max_seconds",
            tr!("Max (s)", "最長(秒)", "最长(秒)"),
            tr!("Upper bound of the predicted length", "予測した長さの上限", "预测时长的上限"),
            Float,
            d.max_seconds.into(),
        ),
        f(
            "max_ref_seconds",
            tr!("Reference audio limit (s)", "参照音声の上限(秒)", "参考音频上限(秒)"),
            tr!(
                "Longer reference audio is cut to this length (empty = no cut)",
                "長い参照音声はこの長さで切る(空 = 切らない)",
                "过长的参考音频会截到此长度(空 = 不截断)"
            ),
            Float,
            d.max_ref_seconds.into(),
        ),
        SamplingField {
            null_label: Some(tr!("Don't normalize", "正規化しない", "不归一化")),
            ..f(
                "ref_normalize_db",
                tr!("Reference loudness (LUFS)", "参照音声の音量(LUFS)", "参考音频响度(LUFS)"),
                tr!(
                    "Normalizes the reference audio to this loudness before use",
                    "参照音声をこの音量にそろえてから使う",
                    "使用前将参考音频统一到此响度"
                ),
                Float,
                d.ref_normalize_db.map(f32_value).into(),
            )
        },
        f(
            "ref_ensure_max",
            tr!("Reference clipping guard", "参照音声の音割れ防止", "参考音频防削波"),
            tr!(
                "When not normalizing, scales down if the peak exceeds 1.0",
                "正規化しないとき、ピークが 1.0 を超えたら縮める",
                "不归一化时,若峰值超过 1.0 则缩小"
            ),
            Bool,
            d.ref_ensure_max.into(),
        ),
        f(
            "trim_tail",
            tr!("Trim trailing silence", "末尾の無音を削る", "裁剪末尾静音"),
            tr!(
                "Cuts silence and noise at the end of the generated audio",
                "生成音声の末尾にある無音・ノイズを切る",
                "裁掉生成音频末尾的静音和噪声"
            ),
            Bool,
            d.trim_tail.into(),
        ),
        f(
            "tail_window_size",
            tr!("Tail detection window", "末尾判定の窓", "末尾判定窗口"),
            tr!("Latent frames used to detect the tail", "末尾判定に使う潜在フレーム数", "用于末尾判定的潜在帧数"),
            Int,
            d.tail_window_size.into(),
        ),
        f(
            "tail_std_threshold",
            tr!("Tail std threshold", "末尾判定の std 閾値", "末尾判定 std 阈值"),
            tr!("Variation below this counts as silence", "これより小さい揺れを無音とみなす", "小于此值的波动视为静音"),
            Float,
            f32_value(d.tail_std_threshold).into(),
        ),
        f(
            "tail_mean_threshold",
            tr!("Tail mean threshold", "末尾判定の mean 閾値", "末尾判定 mean 阈值"),
            tr!("A mean below this counts as silence", "これより小さい平均を無音とみなす", "小于此值的均值视为静音"),
            Float,
            f32_value(d.tail_mean_threshold).into(),
        ),
        f(
            "watermark",
            tr!("Watermark (SilentCipher)", "透かし(SilentCipher)", "水印(SilentCipher)"),
            tr!(
                "Embeds an inaudible watermark in the generated audio (when available)",
                "生成音声に聞こえない透かしを入れる(使えるときのみ)",
                "在生成音频中嵌入听不见的水印(仅在可用时)"
            ),
            Bool,
            d.watermark.into(),
        ),
        f(
            "cfg_scale_text",
            tr!("Text guidance (RF)", "テキストの CFG(RF)", "文本 CFG(RF)"),
            tr!(
                "How strongly to follow the text. RF models only; MeanFlow ignores it",
                "テキストにどれだけ強く従うか。RF のモデルのみ(MeanFlow は無視)",
                "遵循文本的强度。仅 RF 模型(MeanFlow 忽略)"
            ),
            Float,
            d.cfg_scale_text.into(),
        ),
        f(
            "cfg_scale_caption",
            tr!("Caption guidance (RF)", "キャプションの CFG(RF)", "描述 CFG(RF)"),
            tr!(
                "How strongly to follow the caption (voice / style). RF models only",
                "キャプション(声・話し方)にどれだけ強く従うか。RF のモデルのみ",
                "遵循描述(声音、说话方式)的强度。仅 RF 模型"
            ),
            Float,
            d.cfg_scale_caption.into(),
        ),
        f(
            "cfg_scale_speaker",
            tr!("Speaker guidance (RF)", "話者の CFG(RF)", "说话人 CFG(RF)"),
            tr!(
                "How strongly to follow the reference audio. RF models only",
                "参照音声にどれだけ強く寄せるか。RF のモデルのみ",
                "贴近参考音频的强度。仅 RF 模型"
            ),
            Float,
            d.cfg_scale_speaker.into(),
        ),
        f(
            "cfg_scale",
            tr!("All guidance (RF)", "CFG 一括(RF)", "CFG 统一(RF)"),
            tr!(
                "If set, uses this value for text, caption and speaker guidance",
                "指定するとテキスト・キャプション・話者の CFG をすべてこの値にする",
                "指定后文本、描述、说话人的 CFG 都使用此值"
            ),
            Float,
            d.cfg_scale.into(),
        ),
        f(
            "cfg_guidance_mode",
            tr!("Guidance mode (RF)", "CFG のかけ方(RF)", "CFG 方式(RF)"),
            tr!(
                "independent: each condition separately (default) / joint: all at once (equal scales) / alternating: one per step",
                "independent: 条件ごと(既定)/ joint: まとめて(倍率をそろえる)/ alternating: 1 ステップに 1 つずつ",
                "independent:逐个条件(默认)/ joint:一起(倍率需相同)/ alternating:每步一个"
            ),
            Choice(&irodori::sampler::CfgGuidanceMode::NAMES),
            d.cfg_guidance_mode.name().into(),
        ),
        f(
            "cfg_min_t",
            tr!("Guidance from t (RF)", "CFG をかける t の下限(RF)", "施加 CFG 的 t 下限(RF)"),
            tr!(
                "Guidance applies while the time t is between this and the upper bound (1 = noise, 0 = audio)",
                "時刻 t がこれと上限の間のステップだけ CFG をかける(1 = ノイズ、0 = 音声)",
                "仅在时刻 t 介于此值与上限之间时施加 CFG(1 = 噪声,0 = 音频)"
            ),
            Float,
            d.cfg_min_t.into(),
        ),
        f(
            "cfg_max_t",
            tr!("Guidance up to t (RF)", "CFG をかける t の上限(RF)", "施加 CFG 的 t 上限(RF)"),
            tr!("Upper bound of the guidance range", "CFG をかける範囲の上限", "施加 CFG 范围的上限"),
            Float,
            d.cfg_max_t.into(),
        ),
        f(
            "truncation_factor",
            tr!("Noise scale (RF)", "初期ノイズの倍率(RF)", "初始噪声倍率(RF)"),
            tr!(
                "Scales the starting noise; below 1 is steadier, less varied",
                "初期ノイズに掛ける倍率。1 より小さいと安定するが単調になる",
                "初始噪声的倍率;小于 1 更稳定但更单调"
            ),
            Float,
            d.truncation_factor.into(),
        ),
        f(
            "rescale_k",
            tr!("Score rescale k (RF)", "スコア補正 k(RF)", "分数校正 k(RF)"),
            tr!(
                "Temporal score rescaling; takes effect when both k and sigma are set",
                "Temporal score rescaling。k と sigma を両方指定したときだけ効く",
                "Temporal score rescaling;同时指定 k 和 sigma 时才生效"
            ),
            Float,
            d.rescale_k.into(),
        ),
        f(
            "rescale_sigma",
            tr!("Score rescale sigma (RF)", "スコア補正 sigma(RF)", "分数校正 sigma(RF)"),
            tr!("See score rescale k", "スコア補正 k を参照", "参见分数校正 k"),
            Float,
            d.rescale_sigma.into(),
        ),
        f(
            "speaker_kv_scale",
            tr!("Speaker emphasis (RF)", "話者の強調(RF)", "说话人强调(RF)"),
            tr!(
                "Multiplies the reference voice's attention keys/values (above 1 = closer to the reference)",
                "参照音声の注意のキー・値に掛ける倍率(1 より大きいと参照音声に寄る)",
                "参考音频注意力键/值的倍率(大于 1 更贴近参考音频)"
            ),
            Float,
            d.speaker_kv_scale.into(),
        ),
        f(
            "speaker_kv_min_t",
            tr!("Emphasis until t (RF)", "強調をやめる t(RF)", "停止强调的 t(RF)"),
            tr!(
                "The emphasis stops once the time t falls below this (empty = 0.9)",
                "時刻 t がこれを下回ったら強調をやめる(空 = 0.9)",
                "时刻 t 低于此值后停止强调(空 = 0.9)"
            ),
            Float,
            d.speaker_kv_min_t.into(),
        ),
        f(
            "speaker_kv_max_layers",
            tr!("Emphasis layers (RF)", "強調する層数(RF)", "强调层数(RF)"),
            tr!(
                "Applies the emphasis to this many first layers (empty = all)",
                "先頭からこの層数だけ強調する(空 = 全層)",
                "只对前这么多层强调(空 = 全部)"
            ),
            Int,
            d.speaker_kv_max_layers.into(),
        ),
        f(
            "speaker_uncond_mode",
            tr!("Speaker-free branch (RF)", "話者なし側の中身(RF)", "无说话人分支(RF)"),
            tr!(
                "What the speaker guidance compares against: mask (no speaker, default) / noise",
                "話者の CFG で比べる相手: mask(話者なし、既定)/ noise(ノイズ)",
                "说话人 CFG 的比较对象:mask(无说话人,默认)/ noise(噪声)"
            ),
            Choice(&irodori::sampler::SpeakerUncondMode::NAMES),
            d.speaker_uncond_mode.name().into(),
        ),
        f(
            "t_schedule_mode",
            tr!("Time schedule (RF)", "時刻の刻み方(RF)", "时间步划分(RF)"),
            tr!(
                "linear (default) / sway: denser steps on the noise side (uses sway_coeff)",
                "linear(既定)/ sway: ノイズ側を細かく刻む(sway_coeff を使う)",
                "linear(默认)/ sway:在噪声侧划分更细(使用 sway_coeff)"
            ),
            Choice(&T_SCHEDULE_MODES),
            (if d.t_schedule_sway { "sway" } else { "linear" }).into(),
        ),
        f(
            "sway_coeff",
            tr!("Sway coefficient (RF)", "sway の係数(RF)", "sway 系数(RF)"),
            tr!(
                "Negative values refine the noise side, positive the audio side",
                "負だとノイズ側、正だと音声側を細かく刻む",
                "负值细化噪声侧,正值细化音频侧"
            ),
            Float,
            d.sway_coeff.into(),
        ),
    ]
}

/// `t_schedule_mode` の値
const T_SCHEDULE_MODES: [&str; 2] = ["linear", "sway"];

/// f32 の既定値を JSON へ(0.05f32 → 0.05。f64 へ広げたときの端数を見せない)
fn f32_value(x: f32) -> f64 {
    (f64::from(x) * 1e6).round() / 1e6
}

// ---------------------------------------------------------------- Irodori

/// 本番は GPU(wgpu)。CPU 推論は実装しない。
/// TTS と kotoba-whisper が使う。GPU を初期化する前にプロセス間ロックを取る(GUI と sttts-say が同時に GPU を初期化しない)。
pub fn gpu_device() -> Result<irodori::Device> {
    #[cfg(feature = "gpu")]
    {
        crate::util::hold_gpu_process_lock()?;
        irodori::try_gpu_device()
    }
    #[cfg(not(feature = "gpu"))]
    {
        bail!(
            "{}",
            tr!(
                "Built without GPU support (the gpu feature of sttts-engine)",
                "GPU 対応なしでビルドされています(sttts-engine の gpu feature)",
                "构建时未启用 GPU 支持(sttts-engine 的 gpu feature)"
            )
        )
    }
}

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

/// 選べる Irodori のモデル(`tts.model` には `alias` を書く)
#[derive(Debug, Clone, PartialEq)]
pub struct TtsModel {
    pub alias: &'static str,
    pub repo: &'static str,
    /// リポジトリ内の重み(量子化版はサブフォルダ)
    pub weights: &'static str,
    /// 表示名(今の表示言語)
    pub label: &'static str,
    /// MeanFlow(数ステップ)か RF(40 ステップ + CFG)か
    pub meanflow: bool,
    /// モデルの重みのダウンロード量(MB。コーデックと透かしは全モデル共通で別)
    pub download_mb: u32,
    /// 特徴(今の表示言語)
    pub summary: &'static str,
    /// MIT 以外の利用条件(名前, URL)。選ぶときに示す
    pub license: Option<(&'static str, &'static str)>,
}

/// v4 Large は T5Gemma 2 由来のエンコーダを含むので Gemma の利用規約に従う
const GEMMA_TERMS: (&str, &str) = ("Gemma Terms of Use", "https://ai.google.dev/gemma/terms");

/// 既定のモデル
pub const DEFAULT_TTS_MODEL: &str = "v4.1-small-mf";

/// Rust 版 Irodori が扱えるモデルの一覧
pub fn tts_models() -> Vec<TtsModel> {
    vec![
        TtsModel {
            alias: DEFAULT_TTS_MODEL,
            repo: "Aratako/Irodori-TTS-v4.1-Small-MF",
            weights: irodori::pipeline::MODEL_WEIGHTS,
            label: tr!(
                "Irodori v4.1 Small MeanFlow (fast, for conversation)",
                "Irodori v4.1 Small MeanFlow(高速・会話向け)",
                "Irodori v4.1 Small MeanFlow(高速、适合对话)"
            ),
            meanflow: true,
            download_mb: 3093,
            summary: tr!(
                "MeanFlow, 4 steps. Fast enough for live conversation (default)",
                "MeanFlow、4 ステップ。会話に使える速さ(既定)",
                "MeanFlow,4 步。速度足以用于实时对话(默认)"
            ),
            license: None,
        },
        TtsModel {
            alias: "v4.1-small",
            repo: "Aratako/Irodori-TTS-v4.1-Small",
            weights: irodori::pipeline::MODEL_WEIGHTS,
            label: tr!(
                "Irodori v4.1 Small RF (high quality, slow)",
                "Irodori v4.1 Small RF(高品質・低速)",
                "Irodori v4.1 Small RF(高质量、低速)"
            ),
            meanflow: false,
            download_mb: 3064,
            summary: tr!(
                "RF, 40 steps with guidance (CFG). Better kanji reading and voice cloning than MeanFlow, but about 20x the computation: for sttts-say and other offline use",
                "RF、40 ステップ + CFG。漢字の読みと声の再現が MeanFlow より正確だが、計算量は約 20 倍。sttts-say などリアルタイムでない用途向け",
                "RF,40 步 + CFG。汉字读音与声音还原比 MeanFlow 更准确,但计算量约为 20 倍。适合 sttts-say 等非实时用途"
            ),
            license: None,
        },
        TtsModel {
            alias: "v4.1-small-int8",
            repo: "Aratako/Irodori-TTS-v4.1-Small-Quantized",
            weights: "int8-weight-only/model.safetensors",
            label: tr!(
                "Irodori v4.1 Small RF int8 (for GPUs with little memory)",
                "Irodori v4.1 Small RF int8(GPU メモリが少ない環境向け)",
                "Irodori v4.1 Small RF int8(适合显存较少的环境)"
            ),
            meanflow: false,
            download_mb: 914,
            summary: tr!(
                "v4.1 Small (RF) with int8 weights: about a quarter of the GPU memory and download for its main layers, slightly less accurate than the full model",
                "v4.1 Small(RF)の重みを int8 にしたもの。主な層の GPU メモリとダウンロード量が約 1/4 になる代わりに、元のモデルよりわずかに精度が落ちる",
                "将 v4.1 Small(RF)的权重量化为 int8。主要层的显存与下载量约为四分之一,精度略低于原模型"
            ),
            license: None,
        },
        TtsModel {
            alias: "v4-large",
            repo: "Aratako/Irodori-TTS-v4-Large",
            weights: irodori::pipeline::MODEL_WEIGHTS,
            label: tr!("Irodori v4 Large RF (highest quality, needs a large GPU)", "Irodori v4 Large RF(最高品質・大容量の GPU 向け)", "Irodori v4 Large RF(最高质量,需大显存 GPU)"),
            meanflow: false,
            download_mb: 13153,
            summary: tr!(
                "RF, 3.3B parameters. Follows captions (voice design) and long reference audio best; kanji reading is about the same as v4.1 Small. Needs about 16 GB of GPU memory",
                "RF、33 億パラメータ。キャプション(声のデザイン)と長い参照音声への追従が最も良い。漢字の読みは v4.1 Small と同程度。GPU メモリは 16GB 程度必要",
                "RF,33 亿参数。对描述(声音设计)和长参考音频的遵循最好;汉字读音与 v4.1 Small 相当。需要约 16GB 显存"
            ),
            license: Some(GEMMA_TERMS),
        },
        TtsModel {
            alias: "v4-large-int8",
            repo: "Aratako/Irodori-TTS-v4-Large-Quantized",
            weights: "int8-weight-only/model.safetensors",
            label: tr!("Irodori v4 Large RF int8 (high quality, mid-range GPUs)", "Irodori v4 Large RF int8(高品質・中程度の GPU 向け)", "Irodori v4 Large RF int8(高质量,适合中端 GPU)"),
            meanflow: false,
            download_mb: 3840,
            summary: tr!(
                "v4 Large with int8 weights: about a third of the GPU memory and download, slightly less accurate than the full model",
                "v4 Large の重みを int8 にしたもの。GPU メモリとダウンロード量が約 1/3 になる代わりに、元のモデルよりわずかに精度が落ちる",
                "将 v4 Large 的权重量化为 int8。显存与下载量约为三分之一,精度略低于原模型"
            ),
            license: Some(GEMMA_TERMS),
        },
    ]
}

impl TtsModel {
    /// 既定のステップ数(MeanFlow 4、RF 40)
    pub fn default_steps(&self) -> usize {
        if self.meanflow { 4 } else { 40 }
    }
}

/// 別名からモデルを引く。未知ならエラー(選べるものを示す)
pub fn tts_model(alias: &str) -> Result<TtsModel> {
    let models = tts_models();
    if let Some(m) = models.iter().find(|m| m.alias == alias) {
        return Ok(m.clone());
    }
    let known = models.iter().map(|m| m.alias).collect::<Vec<_>>().join(", ");
    bail!(
        "{}",
        trf!(
            "Unsupported TTS model: {alias} (available: {known})",
            "未対応の TTS モデル: {alias}(選べるもの: {known})",
            "不支持的 TTS 模型:{alias}(可选:{known})"
        )
    )
}

const REF_CACHE_MAX: usize = 8;

impl IrodoriTts {
    pub fn load(model_id: &str, num_steps: Option<usize>, progress: &dyn Fn(&str)) -> Result<Self> {
        let model = tts_model(model_id)?;
        if let Some((name, url)) = model.license {
            progress(&trf!(
                "{model_id} is subject to the {name}: {url}",
                "{model_id} は {name} に従って使ってください: {url}",
                "{model_id} 须遵守 {name}:{url}"
            ));
        }
        progress(tr!("Fetching the TTS model", "TTS モデル取得中", "正在获取 TTS 模型"));
        let paths = irodori::pipeline::TtsPaths::ensure_downloaded(model.repo, model.weights, progress)?;
        progress(tr!(
            "Building the TTS model (the first run takes a while to prepare GPU kernels)",
            "TTS モデル構築中(初回は GPU のカーネル準備に時間がかかります)",
            "正在构建 TTS 模型(首次运行需要较长时间准备 GPU 内核)"
        ));
        let _gpu_load = crate::util::gpu_load_guard(); // 重い GPU ロードは直列化する
        let tts = irodori::pipeline::Tts::load(&paths, &gpu_device()?)?;
        progress(&trf!("Loaded: {model_id}", "ロード完了: {model_id}", "加载完成:{model_id}"));
        Ok(Self { model_id: model_id.to_string(), tts, num_steps, ref_cache: Mutex::new(HashMap::new()) })
    }

    fn ref_latent(&self, path: &Path, req: &irodori::pipeline::SamplingRequest, messages: &mut Vec<String>) -> Result<irodori::Tensor<3>> {
        let meta = std::fs::metadata(path).with_context(|| {
            let path = path.display();
            trf!("Cannot open the reference audio: {path}", "参照音声を開けません: {path}", "无法打开参考音频:{path}")
        })?;
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
    let f64_of = |k: &str, v: &Value| {
        v.as_f64().ok_or_else(|| {
            anyhow!(
                "{}",
                trf!(
                    "tts.sampling.{k} must be a number: {v}",
                    "tts.sampling.{k} は数値で指定してください: {v}",
                    "tts.sampling.{k} 必须是数值:{v}"
                )
            )
        })
    };
    let usize_of = |k: &str, v: &Value| {
        v.as_u64().map(|x| x as usize).ok_or_else(|| {
            anyhow!(
                "{}",
                trf!(
                    "tts.sampling.{k} must be a positive integer: {v}",
                    "tts.sampling.{k} は正の整数で指定してください: {v}",
                    "tts.sampling.{k} 必须是正整数:{v}"
                )
            )
        })
    };
    let opt_f64 = |k: &str, v: &Value| if v.is_null() { Ok(None) } else { f64_of(k, v).map(Some) };
    let choice_of = |k: &str, v: &Value, names: &[&str]| {
        v.as_str().filter(|s| names.contains(&s.trim().to_ascii_lowercase().as_str())).map(|s| s.trim().to_ascii_lowercase()).ok_or_else(|| {
            let names = names.join(" / ");
            anyhow!(
                "{}",
                trf!(
                    "tts.sampling.{k} must be one of {names}: {v}",
                    "tts.sampling.{k} は {names} のどれかで指定してください: {v}",
                    "tts.sampling.{k} 必须是 {names} 之一:{v}"
                )
            )
        })
    };
    let bool_of = |k: &str, v: &Value| {
        v.as_bool().ok_or_else(|| {
            anyhow!(
                "{}",
                trf!(
                    "tts.sampling.{k} must be true or false: {v}",
                    "tts.sampling.{k} は true / false で指定してください: {v}",
                    "tts.sampling.{k} 必须是 true 或 false:{v}"
                )
            )
        })
    };
    for (k, v) in sampling {
        match k.as_str() {
            "num_steps" => req.num_steps = if v.is_null() { None } else { Some(usize_of(k, v)?) },
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
            "cfg_scale_text" => req.cfg_scale_text = f64_of(k, v)?,
            "cfg_scale_caption" => req.cfg_scale_caption = f64_of(k, v)?,
            "cfg_scale_speaker" => req.cfg_scale_speaker = f64_of(k, v)?,
            "cfg_scale" => req.cfg_scale = opt_f64(k, v)?,
            "cfg_guidance_mode" => {
                let name = choice_of(k, v, &irodori::sampler::CfgGuidanceMode::NAMES)?;
                req.cfg_guidance_mode = irodori::sampler::CfgGuidanceMode::parse(&name).expect("checked above");
            }
            "cfg_min_t" => req.cfg_min_t = f64_of(k, v)?,
            "cfg_max_t" => req.cfg_max_t = f64_of(k, v)?,
            "truncation_factor" => req.truncation_factor = opt_f64(k, v)?,
            "rescale_k" => req.rescale_k = opt_f64(k, v)?,
            "rescale_sigma" => req.rescale_sigma = opt_f64(k, v)?,
            "speaker_kv_scale" => req.speaker_kv_scale = opt_f64(k, v)?,
            "speaker_kv_min_t" => req.speaker_kv_min_t = opt_f64(k, v)?,
            "speaker_kv_max_layers" => req.speaker_kv_max_layers = if v.is_null() { None } else { Some(usize_of(k, v)?) },
            "speaker_uncond_mode" => {
                let name = choice_of(k, v, &irodori::sampler::SpeakerUncondMode::NAMES)?;
                req.speaker_uncond_mode = irodori::sampler::SpeakerUncondMode::parse(&name).expect("checked above");
            }
            "t_schedule_mode" => req.t_schedule_sway = choice_of(k, v, &T_SCHEDULE_MODES)? == "sway",
            "sway_coeff" => req.sway_coeff = f64_of(k, v)?,
            other => {
                let keys = sampling_fields().iter().map(|f| f.key).collect::<Vec<_>>().join(", ");
                bail!(
                    "{}",
                    trf!(
                        "tts.sampling key {other:?} is not supported by the Rust Irodori (supported keys: {keys})",
                        "tts.sampling の項目 {other:?} は Rust 版 Irodori が対応していません(指定できる項目: {keys})",
                        "Rust 版 Irodori 不支持 tts.sampling 的项 {other:?}(可指定的项:{keys})"
                    )
                )
            }
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
            sr.num_steps = Some(n);
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
        assert_eq!((req.duration_scale, req.num_steps, req.seconds, req.trim_tail, req.max_ref_seconds), (1.1, Some(8), Some(3.0), false, None));
        apply_sampling(
            &mut req,
            &map(json!({"cfg_scale_text": 2.0, "cfg_guidance_mode": "Joint", "t_schedule_mode": "sway", "speaker_kv_max_layers": 4, "num_steps": null})),
        )
        .unwrap();
        assert_eq!(req.cfg_scale_text, 2.0);
        assert_eq!(req.cfg_guidance_mode, irodori::sampler::CfgGuidanceMode::Joint);
        assert!(req.t_schedule_sway);
        assert_eq!((req.speaker_kv_max_layers, req.num_steps), (Some(4), None));
    }

    #[test]
    fn unknown_sampling_key_is_an_error_not_silently_dropped() {
        let mut req = irodori::pipeline::SamplingRequest::default();
        let err = apply_sampling(&mut req, &map(json!({"cfg_scale_txt": 2.0}))).unwrap_err().to_string();
        assert!(err.contains("cfg_scale_txt"), "{err}");
        let err = apply_sampling(&mut req, &map(json!({"cfg_guidance_mode": "both"}))).unwrap_err().to_string();
        assert!(err.contains("independent"), "{err}");
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
                SamplingKind::Choice(names) => json!(names[names.len() - 1]),
            };
            apply_sampling(&mut req, &map(json!({ f.key: value }))).unwrap_or_else(|e| panic!("{}: {e}", f.key));
            if f.null_label.is_some() || f.default.is_null() {
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
