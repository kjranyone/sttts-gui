//! 発話音声の観測値と Irodori の発話単位アノテーション。
//!
//! 転写内容は ASR が決める。ここでは元音声から得た話し方だけを扱い、
//! 選択中の参照音声を元音声で置き換えない。
//!
//! 感情分類(emotion2vec)は Python 専用モデルのため移植していない。`plan_delivery` の
//! `emotion` 引数は、将来 Rust 製の分類器を足すための入り口として残す。

use crate::chunker::count_mora;

/// Irodori-TTS-v4.1-Small/EMOJI_ANNOTATIONS.md のうち、発話全体に安全に指示できるもの。
/// 非言語音(咳・悲鳴など)は分類だけから自動挿入しない。
pub const EMOTION_STYLE: &[(&str, &str, &str)] = &[
    ("happy", "😊", "楽しげに"),
    ("sad", "😭", "悲しげに"),
    ("angry", "😠", "不満げに"),
    ("fearful", "😰", "緊張した話し方で"),
    ("surprised", "😲", "驚いて"),
];

fn emotion_style(emotion: &str) -> Option<(&'static str, &'static str)> {
    EMOTION_STYLE.iter().find(|(name, _, _)| *name == emotion).map(|(_, emoji, style)| (*emoji, *style))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticObservation {
    pub audio_ms: u64,
    pub active_ms: u64,
    pub pause_ms: u64,
    pub rms: f64,
    pub mora_per_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Delivery {
    pub emoji: String,
    pub style: Option<String>,
    pub duration_scale: Option<f64>,
    pub emotion: Option<String>,
    pub source: String,
}

impl Default for Delivery {
    fn default() -> Self {
        Self { emoji: String::new(), style: None, duration_scale: None, emotion: None, source: "acoustic".into() }
    }
}

impl Delivery {
    /// GUI から来た表現指定を検証して取り込む(許可外の絵文字は捨て、倍率は 0.85〜1.15 に丸める)。
    pub fn from_info(info: &sttts_protocol::DeliveryInfo) -> Self {
        let allowed = |e: &str| EMOTION_STYLE.iter().any(|(_, emoji, _)| *emoji == e) || e == "⏩" || e == "🐢";
        let emoji = info.emoji.clone().filter(|e| allowed(e)).unwrap_or_default();
        let scale = info.duration_scale.filter(|s| s.is_finite()).map(|s| s.clamp(0.85, 1.15));
        Self {
            emoji,
            style: info.style.as_ref().filter(|s| !s.is_empty()).map(|s| s.chars().take(100).collect()),
            duration_scale: scale,
            emotion: info.emotion.clone().filter(|e| !e.is_empty()),
            source: if info.source.is_empty() { "manual".into() } else { info.source.chars().take(30).collect() },
        }
    }

    pub fn annotated_text(&self, text: &str) -> String {
        if self.emoji.is_empty() || text.trim_start().starts_with(&self.emoji) {
            text.to_string()
        } else {
            format!("{}{}", self.emoji, text)
        }
    }

    pub fn caption(&self, voice_caption: Option<&str>) -> Option<String> {
        compose_caption(voice_caption, self.style.as_deref())
    }

    pub fn summary(&self) -> sttts_protocol::DeliveryInfo {
        sttts_protocol::DeliveryInfo {
            emoji: (!self.emoji.is_empty()).then(|| self.emoji.clone()),
            style: self.style.clone(),
            duration_scale: self.duration_scale,
            emotion: self.emotion.clone(),
            source: self.source.clone(),
        }
    }
}

/// 声の caption(声質)に発話ごとの話し方を重ねた Irodori の caption。
/// 自動発話(表現計画)と `sttts-say` の `style` で同じ書き方にする。
pub fn compose_caption(voice_caption: Option<&str>, style: Option<&str>) -> Option<String> {
    let voice_caption = voice_caption.filter(|c| !c.is_empty());
    let Some(style) = style.filter(|s| !s.is_empty()) else {
        return voice_caption.map(str::to_string);
    };
    match voice_caption {
        Some(c) => Some(format!("{c}。話し方は{style}。")),
        None => Some(format!("話し方は{style}。")),
    }
}

/// numpy.percentile(線形補間)相当
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (sorted.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// 40ms ごとの RMS で活動時間と長い間を測る。VAD とは独立した軽量計算。
pub fn observe(audio: &[f32], text: &str, sample_rate: u32) -> AcousticObservation {
    let audio_ms = (audio.len() as f64 * 1000.0 / sample_rate as f64).round() as u64;
    if audio.is_empty() {
        return AcousticObservation { audio_ms: 0, active_ms: 0, pause_ms: 0, rms: 0.0, mora_per_s: None };
    }
    let frame = ((sample_rate as f64 * 0.04).round() as usize).max(1);
    let levels: Vec<f64> = audio
        .chunks(frame)
        .map(|c| {
            // 末尾はゼロ詰めしてフレーム長で平均する(numpy の pad + mean と同じ)
            let sum: f64 = c.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            (sum / frame as f64).sqrt()
        })
        .collect();
    let rms = (audio.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / audio.len() as f64).sqrt();
    let mut sorted = levels.clone();
    sorted.sort_by(f64::total_cmp);
    // マイクごとの音量差を緩和する。絶対下限で環境ノイズを活動音扱いしない。
    let threshold = 0.004f64.max(0.035f64.min(percentile(&sorted, 80.0) * 0.28));
    let active: Vec<bool> = levels.iter().map(|&l| l >= threshold).collect();
    let active_ms = audio_ms.min(active.iter().filter(|&&a| a).count() as u64 * 40);
    // 120ms 未満の小さな切れ目は無音として数えない。
    let mut pause_ms = 0u64;
    let mut run = 0u64;
    for &is_active in &active {
        if is_active {
            if run >= 3 {
                pause_ms += run * 40;
            }
            run = 0;
        } else {
            run += 1;
        }
    }
    let mora = count_mora(text);
    let rate = (mora >= 4.0 && active_ms >= 400).then(|| mora / (active_ms as f64 / 1000.0));
    AcousticObservation { audio_ms, active_ms, pause_ms, rms, mora_per_s: rate }
}

/// モデル分類と話者内の相対速度を控えめな Irodori 指示へ写す。
pub fn plan_delivery(obs: &AcousticObservation, emotion: Option<&str>, baseline_mora_per_s: Option<f64>) -> Delivery {
    let emotion = emotion.unwrap_or("").to_lowercase();
    let known = emotion_style(&emotion);
    let (mut emoji, mut style) = match known {
        Some((e, s)) => (e.to_string(), Some(s.to_string())),
        None => (String::new(), None),
    };
    if obs.pause_ms >= 400 && obs.pause_ms as f64 >= obs.active_ms as f64 * 0.2 {
        style = Some(match style {
            Some(s) => format!("{s}、間を取りながら"),
            None => "間を取りながら".to_string(),
        });
    }
    let mut scale = None;
    if let (Some(rate), Some(base)) = (obs.mora_per_s, baseline_mora_per_s)
        && rate > 0.0
        && base > 0.0
        && obs.active_ms >= 800
    {
        let ratio = rate / base;
        if ratio >= 1.35 {
            scale = Some(0.90);
            if emoji.is_empty() {
                emoji = "⏩".into();
            }
        } else if ratio <= 0.74 {
            scale = Some(1.10);
            if emoji.is_empty() {
                emoji = "🐢".into();
            }
        }
    }
    Delivery {
        emoji,
        style,
        duration_scale: scale,
        emotion: known.map(|_| emotion.clone()),
        source: if known.is_some() { "ser".into() } else { "acoustic".into() },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acoustic_observation_and_conservative_delivery() {
        let mut audio = vec![0.08f32; 16000];
        audio.extend(vec![0.0; 4000]);
        audio.extend(vec![0.08f32; 16000]);
        let obs = observe(&audio, "おはようございます", 16000);
        assert_eq!(obs.audio_ms, 2250);
        assert!(obs.pause_ms >= 200);
        assert!(obs.active_ms >= 1900);
        assert_eq!(plan_delivery(&obs, None, None).emoji, "");
        assert_eq!(plan_delivery(&obs, Some("neutral"), None).emoji, "");
        assert_eq!(plan_delivery(&obs, Some("happy"), None).emoji, "😊");
        let slow = AcousticObservation { audio_ms: 3500, active_ms: 2600, pause_ms: 700, rms: 0.08, mora_per_s: None };
        assert_eq!(plan_delivery(&slow, None, None).style.as_deref(), Some("間を取りながら"));
        assert_eq!(plan_delivery(&slow, Some("happy"), None).style.as_deref(), Some("楽しげに、間を取りながら"));
    }

    #[test]
    fn delivery_preserves_text_and_voice_caption() {
        let d = Delivery { emoji: "😠".into(), style: Some("不満げに".into()), duration_scale: Some(0.9), ..Default::default() };
        assert_eq!(d.annotated_text("やめてください"), "😠やめてください");
        assert_eq!(d.annotated_text("😠やめてください"), "😠やめてください");
        assert_eq!(d.caption(Some("落ち着いた声")).as_deref(), Some("落ち着いた声。話し方は不満げに。"));
        let info = |emoji: Option<&str>, scale: Option<f64>| sttts_protocol::DeliveryInfo {
            emoji: emoji.map(str::to_string),
            style: None,
            duration_scale: scale,
            emotion: None,
            source: String::new(),
        };
        assert_eq!(Delivery::from_info(&info(Some("🤐"), Some(9.0))).emoji, "");
        assert_eq!(Delivery::from_info(&info(None, Some(9.0))).duration_scale, Some(1.15));
        assert_eq!(Delivery::from_info(&info(None, Some(f64::NAN))).duration_scale, None);
    }

    #[test]
    fn relative_speed_picks_emoji_and_scale() {
        let obs = AcousticObservation { audio_ms: 3000, active_ms: 2500, pause_ms: 0, rms: 0.1, mora_per_s: Some(12.0) };
        let d = plan_delivery(&obs, None, Some(8.0));
        assert_eq!((d.emoji.as_str(), d.duration_scale), ("⏩", Some(0.90)));
        let d = plan_delivery(&obs, None, Some(20.0));
        assert_eq!((d.emoji.as_str(), d.duration_scale), ("🐢", Some(1.10)));
    }
}
