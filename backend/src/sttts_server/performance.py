"""発話音声の観測値と Irodori の発話単位アノテーション。

転写内容は ASR が決める。ここでは元音声から得た話し方だけを扱い、
選択中の参照音声を元音声で置き換えない。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .chunker import count_mora

# Irodori-TTS-v4.1-Small/EMOJI_ANNOTATIONS.md のうち、発話全体に安全に
# 指示できるもの。非言語音(咳・悲鳴など)は分類だけから自動挿入しない。
# https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/blob/main/EMOJI_ANNOTATIONS.md
EMOTION_STYLE: dict[str, tuple[str, str]] = {
    "happy": ("😊", "楽しげに"),
    "sad": ("😭", "悲しげに"),
    "angry": ("😠", "不満げに"),
    "fearful": ("😰", "緊張した話し方で"),
    "surprised": ("😲", "驚いて"),
}


@dataclass(frozen=True)
class AcousticObservation:
    audio_ms: int
    active_ms: int
    pause_ms: int
    rms: float
    mora_per_s: float | None = None


@dataclass(frozen=True)
class Delivery:
    emoji: str = ""
    style: str | None = None
    duration_scale: float | None = None
    emotion: str | None = None
    source: str = "acoustic"

    @classmethod
    def from_mapping(cls, value: dict) -> "Delivery":
        allowed = {v[0] for v in EMOTION_STYLE.values()} | {"⏩", "🐢"}
        emoji = str(value.get("emoji") or "")
        if emoji not in allowed and emoji:
            emoji = ""
        scale = value.get("duration_scale")
        if scale is not None:
            scale = float(scale)
            if not math.isfinite(scale):
                scale = None
        if scale is not None:
            scale = max(0.85, min(1.15, scale))
        style = value.get("style")
        return cls(
            emoji=emoji,
            style=str(style)[:100] if style else None,
            duration_scale=scale,
            emotion=str(value["emotion"]) if value.get("emotion") else None,
            source=str(value.get("source") or "manual")[:30],
        )

    def annotated_text(self, text: str) -> str:
        if not self.emoji or text.lstrip().startswith(self.emoji):
            return text
        return f"{self.emoji}{text}"

    def caption(self, voice_caption: str | None) -> str | None:
        if not self.style:
            return voice_caption
        if voice_caption:
            return f"{voice_caption}。話し方は{self.style}。"
        return f"話し方は{self.style}。"

    def summary(self) -> dict:
        return {
            "emoji": self.emoji or None,
            "style": self.style,
            "duration_scale": self.duration_scale,
            "emotion": self.emotion,
            "source": self.source,
        }


def observe(audio: np.ndarray, text: str = "", sample_rate: int = 16000) -> AcousticObservation:
    """40ms ごとの RMS で活動時間と長い間を測る。VAD とは独立した軽量計算。"""
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio_ms = round(len(signal) * 1000 / sample_rate)
    if len(signal) == 0:
        return AcousticObservation(0, 0, 0, 0.0)
    frame = max(1, round(sample_rate * 0.04))
    padded = np.pad(signal, (0, (-len(signal)) % frame))
    levels = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1))
    rms = float(np.sqrt(np.mean(signal**2)))
    # マイクごとの音量差を緩和する。絶対下限で環境ノイズを活動音扱いしない。
    threshold = max(0.004, min(0.035, float(np.percentile(levels, 80)) * 0.28))
    active = levels >= threshold
    active_ms = min(audio_ms, int(np.count_nonzero(active) * 40))
    # 120ms 未満の小さな切れ目は無音として数えない。
    pause_ms = 0
    run = 0
    for is_active in active:
        if is_active:
            if run >= 3:
                pause_ms += run * 40
            run = 0
        else:
            run += 1
    mora = count_mora(text)
    rate = mora / (active_ms / 1000) if mora >= 4 and active_ms >= 400 else None
    return AcousticObservation(audio_ms, active_ms, pause_ms, rms, rate)


def plan_delivery(
    observation: AcousticObservation,
    *,
    emotion: str | None = None,
    baseline_mora_per_s: float | None = None,
) -> Delivery:
    """モデル分類と話者内の相対速度を控えめな Irodori 指示へ写す。"""
    emotion = (emotion or "").lower()
    emoji, style = EMOTION_STYLE.get(emotion, ("", None))
    if observation.pause_ms >= 400 and observation.pause_ms >= observation.active_ms * 0.2:
        style = f"{style}、間を取りながら" if style else "間を取りながら"
    scale = None
    rate = observation.mora_per_s
    if rate and baseline_mora_per_s and observation.active_ms >= 800:
        ratio = rate / baseline_mora_per_s
        if ratio >= 1.35:
            scale = 0.90
            if not emoji:
                emoji = "⏩"
        elif ratio <= 0.74:
            scale = 1.10
            if not emoji:
                emoji = "🐢"
    return Delivery(
        emoji=emoji,
        style=style,
        duration_scale=scale,
        emotion=emotion if emotion in EMOTION_STYLE else None,
        source="ser" if emotion in EMOTION_STYLE else "acoustic",
    )


class Emotion2VecClassifier:
    """任意インストールの emotion2vec+ を CPU で実行する。

    スコアは確率として表示しない。曖昧な判定は適用しない。
    """

    def __init__(self, model_path: str) -> None:
        from funasr import AutoModel  # noqa: PLC0415

        self._model = AutoModel(
            model=model_path,
            device="cpu",
            disable_update=True,
        )

    def classify(self, audio: np.ndarray) -> str | None:
        results = self._model.generate(
            input=np.asarray(audio, dtype=np.float32).reshape(-1),
            fs=16000,
            granularity="utterance",
            extract_embedding=False,
        )
        if not results:
            return None
        labels = results[0].get("labels")
        scores = results[0].get("scores")
        if labels is None or scores is None:
            return None
        if len(labels) != len(scores) or len(scores) == 0:
            return None
        ranked = sorted(zip(scores, labels), reverse=True)
        top_score, top_label = ranked[0]
        next_score = ranked[1][0] if len(ranked) > 1 else 0.0
        # 配布モデルの tokens.txt は「开心/happy」のような二言語ラベル。
        label = str(top_label).rsplit("/", 1)[-1].lower()
        if float(top_score) < 0.65 or float(top_score) - float(next_score) < 0.2:
            return None
        return label if label in EMOTION_STYLE else None
