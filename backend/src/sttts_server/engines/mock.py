"""モックエンジン群(--mock)。重い依存に一切頼らず、GUI/配線の開発を可能にする。

- MockTts: テキスト長に比例した長さのビープWAVを生成
- MockSession: 台本に従って asr_partial / asr_final / mic_level を発生
- MockAsr: 実 VAD + 実音声入力と組み合わせるための ASR 代替(ベンチ用、asr.engine="mock")
"""

from __future__ import annotations

import io
import math
import struct
import threading
import time
import wave

from ..app import SynthResult

SAMPLE_RATE = 48000


def beep_wav_bytes(duration_s: float, freq: float = 660.0, seed: int | None = None) -> bytes:
    """短いビープ(フェード付き)を WAV バイト列として生成する。"""
    if seed is not None:
        freq = 440.0 + (seed % 7) * 60.0
    n = max(1, int(SAMPLE_RATE * duration_s))
    fade = SAMPLE_RATE // 40  # 25ms
    frames = bytearray()
    for i in range(n):
        t = i / SAMPLE_RATE
        amp = 0.35
        if i < fade:
            amp *= i / fade
        elif n - i < fade:
            amp *= (n - i) / fade
        v = amp * math.sin(2 * math.pi * freq * t)
        frames += struct.pack("<h", int(v * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class MockTts:
    """stdlib のみで動く TTS エンジン代替。"""

    def __init__(self, model_id: str, *, delay_ms: float = 50.0, rtf: float = 0.0, **_ignored) -> None:
        """delay_ms: 固定の合成時間、rtf: 音声長に比例する合成時間(ベンチで実機相当を模倣)。"""
        self.model_id = model_id
        self.delay_ms = float(delay_ms)
        self.rtf = float(rtf)

    def load(self, progress=None) -> None:
        time.sleep(0.3)  # ロードを模倣

    def synthesize(self, text, *, caption=None, ref_wavs=None, seed=None, progress=None) -> SynthResult:
        t0 = time.perf_counter()
        # 読了時間風: 文字数×90ms + 400ms、上限8秒
        duration = min(0.4 + 0.09 * len(text), 8.0)
        wav = beep_wav_bytes(duration, seed=seed)
        # 合成を模倣(ビープ生成時間を差し引いて delay + rtf*duration に合わせる)
        target = (self.delay_ms / 1000.0) + self.rtf * duration
        remaining = target - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
        return SynthResult(
            wav_bytes=wav,
            sample_rate=SAMPLE_RATE,
            duration_ms=int(duration * 1000),
            gen_ms=int((time.perf_counter() - t0) * 1000),
            used_seed=seed if seed is not None else 0,
        )

    def unload(self) -> None:
        pass


class MockAsr:
    """固定テキストを返す ASR エンジン代替。latency_ms で推論時間を模倣する。

    確定(transcribe_utterance)ごとに texts を順に返す。partial は次に確定する
    テキストの先頭を音声長に比例して返す(1秒あたり約6文字)。
    """

    engine_name = "mock"
    DEFAULT_TEXTS = [
        "こんにちは、今日はいい天気ですね。",
        "音声合成のレイテンシを測定しています。",
        "これは三つ目の発話です。チャンク分割を確認します。",
    ]

    def __init__(
        self,
        latency_ms: int = 0,
        texts: list[str] | None = None,
        load_delay_ms: int = 0,
    ) -> None:
        self.model_id = "mock-asr"
        self.device = "cpu"
        self.compute_type = "mock"
        self.latency_ms = max(0, int(latency_ms))
        self.load_delay_ms = max(0, int(load_delay_ms))
        self.texts = list(texts) if texts else list(self.DEFAULT_TEXTS)
        self._finals = 0

    def load(self, progress=None) -> None:
        if self.load_delay_ms:
            time.sleep(self.load_delay_ms / 1000.0)

    def _sleep(self) -> None:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)

    def _current(self) -> str:
        return self.texts[self._finals % len(self.texts)]

    def transcribe_utterance(self, audio) -> str:
        self._sleep()
        text = self._current()
        self._finals += 1
        return text

    def transcribe_partial(self, audio) -> str:
        self._sleep()
        text = self._current()
        n = max(1, min(len(text), int(len(audio) / 16000 * 6)))
        return text[:n]


class MockSession:
    """台本どおりに partial → final を発生させる疑似 ASR セッション。"""

    SCRIPT = [
        ["モックの", "モックの文字起こし", "モックの文字起こしテストです"],
        ["次の文です", "次の文です。これはストリーミング表示の確認です"],
        ["三つ目", "三つ目の発話です。チャンク分割と", "三つ目の発話です。チャンク分割と自動発話を確認しています"],
    ]

    def __init__(self, app) -> None:
        self.app = app
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mock-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        # モデル準備完了を通知
        time.sleep(0.2)
        if self._stop.is_set():
            return
        self.app.on_asr_model_ready("mock-asr")

        utt = 1
        for partials in self.SCRIPT:
            if self._stop.is_set():
                return
            for text in partials:
                if self._stop.is_set():
                    return
                # マイクレベル風の値も流す
                self.app.on_mic_level(rms=0.05, db=-26.0)
                self.app.on_asr_partial(utt, text)
                time.sleep(0.45)
            # 計測フィールドも模擬(話し終わり 280ms 後に VAD が確定、ASR は即時)
            now = time.monotonic()
            self.app.on_asr_final(
                utt,
                partials[-1],
                timing={"speech_end": now - 0.28, "vad_end": now, "asr_ms": 0, "audio_ms": 1500},
            )
            utt += 1
            time.sleep(1.2)

        # 以降も疑似レベルを流し続ける(mock は実マイクを持たないが、
        # レベルメータの UI 経路を常時確認できるようにする)
        t0 = time.monotonic()
        while not self._stop.is_set():
            time.sleep(0.45)
            if self._stop.is_set():
                break
            db = -30.0 + 9.0 * math.sin((time.monotonic() - t0) * 1.7)
            self.app.on_mic_level(rms=10.0 ** (db / 20.0), db=db)
