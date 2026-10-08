"""モックエンジン群(--mock)。重い依存に一切頼らず、GUI/配線の開発を可能にする。

- MockTts: テキスト長に比例した長さのビープWAVを生成
- MockSession: 台本に従って asr_partial / asr_final / mic_level を発生
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

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def load(self, progress=None) -> None:
        time.sleep(0.3)  # ロードを模倣

    def synthesize(self, text, *, caption=None, ref_wavs=None, seed=None, progress=None) -> SynthResult:
        t0 = time.perf_counter()
        # 読了時間風: 文字数×90ms + 400ms、上限8秒
        duration = min(0.4 + 0.09 * len(text), 8.0)
        wav = beep_wav_bytes(duration, seed=seed)
        time.sleep(0.05)  # 合成を模倣
        return SynthResult(
            wav_bytes=wav,
            sample_rate=SAMPLE_RATE,
            duration_ms=int(duration * 1000),
            gen_ms=int((time.perf_counter() - t0) * 1000),
            used_seed=seed if seed is not None else 0,
        )

    def unload(self) -> None:
        pass


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
            self.app.on_asr_final(utt, partials[-1])
            utt += 1
            time.sleep(1.2)

        # 以降はループせず待機(停止指示を待つ)
        while not self._stop.is_set():
            time.sleep(0.2)
