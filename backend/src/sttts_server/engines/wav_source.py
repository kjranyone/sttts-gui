"""WAV ファイルをマイクの代わりに実時間ペースで流す音声ソース(ベンチ・再現テスト用)。

MicStream と同じインターフェース(start / stop、on_block に 16kHz mono float32 を渡す)。
各ファイルの後ろに gap_s 秒の無音を足して VAD の発話終了を確実に発生させる。
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("sttts.wav")

TARGET_RATE = 16000
BLOCK_SECONDS = 0.03  # MicStream と同じ 30ms


def load_wav_16k(path: str) -> np.ndarray:
    import soundfile as sf  # noqa: PLC0415

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != TARGET_RATE:
        import soxr  # noqa: PLC0415

        audio = soxr.resample(audio, sr, TARGET_RATE).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


class WavSource:
    def __init__(
        self,
        paths: list[str],
        on_block,
        *,
        gap_s: float = 2.0,
        lead_s: float = 0.5,
        realtime: bool = True,
        on_eof=None,
    ) -> None:
        self.paths = list(paths)
        self.on_block = on_block
        self.gap_s = gap_s
        self.lead_s = lead_s
        self.realtime = realtime
        self.on_eof = on_eof
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        clips = [load_wav_16k(p) for p in self.paths]
        self._thread = threading.Thread(target=self._run, args=(clips,), name="wav-source", daemon=True)
        self._thread.start()
        log.info("wav source started: %d files", len(clips))
        return TARGET_RATE

    def _run(self, clips: list[np.ndarray]) -> None:
        block = int(TARGET_RATE * BLOCK_SECONDS)
        parts = [np.zeros(int(TARGET_RATE * self.lead_s), dtype=np.float32)]
        for clip in clips:
            parts.append(clip)
            parts.append(np.zeros(int(TARGET_RATE * self.gap_s), dtype=np.float32))
        stream = np.concatenate(parts)
        t0 = time.monotonic()
        for i, start in enumerate(range(0, len(stream), block)):
            if self._stop.is_set():
                return
            if self.realtime:
                # ブロック末尾の時刻まで待つ(マイクと同じく「録り終わった瞬間」に届く)
                due = t0 + (start + block) / TARGET_RATE
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self.on_block(stream[start : start + block])
        if self.on_eof is not None and not self._stop.is_set():
            self.on_eof()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
