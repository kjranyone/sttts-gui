"""マイク取り込み(sounddevice / PortAudio WASAPI 共有モード)。

デバイス既定レート(通常48kHz)で開いて soxr で 16kHz モノラルへリサンプルする。
PortAudio の排他モードや 16k 直開きは Windows で失敗しやすいため避ける(既知Issue対策)。
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd
import soxr

log = logging.getLogger("sttts.mic")

TARGET_RATE = 16000


class MicStream:
    def __init__(self, device_index: int | None, on_block) -> None:
        """on_block: callable(np.ndarray[float32, 16kHz mono]) — PortAudio コールバックスレッドから呼ばれる。"""
        self.device_index = device_index
        self.on_block = on_block
        self._stream: sd.InputStream | None = None
        self._resampler: soxr.ResampleStream | None = None
        self._lock = threading.Lock()

    def start(self) -> int:
        info = sd.query_devices(self.device_index) if self.device_index is not None else sd.query_devices(kind="input")
        rate = int(info["default_samplerate"])
        channels = min(1, int(info["max_input_channels"]) or 1)
        self._resampler = soxr.ResampleStream(rate, TARGET_RATE, channels, dtype="float32")
        self._stream = sd.InputStream(
            samplerate=rate,
            channels=channels,
            dtype="float32",
            blocksize=int(rate * 0.03),  # 30ms
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        log.info("mic started: %s @ %dHz", info["name"], rate)
        return rate

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("mic status: %s", status)
        block = indata[:, 0] if indata.ndim > 1 else indata
        try:
            out = self._resampler.resample_chunk(block)
        except Exception:
            log.exception("resample failed")
            return
        if out.size:
            self.on_block(out.astype(np.float32))

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    log.exception("mic stop failed")
                self._stream = None
