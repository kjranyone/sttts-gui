"""silero-vad (ONNX) による発話区間検出。512サンプル(=32ms@16kHz)単位で処理する。"""

from __future__ import annotations

import numpy as np

FRAME = 512  # 16kHz 固定(silero-vad の標準ストリーミング単位)


class SileroVad:
    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 280) -> None:
        from silero_vad import VADIterator, load_silero_vad  # noqa: PLC0415

        model = load_silero_vad(onnx=True)
        self._iter = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=16000,
            min_silence_duration_ms=min_silence_ms,
        )

    def reset(self) -> None:
        # silero-vad の VADIterator は reset_states()(reset() は存在しない)。
        # 以前は reset() を呼んでいたため、最初の発話終了で VAD スレッドが例外終了していた。
        self._iter.reset_states()

    def process(self, chunk: np.ndarray) -> dict | None:
        """512サンプルを与えると発話開始/終了イベント({'start':n} / {'end':n})を返す。"""
        out = self._iter(chunk)
        return out if out else None
