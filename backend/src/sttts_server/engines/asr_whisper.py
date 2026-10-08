"""faster-whisper(CTranslate2)による日本語 ASR。

kotoba-whisper-v2.0-faster(kotoba-tech, MIT/Apache)を想定。
CTranslate2 は Intel XPU 非対応のため CPU で動かす(XPU は TTS 専用)。
"""

from __future__ import annotations

import numpy as np


class StreamingAsr:
    def __init__(
        self,
        model_id: str = "kotoba-tech/kotoba-whisper-v2.0-faster",
        compute_type: str = "int8",
        language: str = "ja",
    ) -> None:
        self.model_id = model_id
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self, progress=None) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        if progress is not None:
            progress(f"ASRモデル取得中: {self.model_id}")
        self._model = WhisperModel(self.model_id, device="cpu", compute_type=self.compute_type)
        if progress is not None:
            progress(f"ASR準備完了: {self.model_id}")

    def _transcribe(self, audio: np.ndarray, beam_size: int) -> str:
        if self._model is None:
            raise RuntimeError("ASR not loaded")
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            condition_on_previous_text=False,
            vad_filter=False,  # VAD は上位の silero が担う
        )
        return "".join(seg.text for seg in segments).strip()

    def transcribe_utterance(self, audio: np.ndarray) -> str:
        """発話全体の高品質デコード(確定テキスト用)。"""
        return self._transcribe(audio, beam_size=2)

    def transcribe_partial(self, audio: np.ndarray) -> str:
        """部分文字起こし(表示用)。速さ優先で beam_size=1。"""
        return self._transcribe(audio, beam_size=1)
