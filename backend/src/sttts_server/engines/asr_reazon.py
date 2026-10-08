"""ReazonSpeech(k2 zipformer, sherpa-onnx)による日本語 ASR(任意エンジン)。

asr.engine = "reazonspeech" で選択する。`uv sync --extra reazonspeech` が必要。

トレードオフ(kotoba-whisper との比較、CPU 実測は README 参照):
- 速い: 数秒の発話で 0.1〜0.2 秒程度(kotoba int8 CPU は 3 秒前後)。
- 句読点を出力しない → TTS チャンク分割は長さベースのフォールバックに頼る。
- 固有名詞・英字略語に弱い(例: 「NLP」→「エネルギー」)。
- int8 量子化版は短い発話で崩れやすいため既定は fp32(asr.reazon_precision)。
モデル: reazon-research/reazonspeech-k2-v2(Apache-2.0)。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("sttts.asr")

DEFAULT_REPO = "reazon-research/reazonspeech-k2-v2"
_BASENAMES = ("encoder-epoch-99-avg-1", "decoder-epoch-99-avg-1", "joiner-epoch-99-avg-1")


def model_files(model_dir: str | os.PathLike, precision: str = "fp32") -> dict[str, str]:
    """モデルディレクトリ内の encoder/decoder/joiner/tokens のパスを返す。"""
    d = Path(model_dir)
    suffix = ".int8.onnx" if precision == "int8" else ".onnx"
    enc, dec, join = (str(d / f"{b}{suffix}") for b in _BASENAMES)
    return {"encoder": enc, "decoder": dec, "joiner": join, "tokens": str(d / "tokens.txt")}


class ReazonSpeechAsr:
    engine_name = "reazonspeech"

    def __init__(
        self,
        model_id: str = DEFAULT_REPO,
        model_dir: str | None = None,
        precision: str = "fp32",
        num_threads: int = 4,
        provider: str = "cpu",
    ) -> None:
        self.model_id = model_id or DEFAULT_REPO
        self.model_dir = model_dir
        self.precision = "int8" if str(precision).lower() == "int8" else "fp32"
        self.num_threads = max(1, int(num_threads or 4))
        self.provider = provider or "cpu"
        self.device = self.provider
        self.compute_type = self.precision
        self._rec = None

    def _resolve_dir(self, progress=None) -> str:
        if self.model_dir:
            return self.model_dir
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        if progress is not None:
            progress(f"ASRモデル取得中: {self.model_id} ({self.precision})")
        suffix = "*.int8.onnx" if self.precision == "int8" else "*-avg-1.onnx"
        return snapshot_download(self.model_id, allow_patterns=[suffix, "tokens.txt"])

    def load(self, progress=None) -> None:
        try:
            import sherpa_onnx  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - 環境依存
            raise RuntimeError(
                "ReazonSpeech エンジンには sherpa-onnx が必要です: uv sync --extra reazonspeech(+ torch extra)"
            ) from e
        files = model_files(self._resolve_dir(progress), self.precision)
        for key, path in files.items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"ReazonSpeech {key} が見つかりません: {path}")
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            tokens=files["tokens"],
            num_threads=self.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=self.provider,
        )
        log.info("ASR ready: ReazonSpeech %s (%s, %d threads)", self.model_id, self.precision, self.num_threads)
        if progress is not None:
            progress(f"ASR準備完了: ReazonSpeech ({self.precision})")

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._rec is None:
            raise RuntimeError("ASR not loaded")
        stream = self._rec.create_stream()
        stream.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
        self._rec.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_utterance(self, audio: np.ndarray) -> str:
        return self._transcribe(audio)

    def transcribe_partial(self, audio: np.ndarray) -> str:
        return self._transcribe(audio)
