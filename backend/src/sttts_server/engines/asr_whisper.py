"""faster-whisper(CTranslate2)による日本語 ASR。

kotoba-whisper-v2.0-faster(kotoba-tech, MIT/Apache)を想定。

デバイス選択(asr.device / asr.compute_type):
- "auto": CTranslate2 が CUDA デバイスを認識できれば cuda/float16、無ければ cpu/int8。
- CTranslate2 は Intel XPU 非対応のため、XPU 環境では CPU で動く(XPU は TTS 専用)。
- CUDA でのロードに失敗した場合(cuDNN/cuBLAS DLL 不足など)は cpu/int8 へフォールバックする。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("sttts.asr")

DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"


def cuda_device_count() -> int:
    """CTranslate2 から見える CUDA デバイス数(取得失敗時は 0)。"""
    try:
        import ctranslate2  # noqa: PLC0415

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # pragma: no cover - 環境依存
        return 0


def resolve_device(device: str | None, compute_type: str | None, *, cuda_count: int | None = None) -> tuple[str, str]:
    """(device, compute_type) を決定する。"auto"/None は環境から推定する。"""
    dev = (device or "auto").lower()
    if dev == "auto":
        count = cuda_device_count() if cuda_count is None else cuda_count
        dev = "cuda" if count > 0 else "cpu"
    ct = (compute_type or "auto").lower()
    if ct == "auto":
        ct = "float16" if dev == "cuda" else "int8"
    return dev, ct


def _add_windows_cuda_dll_dirs() -> None:
    """Windows で torch 同梱の cuDNN/cuBLAS DLL を CTranslate2 から見えるようにする。

    注意: CTranslate2 は CUDA 12 系(cublas64_12.dll)を要求する。cu130 の torch は CUDA 13 の
    DLL を同梱するため、kotoba を CUDA で動かすには別途 CUDA 12 のランタイムが要る(無ければ
    CPU にフォールバックする)。

    CTranslate2 は cudnn64_9.dll / cublas64_12.dll を PATH から探すため、torch/lib と
    nvidia-* wheel の bin を PATH と DLL 検索パスに足す(存在するものだけ)。
    """
    if sys.platform != "win32":
        return
    candidates: list[Path] = []
    try:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            candidates.append(Path(spec.origin).parent / "lib")
    except Exception:
        pass
    site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if site.is_dir():
        candidates.extend(p for p in site.glob("*/bin") if p.is_dir())
    for d in candidates:
        if not d.is_dir():
            continue
        try:
            os.add_dll_directory(str(d))  # type: ignore[attr-defined]
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


class StreamingAsr:
    """kotoba-whisper(faster-whisper)エンジン。ASR エンジン共通インターフェース:

    - load(progress)
    - transcribe_utterance(audio_16k_float32) -> str  (確定用・高品質)
    - transcribe_partial(audio_16k_float32) -> str    (表示用・速度優先)
    - model_id / device / compute_type 属性
    """

    engine_name = "kotoba"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        compute_type: str = "auto",
        language: str = "ja",
        device: str = "auto",
        cpu_threads: int = 0,
        final_beam_size: int = 2,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.device, self.compute_type = "cpu", "int8"
        self.language = language
        self.cpu_threads = int(cpu_threads or 0)
        self.final_beam_size = max(1, int(final_beam_size))
        self._model = None

    def load(self, progress=None) -> None:
        device, compute_type = resolve_device(self.requested_device, self.requested_compute_type)
        if device == "cuda":
            _add_windows_cuda_dll_dirs()
        from faster_whisper import WhisperModel  # noqa: PLC0415

        if progress is not None:
            progress(f"ASRモデル取得中: {self.model_id} ({device}/{compute_type})")
        try:
            self._model = WhisperModel(
                self.model_id, device=device, compute_type=compute_type, cpu_threads=self.cpu_threads
            )
            if device == "cuda":
                # cuDNN/cuBLAS の欠落は最初の推論で初めて表面化するため、ここで1回走らせる
                self._transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
        except Exception as e:
            if device == "cpu":
                raise
            log.warning("ASR %s/%s の初期化に失敗、cpu/int8 へフォールバック: %s", device, compute_type, e)
            if progress is not None:
                progress(f"ASR CUDA 初期化失敗 → CPU int8 へフォールバック: {e}")
            device, compute_type = "cpu", "int8"
            self._model = WhisperModel(
                self.model_id, device=device, compute_type=compute_type, cpu_threads=self.cpu_threads
            )
        self.device, self.compute_type = device, compute_type
        log.info("ASR ready: %s @ %s/%s", self.model_id, device, compute_type)
        if progress is not None:
            progress(f"ASR準備完了: {self.model_id} ({device}/{compute_type})")

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
        return self._transcribe(audio, beam_size=self.final_beam_size)

    def transcribe_partial(self, audio: np.ndarray) -> str:
        """部分文字起こし(表示用)。速さ優先で beam_size=1。"""
        return self._transcribe(audio, beam_size=1)


KotobaWhisperAsr = StreamingAsr
