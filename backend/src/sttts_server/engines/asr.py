"""ASR エンジンのファクトリ。asr.engine で選択する(既定 "kotoba")。

全エンジンは同じインターフェースを持つ:
    load(progress) / transcribe_utterance(audio) / transcribe_partial(audio) / model_id
audio は 16kHz mono float32 の numpy 配列。
"""

from __future__ import annotations

from typing import Any

ENGINES = ("kotoba", "reazonspeech", "mock")


def create_asr(cfg: dict[str, Any]):
    engine = str(cfg.get("engine") or "kotoba").lower()
    if engine in ("kotoba", "whisper", "faster-whisper"):
        from .asr_whisper import StreamingAsr  # noqa: PLC0415

        return StreamingAsr(
            model_id=cfg.get("model") or "kotoba-tech/kotoba-whisper-v2.0-faster",
            compute_type=cfg.get("compute_type") or "auto",
            language=cfg.get("language") or "ja",
            device=cfg.get("device") or "auto",
            cpu_threads=int(cfg.get("cpu_threads") or 0),
            final_beam_size=int(cfg.get("final_beam_size") or 2),
        )
    if engine in ("reazonspeech", "reazon"):
        from .asr_reazon import DEFAULT_REPO, ReazonSpeechAsr  # noqa: PLC0415

        return ReazonSpeechAsr(
            model_id=cfg.get("reazon_model") or DEFAULT_REPO,
            model_dir=cfg.get("reazon_model_dir"),
            precision=cfg.get("reazon_precision") or "fp32",
            num_threads=int(cfg.get("reazon_threads") or 4),
        )
    if engine == "mock":
        from .mock import MockAsr  # noqa: PLC0415

        return MockAsr(
            latency_ms=int(cfg.get("mock_latency_ms") or 0),
            texts=cfg.get("mock_texts"),
            load_delay_ms=int(cfg.get("mock_load_delay_ms") or 0),
        )
    raise ValueError(f"unknown asr.engine: {engine!r} (choices: {', '.join(ENGINES)})")
