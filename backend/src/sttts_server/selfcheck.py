"""self-check 用ユーティリティ(CLI から独立したファイルにして import を最小化する)。"""

from __future__ import annotations

import json
import sys

import numpy as np
import sounddevice as sd
import soxr


def self_check_asr(*, seconds: float = 5.0, model: str | None = None, mock: bool = False) -> int:
    """マイクから seconds 秒録音し、ASR エンジンで文字起こしして JSON を stdout に出す。"""
    if mock:
        print(json.dumps({"ok": True, "text": "(mock) セルフチェックの文字起こしです", "seconds": seconds}))
        return 0

    from sttts_server.engines.asr_whisper import StreamingAsr

    info = sd.query_devices(kind="input")
    in_rate = int(info["default_samplerate"])
    print(f"recording {seconds}s from '{info['name']}' @ {in_rate}Hz ...", file=sys.stderr)
    blocks: list[np.ndarray] = []
    total = 0.0
    with sd.InputStream(
        samplerate=in_rate, channels=1, dtype="float32", blocksize=int(in_rate * 0.1)
    ) as stream:
        while total < seconds:
            data, _ = stream.read(int(in_rate * 0.1))
            blocks.append(data[:, 0].copy())
            total += data.shape[0] / in_rate
    audio = soxr.resample(np.concatenate(blocks), in_rate, 16000).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"recorded {audio.size / 16000:.1f}s peak={peak:.3f}", file=sys.stderr)
    if peak < 1e-4:
        print(json.dumps({"ok": False, "error": "no mic input detected (peak ~0)"}))
        return 1

    asr = StreamingAsr(
        model_id=model or "kotoba-tech/kotoba-whisper-v2.0-faster", compute_type="int8", language="ja"
    )
    asr.load(lambda m, f=None: print(f"[load] {m}", file=sys.stderr))
    text = asr.transcribe_utterance(audio)
    print(
        json.dumps({"ok": True, "text": text, "seconds": audio.size / 16000}, ensure_ascii=False)
    )
    return 0
