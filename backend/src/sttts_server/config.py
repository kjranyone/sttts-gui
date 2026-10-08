"""設定の既定値とマージ。設定の永続化は GUI(Rust)側が担い、backend は configure で受ける。"""

from __future__ import annotations

import copy
from typing import Any

DEFAULTS: dict[str, Any] = {
    "tts": {
        "model": "v4.1-small-mf",
        "device": "auto",  # "auto" | "xpu" | "cpu"
        "num_steps": None,  # None で checkpoint 既定(MF:4 / RF:40)
        "decode_mode": "sequential",  # 低VRAM 既定
    },
    "asr": {
        "model": "kotoba-tech/kotoba-whisper-v2.0-faster",
        "compute_type": "int8",
        "partial_interval_ms": 800,
        "language": "ja",
    },
    "audio": {
        "input_device_index": None,
    },
    "voice": {
        "caption": None,
        "ref_wavs": [],
        "no_ref": True,
    },
    "pipeline": {
        "auto_speak": True,
        "chunk_min_chars": 16,
        "first_chunk_min_chars": 1,
    },
}


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULTS)


def merge_config(base: dict[str, Any], patch: dict[str, Any] | None) -> dict[str, Any]:
    """セクション単位の深いマージ。patch を破壊しない。"""
    out = copy.deepcopy(base)
    if not patch:
        return out
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
