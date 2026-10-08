"""設定の既定値とマージ。設定の永続化は GUI(Rust)側が担い、backend は configure で受ける。"""

from __future__ import annotations

import copy
from typing import Any

DEFAULTS: dict[str, Any] = {
    "tts": {
        "model": "v4.1-small-mf",
        "device": "auto",  # "auto" | "cuda" | "xpu" | "cpu"
        "num_steps": None,  # None で checkpoint 既定(MF:4 / RF:40)
        "decode_mode": "sequential",  # 低VRAM 既定
        # "auto": CUDA cc<8.0(RTX 20xx 等)→fp32、cc>=8.0 / XPU→bf16、CPU→fp32
        "precision": "auto",
        "warmup": True,  # 起動後(初回 configure 時)にモデルをロードして短文を1回合成しておく
        "compile": False,  # torch.compile(初回が遅く、Windows では triton が必要)
        "cache_conditions": True,  # text/caption/speaker エンコード結果のメモ化
        "ref_latent_cache": True,  # 参照 WAV の DACVAE latent をディスクにキャッシュ
        "ref_cache_dir": None,  # None で ~/.cache/sttts-gui/ref_latents(Windows は %LOCALAPPDATA%)
    },
    "asr": {
        "engine": "kotoba",  # "kotoba"(faster-whisper) | "reazonspeech"(sherpa-onnx) | "mock"
        "model": "kotoba-tech/kotoba-whisper-v2.0-faster",
        "device": "auto",  # "auto"(CUDA があれば cuda) | "cuda" | "cpu"
        "compute_type": "auto",  # "auto"(cuda→float16 / cpu→int8) | "int8" | "float16" | ...
        "cpu_threads": 0,  # 0 = CTranslate2 既定
        "final_beam_size": 2,
        "partial_interval_ms": 800,  # 0 で partial(途中経過表示)を無効化
        "language": "ja",
        # silero VAD: 無音がこの長さ続いたら発話終了(従来 400ms)。短いほど速いが文中の間で切れやすい
        "vad_min_silence_ms": 280,
        "vad_threshold": 0.5,
        # ReazonSpeech(asr.engine = "reazonspeech")用
        "reazon_model": "reazon-research/reazonspeech-k2-v2",
        "reazon_model_dir": None,  # 指定時は HF からDLせずこのディレクトリを使う
        "reazon_precision": "fp32",  # int8 は短い発話で崩れやすいので非推奨
        "reazon_threads": 4,
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
        "chunk_min_chars": 16,  # 2チャンク目以降の最小文字数
        "chunk_max_chars": 80,  # これを超える塊は読点/文節境界で分割(句読点なし ASR 出力対策)
        "first_chunk_min_chars": 1,
        # 先頭チャンクを読点か約 8〜12 モーラで切って初音を早める(max=0 で無効)
        "first_chunk_mora_min": 8,
        "first_chunk_mora_max": 12,
        # 投機的 TTS(既定 OFF): 同じ先頭チャンクが N 回連続した partial から先行合成し、
        # 確定文の先頭チャンクと完全一致した場合だけ再生に回す(不一致なら破棄)
        "speculative_tts": False,
        "speculative_stable_partials": 2,
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
