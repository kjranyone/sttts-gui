"""設定の既定値とマージ。

設定の永続化は GUI(Rust)側が担い、backend は configure で受ける。GUI に UI が無い
上級設定(ASR エンジン・VAD・投機的 TTS 等)は、任意の JSON ファイル
`<repo>/data/backend.json`(環境変数 STTTS_CONFIG / --config で変更可)に書くと
起動時に既定値へマージされる。GUI からの configure はその上に適用される。
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("sttts.config")

SECTIONS = ("tts", "asr", "audio", "voice", "pipeline")

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
        "engine": "kotoba",  # "kotoba"(faster-whisper) | "reazonspeech"(sherpa-onnx) | "nemotron"(onnxruntime) | "mock"
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
        # Nemotron 3.5 ASR(asr.engine = "nemotron")用
        "nemotron_repo": "codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx",
        "nemotron_model_dir": None,  # 指定時は HF からDLせずこのディレクトリを使う(自前export向け)
        "nemotron_chunk_ms": 320,  # 320(HFパッケージ既定)。他は自前exportが必要
        "nemotron_precision": "fp16",  # int8 は dynamic quantum で精度劣化するため非推奨
        "nemotron_threads": 4,  # onnxruntime の intra_op スレッド数
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


def default_user_config_path() -> Path:
    env = os.environ.get("STTTS_CONFIG")
    if env:
        return Path(env)
    # src/sttts_server/config.py → リポジトリルート/data/backend.json
    return Path(__file__).resolve().parents[3] / "data" / "backend.json"


def load_user_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """ユーザー設定 JSON を読む。無ければ {}。未知のセクションは無視して警告する。"""
    p = Path(path) if path else default_user_config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("設定ファイルを読めません(無視します): %s: %s", p, e)
        return {}
    if not isinstance(data, dict):
        log.warning("設定ファイルの形式が不正です(オブジェクトではない): %s", p)
        return {}
    unknown = sorted(k for k in data if k not in SECTIONS)
    if unknown:
        log.warning("設定ファイルの未知のセクションを無視: %s", unknown)
    log.info("設定ファイルを読み込みました: %s", p)
    return {k: v for k, v in data.items() if k in SECTIONS and isinstance(v, dict)}
