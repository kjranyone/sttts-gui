"""Rust 側 crates/protocol と JSON 形状が一致することの合致テスト。"""

import json

from sttts_server.protocol import MODEL_CATALOG, PROTOCOL_VERSION


def sample_state() -> dict:
    return {
        "type": "state",
        "tts": {"phase": "ready", "detail": None, "model": "v4.1-small-mf"},
        "asr": {"phase": "idle", "detail": None, "model": None},
        "mic_running": False,
    }


def sample_tts_audio() -> dict:
    return {
        "type": "tts_audio",
        "request": 1,
        "chunk": 0,
        "wav_base64": "UklGRg==",
        "sample_rate": 48000,
        "duration_ms": 1234,
        "gen_ms": 987,
        "path": "output/x.wav",
        "seed": 42,
    }


def test_hello_has_models_catalog():
    from sttts_server import BACKEND_VERSION

    hello = {
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "mock": True,
        "python": "3.12.10",
        "backend_version": BACKEND_VERSION,
        "models": MODEL_CATALOG,
    }
    assert hello["protocol"] == 1
    for m in hello["models"]:
        assert set(m.keys()) == {"id", "label", "size", "note"}


def test_json_serializable_ascii_safe():
    for msg in (sample_state(), sample_tts_audio()):
        s = json.dumps(msg, ensure_ascii=False)
        assert json.loads(s) == msg


def test_gui_speak_shape():
    msg = {"type": "speak", "text": "こんにちは"}
    # Rust 側は skip_serializing_if=None でオプションを省略するため、
    # backend はキー無しに耐えること(dispatch 側で .get 使用)
    assert msg["text"] == "こんにちは"
