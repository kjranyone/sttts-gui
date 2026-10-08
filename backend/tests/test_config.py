from sttts_server.config import DEFAULTS, default_config, merge_config


def test_defaults_shape():
    cfg = default_config()
    assert set(cfg.keys()) == {"tts", "asr", "audio", "voice", "pipeline"}
    assert cfg["tts"]["model"] == "v4.1-small-mf"
    assert cfg["pipeline"]["auto_speak"] is True
    assert cfg["asr"]["engine"] == "kotoba"  # 既定エンジンは変えない


def test_merge_partial_section():
    cfg = merge_config(default_config(), {"tts": {"model": "v4-large"}})
    assert cfg["tts"]["model"] == "v4-large"
    assert cfg["tts"]["device"] == "auto"  # 他キーは保持
    assert cfg["asr"]["model"] == DEFAULTS["asr"]["model"]


def test_merge_does_not_mutate():
    base = default_config()
    merge_config(base, {"voice": {"caption": "落ち着いた声"}})
    assert base["voice"]["caption"] is None
