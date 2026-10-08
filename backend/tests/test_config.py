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


def test_user_config_file_is_merged(tmp_path, monkeypatch):
    import json

    from sttts_server.config import load_user_config

    p = tmp_path / "backend.json"
    p.write_text(
        json.dumps({"asr": {"engine": "reazonspeech"}, "pipeline": {"speculative_tts": True}, "bogus": {}}),
        encoding="utf-8",
    )
    user = load_user_config(p)
    assert set(user) == {"asr", "pipeline"}
    cfg = merge_config(default_config(), user)
    assert cfg["asr"]["engine"] == "reazonspeech"
    assert cfg["asr"]["vad_min_silence_ms"] == 280  # 他キーは既定のまま
    assert cfg["pipeline"]["speculative_tts"] is True
    monkeypatch.setenv("STTTS_CONFIG", str(p))
    assert load_user_config() == user


def test_missing_or_broken_user_config_is_ignored(tmp_path):
    from sttts_server.config import load_user_config

    assert load_user_config(tmp_path / "none.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_user_config(bad) == {}
