"""tts.sampling: Irodori の SamplingRequest 項目を GUI 非対応でも設定から使えること。"""

import pytest

from sttts_server.config import DEFAULTS, merge_config
from sttts_server.engines.tts_irodori import RESERVED_SAMPLING_KEYS, check_sampling_overrides


def test_sampling_defaults_empty_and_user_config_merges():
    assert DEFAULTS["tts"]["sampling"] == {}
    cfg = merge_config(DEFAULTS, {"tts": {"sampling": {"cfg_scale_text": 2.0, "duration_scale": 1.1}}})
    assert cfg["tts"]["sampling"] == {"cfg_scale_text": 2.0, "duration_scale": 1.1}
    # 後続 patch は項目単位でマージされる
    cfg = merge_config(cfg, {"tts": {"sampling": {"cfg_scale_text": 4.0}}})
    assert cfg["tts"]["sampling"] == {"cfg_scale_text": 4.0, "duration_scale": 1.1}


def test_check_sampling_passes_irodori_options():
    opts = {"cfg_scale_speaker": 6.0, "truncation_factor": 0.8, "lora_adapter": "x", "num_steps": 8}
    assert check_sampling_overrides(opts) == opts
    assert check_sampling_overrides(None) == {}


@pytest.mark.parametrize("key", sorted(RESERVED_SAMPLING_KEYS))
def test_check_sampling_rejects_app_owned_keys(key):
    with pytest.raises(ValueError, match=key):
        check_sampling_overrides({key: 1})


def test_mock_engine_accepts_sampling(mock_app):
    from sttts_server.engines.mock import MockTts

    r = MockTts(model_id="m").synthesize("あ", sampling={"cfg_scale_text": 2.0})
    assert r.sample_rate > 0
