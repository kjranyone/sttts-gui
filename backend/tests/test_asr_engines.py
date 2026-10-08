import os
from pathlib import Path

import numpy as np
import pytest

from sttts_server.config import default_config
from sttts_server.engines.asr import create_asr
from sttts_server.engines.asr_whisper import StreamingAsr, resolve_device


def test_resolve_device_auto_prefers_cuda_fp16():
    assert resolve_device("auto", "auto", cuda_count=1) == ("cuda", "float16")


def test_resolve_device_auto_falls_back_to_cpu_int8():
    assert resolve_device("auto", "auto", cuda_count=0) == ("cpu", "int8")
    assert resolve_device(None, None, cuda_count=0) == ("cpu", "int8")


def test_resolve_device_explicit_values_win():
    assert resolve_device("cpu", "auto", cuda_count=2) == ("cpu", "int8")
    assert resolve_device("cuda", "int8_float16", cuda_count=0) == ("cuda", "int8_float16")
    assert resolve_device("auto", "float32", cuda_count=0) == ("cpu", "float32")


def test_default_engine_is_kotoba_auto():
    cfg = default_config()["asr"]
    assert cfg["engine"] == "kotoba"
    asr = create_asr(cfg)
    assert isinstance(asr, StreamingAsr)
    assert asr.requested_device == "auto"
    assert asr.requested_compute_type == "auto"
    assert asr.model_id == "kotoba-tech/kotoba-whisper-v2.0-faster"


def test_create_reazon_engine_without_loading():
    from sttts_server.engines.asr_reazon import ReazonSpeechAsr, model_files

    cfg = dict(default_config()["asr"], engine="reazonspeech", reazon_model_dir="/tmp/x")
    asr = create_asr(cfg)
    assert isinstance(asr, ReazonSpeechAsr)
    assert asr.precision == "fp32"
    files = model_files("/m", "fp32")
    assert files["encoder"].endswith("encoder-epoch-99-avg-1.onnx")
    assert model_files("/m", "int8")["joiner"].endswith("joiner-epoch-99-avg-1.int8.onnx")


def test_reazon_missing_files_raise(tmp_path):
    pytest.importorskip("sherpa_onnx")
    cfg = dict(default_config()["asr"], engine="reazonspeech", reazon_model_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        create_asr(cfg).load()


def test_unknown_engine_rejected():
    with pytest.raises(ValueError):
        create_asr({"engine": "nope"})


def test_mock_asr_cycles_texts():
    asr = create_asr({"engine": "mock", "mock_texts": ["あいうえお", "かきくけこ"]})
    asr.load()
    one_sec = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe_partial(one_sec) == "あいうえお"[:5]
    assert asr.transcribe_utterance(one_sec) == "あいうえお"
    assert asr.transcribe_utterance(one_sec) == "かきくけこ"
    assert asr.transcribe_utterance(one_sec) == "あいうえお"


@pytest.mark.skipif(not os.environ.get("STTTS_REAZON_DIR"), reason="STTTS_REAZON_DIR 未設定(実モデル)")
def test_reazon_real_model_transcribes():
    sf = pytest.importorskip("soundfile")
    wav = Path(os.environ["STTTS_REAZON_DIR"]) / "test_wavs" / "1.wav"
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000
    asr = create_asr({"engine": "reazonspeech", "reazon_model_dir": os.environ["STTTS_REAZON_DIR"]})
    asr.load()
    assert asr.transcribe_utterance(audio)
