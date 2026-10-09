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


def test_create_nemotron_engine_without_loading():
    from sttts_server.engines.asr_nemotron_onnx import NemotronOnnxAsr

    cfg = dict(
        default_config()["asr"],
        engine="nemotron",
        nemotron_model_dir="/tmp/x",
        nemotron_chunk_ms=320,
        nemotron_precision="fp16",
    )
    asr = create_asr(cfg)
    assert isinstance(asr, NemotronOnnxAsr)
    assert asr.chunk_ms == 320
    assert asr.precision == "fp16"
    assert asr.language == "ja-JP"  # "ja" は locale 辞書向けに正規化される
    assert asr.engine_name == "nemotron"


def test_nemotron_language_normalization():
    from sttts_server.engines.asr_nemotron_onnx import _normalize_language

    assert _normalize_language("ja") == "ja-JP"
    assert _normalize_language("en") == "en-US"
    assert _normalize_language("ja-JP") == "ja-JP"
    assert _normalize_language("auto") == "auto"
    assert _normalize_language(None) == "auto"
    assert _normalize_language("zh") == "zh-CN"


def test_nemotron_not_loaded_raises():
    cfg = dict(default_config()["asr"], engine="nemotron", nemotron_model_dir="/tmp/x")
    asr = create_asr(cfg)
    with pytest.raises(RuntimeError, match="not loaded"):
        asr.transcribe_utterance(np.zeros(1600, dtype=np.float32))


@pytest.mark.skipif(
    not os.environ.get("STTTS_NEMOTRON_DIR"), reason="STTTS_NEMOTRON_DIR 未設定(実モデル)"
)
def test_nemotron_real_model_transcribes():
    pytest.importorskip("onnxruntime")
    asr = create_asr(
        {
            "engine": "nemotron",
            "nemotron_model_dir": os.environ["STTTS_NEMOTRON_DIR"],
            "nemotron_chunk_ms": int(os.environ.get("STTTS_NEMOTRON_CHUNK_MS", "320")),
        }
    )
    asr.load()
    # 1秒の無音は空文字、短い定常音でもクラッシュしない
    assert isinstance(asr.transcribe_utterance(np.zeros(16000, dtype=np.float32)), str)


def test_create_gemini_engine_without_loading():
    from sttts_server.engines.asr_gemini import DEFAULT_MODEL, GeminiLiveAsr

    cfg = dict(default_config()["asr"], engine="gemini")
    asr = create_asr(cfg)
    assert isinstance(asr, GeminiLiveAsr)
    assert asr.model_id == DEFAULT_MODEL
    assert asr.mode == "SMART"
    assert asr.language == "ja-JP"  # "ja" は BCP-47 へ正規化
    assert asr.engine_name == "gemini"


def test_gemini_language_normalization():
    from sttts_server.engines.asr_gemini import GeminiLiveAsr

    assert GeminiLiveAsr(language="ja").language == "ja-JP"
    assert GeminiLiveAsr(language="ja-JP").language == "ja-JP"
    assert GeminiLiveAsr(language="en").language == "en"  # ja 以外はそのまま
    assert GeminiLiveAsr(mode="verbatim").mode == "VERBATIM"
    assert GeminiLiveAsr(mode="smart").mode == "SMART"


def test_gemini_api_key_resolution(monkeypatch):
    from sttts_server.engines.asr_gemini import resolve_api_key

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert resolve_api_key(None) is None
    assert resolve_api_key("cfg-key") == "cfg-key"
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert resolve_api_key(None) == "env-key"
    assert resolve_api_key("cfg-key") == "cfg-key"  # config が優先


def test_gemini_missing_key_raises_at_load(mock_app, monkeypatch):
    asr = create_asr({"engine": "gemini"})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API キー"):
        asr.load()


def test_gemini_loads_with_sdk_and_key():
    """venv に google-genai がある環境ではキー付き load が成功する(SDK 欠落時の
    メッセージは実行時パスのためここではカバーしない)。"""
    pytest.importorskip("google.genai")
    from sttts_server.engines.asr_gemini import GeminiLiveAsr

    asr = GeminiLiveAsr(api_key="dummy-key-for-init")
    asr.load()
    assert asr._client is not None


@pytest.mark.skipif(
    not os.environ.get("STTTS_GEMINI_API_KEY"), reason="STTTS_GEMINI_API_KEY 未設定(実API)"
)
def test_gemini_real_api_transcribes():
    pytest.importorskip("google.genai")
    import soundfile as sf

    wav = Path(__file__).parents[2] / "data" / "gemini_test.wav"
    if not wav.is_file():
        pytest.skip("data/gemini_test.wav がありません")
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000
    asr = create_asr(
        {
            "engine": "gemini",
            "gemini_api_key": os.environ["STTTS_GEMINI_API_KEY"],
        }
    )
    asr.load()
    text = asr.transcribe_utterance(audio)
    assert isinstance(text, str) and len(text) > 0
