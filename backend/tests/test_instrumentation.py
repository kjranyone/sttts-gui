"""計測フィールド(protocol の timing)と WAV 入力ソースのテスト。"""

import threading
import time

import numpy as np

from sttts_server.protocol import TIMING_FIELDS

from .test_speculative import _wait


def _run_worker(app):
    t = threading.Thread(target=app._tts_worker, daemon=True)
    t.start()
    return t


def test_auto_speak_carries_speech_end_to_e2e(mock_app):
    app = mock_app
    t = _run_worker(app)
    speech_end = time.monotonic() - 0.5  # 0.5 秒前に話し終わった
    app.on_asr_final(1, "こんにちは、今日はいい天気ですね。", timing={"speech_end": speech_end, "vad_end": speech_end + 0.28, "asr_ms": 120, "audio_ms": 2000})
    assert _wait(lambda: len(app.of_type("speak_done")) == 1, timeout=5)
    app._tts_queue.put(None)
    t.join(2)
    final = app.of_type("asr_final")[0]
    assert final["asr_ms"] == 120 and final["audio_ms"] == 2000
    assert 270 <= final["vad_wait_ms"] <= 290
    assert abs(final["speech_end_ms"] - speech_end * 1000) < 1
    acc = app.of_type("speak_accepted")[0]
    assert acc["utterance"] == 1 and acc["speech_end_ms"] == final["speech_end_ms"]
    audio = app.of_type("tts_audio")
    first, rest = audio[0], audio[1:]
    assert first["first_chunk"] is True and first["e2e_ms"] >= 500
    assert first["first_chunk_ms"] <= first["e2e_ms"]
    assert all("e2e_ms" not in a and a["first_chunk"] is False for a in rest)
    assert all(a["rtf"] is not None and a["queue_wait_ms"] is not None for a in audio)


def test_timing_fields_are_documented_for_every_sent_field(mock_app):
    app = mock_app
    t = _run_worker(app)
    app.on_asr_partial(1, "こんにちは", asr_ms=30)
    app.on_asr_final(1, "こんにちは。", timing={"speech_end": time.monotonic(), "vad_end": time.monotonic(), "asr_ms": 1, "audio_ms": 1})
    assert _wait(lambda: len(app.of_type("speak_done")) == 1, timeout=5)
    app._tts_queue.put(None)
    t.join(2)
    for mtype, fields in TIMING_FIELDS.items():
        msgs = app.of_type(mtype)
        assert msgs, mtype
        for f in fields:
            if f in ("first_chunk_ms", "e2e_ms"):
                assert f in msgs[0], (mtype, f)
            else:
                assert all(f in m for m in msgs), (mtype, f)


def test_mock_tts_rtf_simulation():
    from sttts_server.engines.mock import MockTts

    tts = MockTts("m", delay_ms=0, rtf=0.1)
    r = tts.synthesize("あ" * 20)  # 0.4 + 1.8 = 2.2 秒 → 合成 ≈ 220ms
    assert 180 <= r.gen_ms <= 400


def test_wav_source_streams_realtime_and_signals_eof(tmp_path):
    import soundfile as sf

    from sttts_server.engines.wav_source import WavSource

    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(48000 // 2, dtype=np.float32), 48000)  # 0.5 秒 @48k → 16k へ変換
    blocks, eof = [], threading.Event()
    src = WavSource([str(p)], blocks.append, gap_s=0.2, lead_s=0.1, on_eof=eof.set)
    t0 = time.monotonic()
    src.start()
    assert eof.wait(5)
    elapsed = time.monotonic() - t0
    total = sum(len(b) for b in blocks)
    assert abs(total - int(16000 * 0.8)) <= 480
    assert elapsed >= 0.7  # 実時間ペース(0.8 秒分)
    assert all(b.dtype == np.float32 for b in blocks)
