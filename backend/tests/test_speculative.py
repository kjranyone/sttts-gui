"""投機的 TTS のテスト(モックモード、実モデル不要)。"""

import threading
import time

import pytest

from sttts_server.engines.mock import MockTts
from sttts_server.protocol import READY


class GateTts(MockTts):
    def __init__(self):
        super().__init__("gate")
        self.texts: list[str] = []
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()

    def synthesize(self, text, **kw):
        self.texts.append(text)
        self.started.set()
        self.gate.wait(5)
        return super().synthesize(text, **kw)


@pytest.fixture
def spec_app(mock_app):
    app = mock_app
    app.config["pipeline"]["speculative_tts"] = True
    app.config["pipeline"]["speculative_stable_partials"] = 2
    eng = GateTts()
    app._engine = eng
    app._tts_loaded_model = app.config["tts"]["model"]
    app._tts_phase = READY
    t = threading.Thread(target=app._tts_worker, daemon=True)
    t.start()
    app.eng = eng
    yield app
    app._tts_queue.put(None)
    t.join(2)


def _wait(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _audio(app):
    return app.of_type("tts_audio")


def _done(app):
    return app.of_type("speak_done")


PARTIAL = "こんにちは、今日はいい天気"
FINAL = "こんにちは、今日はいい天気ですね。"


def test_speculative_disabled_by_default(mock_app):
    assert mock_app.config["pipeline"]["speculative_tts"] is False
    for _ in range(3):
        mock_app.on_asr_partial(1, PARTIAL)
    assert mock_app._tts_queue.empty()


def test_match_reuses_speculative_first_chunk(spec_app):
    app = spec_app
    app.on_asr_partial(1, PARTIAL)
    app.on_asr_partial(1, PARTIAL)
    assert _wait(lambda: app.eng.texts == ["こんにちは、"])
    time.sleep(0.15)
    assert _audio(app) == []  # 確定前は絶対に送らない
    app.on_asr_final(1, FINAL)
    assert _wait(lambda: len(_done(app)) == 1)
    audio = _audio(app)
    assert [(a["chunk"], a["speculative"]) for a in audio] == [(0, True), (1, False)]
    starts = [m["text"] for m in app.of_type("tts_chunk_start")]
    assert starts == ["こんにちは、", "今日はいい天気ですね。"]
    assert app.eng.texts == ["こんにちは、", "今日はいい天気ですね。"]  # 先頭は1回だけ合成
    assert audio[0]["seed"] == audio[1]["seed"]  # 投機チャンクとも seed を共有
    assert _done(app)[0]["chunks"] == 2 and not _done(app)[0]["cancelled"]


def test_mismatch_discards_speculative_audio(spec_app):
    app = spec_app
    app.on_asr_partial(1, PARTIAL)
    app.on_asr_partial(1, PARTIAL)
    assert _wait(lambda: app.eng.texts == ["こんにちは、"])
    app.on_asr_final(1, "こんばんは、今日はいい天気ですね。")
    assert _wait(lambda: len(_done(app)) == 1)
    starts = [m["text"] for m in app.of_type("tts_chunk_start")]
    assert starts == ["こんばんは、", "今日はいい天気ですね。"]
    assert all(a["speculative"] is False for a in _audio(app))
    assert len(_audio(app)) == 2


def test_bind_while_speculation_in_flight_keeps_order(spec_app):
    app = spec_app
    app.eng.gate.clear()
    app.on_asr_partial(1, PARTIAL)
    app.on_asr_partial(1, PARTIAL)
    assert app.eng.started.wait(2)  # 投機合成が走っている最中に確定
    app.on_asr_final(1, FINAL)
    time.sleep(0.1)
    assert _audio(app) == []
    app.eng.gate.set()
    assert _wait(lambda: len(_done(app)) == 1)
    audio = _audio(app)
    assert [(a["chunk"], a["speculative"]) for a in audio] == [(0, True), (1, False)]
    assert app.eng.texts == ["こんにちは、", "今日はいい天気ですね。"]


def test_mismatch_while_in_flight_never_plays_wrong_audio(spec_app):
    app = spec_app
    app.eng.gate.clear()
    app.on_asr_partial(1, PARTIAL)
    app.on_asr_partial(1, PARTIAL)
    assert app.eng.started.wait(2)
    app.on_asr_final(1, "こんばんは、今日はいい天気ですね。")
    app.eng.gate.set()
    assert _wait(lambda: len(_done(app)) == 1)
    starts = [m["text"] for m in app.of_type("tts_chunk_start")]
    assert "こんにちは、" not in starts
    assert starts == ["こんばんは、", "今日はいい天気ですね。"]


def test_unstable_partials_do_not_speculate(spec_app):
    app = spec_app
    app.on_asr_partial(1, PARTIAL)
    app.on_asr_partial(1, "こんばんは、今日はいい天気")
    app.on_asr_partial(1, PARTIAL)
    time.sleep(0.2)
    assert app.eng.texts == []


def test_partial_without_boundary_does_not_speculate(spec_app):
    app = spec_app
    for _ in range(3):
        app.on_asr_partial(1, "こんにちは")  # 先頭チャンクの後ろが未確定
    time.sleep(0.2)
    assert app.eng.texts == []


def test_voice_change_between_spec_and_final_discards(spec_app):
    app = spec_app
    app.on_asr_partial(1, PARTIAL)
    app.on_asr_partial(1, PARTIAL)
    assert _wait(lambda: app.eng.texts == ["こんにちは、"])
    app.config["voice"]["caption"] = "落ち着いた女性の声"
    app.on_asr_final(1, FINAL)
    assert _wait(lambda: len(_done(app)) == 1)
    assert all(a["speculative"] is False for a in _audio(app))
    assert app.eng.texts == ["こんにちは、", "こんにちは、", "今日はいい天気ですね。"]
