"""キャンセル: 合成中チャンクの破棄・speak_done の一意性・キャンセル後の再開。"""

import threading
import time

import pytest

from sttts_server.protocol import READY

from .test_speculative import GateTts, _wait


@pytest.fixture
def gate_app(mock_app):
    app = mock_app
    eng = GateTts()
    app._engine = eng
    app._tts_loaded_model = app.config["tts"]["model"]
    app._tts_phase = READY
    t = threading.Thread(target=app._tts_worker, daemon=True)
    t.start()
    app.eng = eng
    yield app
    app.eng.gate.set()
    app._tts_queue.put(None)
    t.join(2)


def test_cancel_drops_chunk_being_synthesized(gate_app):
    app = gate_app
    app.eng.gate.clear()
    app.speak({"text": "これは一つだけのチャンクです。"})
    assert app.eng.started.wait(2)  # 合成中(Irodori は中断できない)
    app.cancel_speak()
    app.eng.gate.set()
    time.sleep(0.3)
    assert app.of_type("tts_audio") == []  # 合成中だったチャンクも送らない
    done = app.of_type("speak_done")
    assert len(done) == 1 and done[0]["cancelled"] is True


def test_cancel_drops_queued_and_inflight_chunks_of_multi_chunk_request(gate_app):
    app = gate_app
    app.eng.gate.clear()
    app.speak({"text": "こんにちは、今日はいい天気ですね。明日も晴れるといいですね。"})
    assert app.eng.started.wait(2)
    app.cancel_speak()
    app.eng.gate.set()
    time.sleep(0.3)
    assert app.of_type("tts_audio") == []
    assert len(app.eng.texts) == 1  # 後続チャンクは合成もしない
    assert [d["cancelled"] for d in app.of_type("speak_done")] == [True]


def test_speak_after_cancel_plays_normally(gate_app):
    app = gate_app
    app.eng.gate.clear()
    app.speak({"text": "一つ目です。"})
    assert app.eng.started.wait(2)
    app.cancel_speak()
    app.eng.gate.set()
    app.speak({"text": "二つ目です。"})
    assert _wait(lambda: len(app.of_type("speak_done")) == 2)
    audio = app.of_type("tts_audio")
    assert [a["request"] for a in audio] == [2]
    assert app.of_type("speak_done")[1] == {
        "type": "speak_done",
        "request": 2,
        "chunks": 1,
        "cancelled": False,
        "failed": False,
    }


def test_cancel_discards_speculation(gate_app):
    app = gate_app
    app.config["pipeline"]["speculative_tts"] = True
    app.eng.gate.clear()
    app.on_asr_partial(1, "こんにちは、今日はいい天気")
    app.on_asr_partial(1, "こんにちは、今日はいい天気")
    assert app.eng.started.wait(2)
    app.cancel_speak()
    app.eng.gate.set()
    app.on_asr_final(1, "こんにちは、今日はいい天気ですね。")
    assert _wait(lambda: len(app.of_type("speak_done")) == 1)
    assert all(a["speculative"] is False for a in app.of_type("tts_audio"))
