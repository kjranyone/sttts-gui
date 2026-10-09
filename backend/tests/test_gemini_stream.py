import asyncio
import sys
import types

import numpy as np
import pytest

from sttts_server.engines.asr_gemini import GeminiLiveAsr


class FakeSession:
    """実 API(gemini-3.5-transcribe-live, サーバ VAD 無効)の応答順を模す:
    interim … → activity_end 後に input_transcription → generation_complete。turn_complete は来ない。"""

    def __init__(self, *, final=True, complete=True):
        self.final = final
        self.complete = complete
        self.events = []
        self.audio = []
        self.audio_ready = asyncio.Event()
        self.end = asyncio.Event()

    async def send_realtime_input(self, *, audio=None, audio_stream_end=None, activity_start=None, activity_end=None):
        if activity_start is not None:
            self.events.append("activity_start")
        if audio is not None:
            self.events.append("audio")
            self.audio.append(audio.data)
            self.audio_ready.set()
        if audio_stream_end:
            self.events.append("audio_stream_end")
        if activity_end is not None:
            self.events.append("activity_end")
            self.end.set()

    async def receive(self):
        await self.audio_ready.wait()
        yield types.SimpleNamespace(server_content=types.SimpleNamespace(
            interim_input_transcription=types.SimpleNamespace(text="こん"),
            input_transcription=None, turn_complete=False, generation_complete=False,
        ))
        await self.end.wait()
        if self.final:
            yield types.SimpleNamespace(server_content=types.SimpleNamespace(
                interim_input_transcription=None,
                input_transcription=types.SimpleNamespace(text="こんにちは"),
                turn_complete=None, generation_complete=None,
            ))
        if self.complete:
            yield types.SimpleNamespace(server_content=types.SimpleNamespace(
                interim_input_transcription=None, input_transcription=None,
                turn_complete=None, generation_complete=True,
            ))
        await asyncio.Event().wait()  # 以後サーバは何も送らない


class FakeConnection:
    def __init__(self, session, config=None):
        self.session = session
        self.config = config

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_):
        return False


def _fake_asr(monkeypatch, session, *, timeout_s=2.0):
    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = types.SimpleNamespace(
        LiveConnectConfig=lambda **kw: kw,
        AudioTranscriptionConfig=lambda **kw: kw,
        RealtimeInputConfig=lambda **kw: kw,
        AutomaticActivityDetection=lambda **kw: kw,
        ActivityStart=lambda: "start",
        ActivityEnd=lambda: "end",
        Blob=lambda **kw: types.SimpleNamespace(**kw),
    )
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    connections = []

    def connect(**kw):
        connections.append(FakeConnection(session, kw.get("config")))
        return connections[-1]

    asr = GeminiLiveAsr(api_key="dummy", timeout_s=timeout_s)
    asr._client = types.SimpleNamespace(aio=types.SimpleNamespace(live=types.SimpleNamespace(connect=connect)))
    return asr, connections


def test_gemini_stream_sends_audio_before_vad_end(monkeypatch):
    session = FakeSession()
    asr, connections = _fake_asr(monkeypatch, session)
    partials = []
    block = np.full(1600, 0.1, dtype=np.float32)
    asr.begin_stream(3, lambda utterance, text: partials.append((utterance, text)))
    asr.feed_stream(3, block)
    # VAD 終了前に WebSocket へ 100ms の音声が送られる。
    import time
    until = time.monotonic() + 1
    while not session.audio and time.monotonic() < until:
        time.sleep(0.01)
    assert session.audio and not session.end.is_set()
    asr.end_stream(3)
    assert asr.finish_stream(3, block) == "こんにちは"
    assert partials == [(3, "こん")]
    assert session.end.is_set()
    assert 3 not in asr._streams
    # 発話区切りはローカル VAD が決める: サーバ自動 VAD 無効 + activity_start/end を明示送信
    rtc = connections[0].config["realtime_input_config"]
    assert rtc["automatic_activity_detection"] == {"disabled": True}
    assert session.events[0] == "activity_start"
    assert session.events[-1] == "activity_end"
    assert "audio_stream_end" not in session.events


def test_gemini_stream_keeps_final_text_without_completion_signal(monkeypatch):
    # 確定テキスト受信後に完了シグナルが来なくても、タイムアウトで捨てずに返す
    session = FakeSession(complete=False)
    asr, _ = _fake_asr(monkeypatch, session, timeout_s=0.5)
    asr.begin_stream(1, lambda *_: None)
    block = np.full(1600, 0.1, dtype=np.float32)
    asr.feed_stream(1, block)
    assert asr.finish_stream(1, block) == "こんにちは"
    assert 1 not in asr._streams


def test_gemini_stream_times_out_without_any_final(monkeypatch):
    session = FakeSession(final=False, complete=False)
    asr, _ = _fake_asr(monkeypatch, session, timeout_s=0.5)
    asr.begin_stream(1, lambda *_: None)
    block = np.full(1600, 0.1, dtype=np.float32)
    asr.feed_stream(1, block)
    with pytest.raises(TimeoutError):
        asr.finish_stream(1, block)


def test_stop_aborts_open_gemini_stream(monkeypatch):
    session = FakeSession()
    asr, _ = _fake_asr(monkeypatch, session)
    asr.begin_stream(1, lambda *_: None)
    stream = asr._streams[1]
    asr.abort_all_streams()
    assert stream._done.is_set()
    assert not stream._thread.is_alive()
