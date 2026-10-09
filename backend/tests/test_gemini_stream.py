import asyncio
import sys
import types

import numpy as np

from sttts_server.engines.asr_gemini import GeminiLiveAsr


class FakeSession:
    def __init__(self):
        self.audio = []
        self.audio_ready = asyncio.Event()
        self.end = asyncio.Event()

    async def send_realtime_input(self, *, audio=None, audio_stream_end=False):
        if audio is not None:
            self.audio.append(audio.data)
            self.audio_ready.set()
        if audio_stream_end:
            self.end.set()

    async def receive(self):
        await self.audio_ready.wait()
        yield types.SimpleNamespace(server_content=types.SimpleNamespace(
            interim_input_transcription=types.SimpleNamespace(text="こん"),
            input_transcription=None, turn_complete=False, generation_complete=False,
        ))
        await self.end.wait()
        yield types.SimpleNamespace(server_content=types.SimpleNamespace(
            interim_input_transcription=None,
            input_transcription=types.SimpleNamespace(text="こんにちは"),
            turn_complete=True, generation_complete=False,
        ))


class FakeConnection:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_):
        return False


def test_gemini_stream_sends_audio_before_vad_end(monkeypatch):
    session = FakeSession()
    fake_types = types.SimpleNamespace(
        LiveConnectConfig=lambda **kw: kw,
        AudioTranscriptionConfig=lambda **kw: kw,
        Blob=lambda **kw: types.SimpleNamespace(**kw),
    )
    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = fake_types
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    live = types.SimpleNamespace(connect=lambda **kw: FakeConnection(session))
    asr = GeminiLiveAsr(api_key="dummy", timeout_s=2)
    asr._client = types.SimpleNamespace(aio=types.SimpleNamespace(live=live))
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


def test_stop_aborts_open_gemini_stream(monkeypatch):
    session = FakeSession()
    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = types.SimpleNamespace(
        LiveConnectConfig=lambda **kw: kw,
        AudioTranscriptionConfig=lambda **kw: kw,
        Blob=lambda **kw: types.SimpleNamespace(**kw),
    )
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    asr = GeminiLiveAsr(api_key="dummy", timeout_s=2)
    asr._client = types.SimpleNamespace(aio=types.SimpleNamespace(
        live=types.SimpleNamespace(connect=lambda **kw: FakeConnection(session))
    ))
    asr.begin_stream(1, lambda *_: None)
    stream = asr._streams[1]
    asr.abort_all_streams()
    assert stream._done.is_set()
    assert not stream._thread.is_alive()
