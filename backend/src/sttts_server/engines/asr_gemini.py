"""Gemini Live API(gemini-3.5-transcribe-live)によるクラウド ASR(任意エンジン)。

asr.engine = "gemini" で選択。google-genai は標準依存。
API キーは asr.gemini_api_key(data/backend.json)または環境変数 GEMINI_API_KEY /
GOOGLE_API_KEY から解決する。

統合形態: ローカル silero VAD の start で発話専用 Live セッションを開き、
activity_start → 録音中から 100ms 単位で音声 → end で activity_end を送って確定を待つ。
サーバ側自動 VAD は無効化する(`_live_config`)。有効のままだと、発話後の無音でサーバが
先に確定(input_transcription + generation_complete)を返し、その後の audio_stream_end には
何も返らないため、確定待ちが毎回タイムアウトしていた(2026-10-09 実 API で確認)。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

log = logging.getLogger("sttts.asr")

DEFAULT_MODEL = "gemini-3.5-transcribe-live"


def _live_config(types, language: str, mode: str):
    """発話区切りはローカル VAD が決める(サーバ自動 VAD 無効、activity_start/end を明示送信)。"""
    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[language], mode=mode),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
    )


def resolve_api_key(configured: str | None) -> str | None:
    """config のキー > GEMINI_API_KEY > GOOGLE_API_KEY の順に解決する。"""
    if configured:
        return configured
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiLiveAsr:
    engine_name = "gemini"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        language: str = "ja-JP",
        mode: str = "VERBATIM",
        timeout_s: float = 20.0,
    ) -> None:
        self.model_id = model_id or DEFAULT_MODEL
        self.api_key = api_key
        # "ja" 等の短いコードは BCP-47 に揃える
        self.language = language if "-" in language else f"{language}-JP" if language == "ja" else language
        self.mode = "SMART" if str(mode).upper() == "SMART" else "VERBATIM"
        self.timeout_s = float(timeout_s)
        self._client = None
        self._streams: dict[int, _LiveUtterance] = {}
        self._streams_lock = threading.Lock()

    @property
    def model_id_resolved(self) -> str:
        return self.model_id

    def load(self, progress=None) -> None:
        key = resolve_api_key(self.api_key)
        if not key:
            raise RuntimeError(
                "Gemini API キーがありません。GUI の「キー」欄に AI Studio で発行したキーを"
                "入力してください(data/backend.json の asr.gemini_api_key / 環境変数 GEMINI_API_KEY でも可)"
            )
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "Gemini エンジンには google-genai が必要です: uv sync を実行してください"
            ) from e
        if progress is not None:
            progress(f"Gemini 接続準備: {self.model_id} ({self.mode})")
        self._client = genai.Client(api_key=key)
        log.info("ASR ready: Gemini Live API (%s, %s, lang=%s)", self.model_id, self.mode, self.language)
        if progress is not None:
            progress(f"ASR準備完了: Gemini ({self.mode})")

    def transcribe_utterance(self, audio: np.ndarray) -> str:
        return self._transcribe(audio)

    def transcribe_partial(self, audio: np.ndarray) -> str:
        # 発話途中バッファも同様に一括送信する(interim は使わず確定を返す)
        return self._transcribe(audio)

    def begin_stream(self, utterance: int, on_partial: Callable[[int, str], None]) -> None:
        if self._client is None:
            raise RuntimeError("ASR not loaded")
        stream = _LiveUtterance(self, utterance, on_partial)
        with self._streams_lock:
            if utterance in self._streams:
                raise RuntimeError(f"duplicate Gemini utterance {utterance}")
            self._streams[utterance] = stream
        stream.start()

    def feed_stream(self, utterance: int, audio: np.ndarray) -> None:
        with self._streams_lock:
            stream = self._streams.get(utterance)
        if stream is not None:
            pcm = np.clip(audio, -1.0, 1.0)
            stream.feed((pcm * 32767.0).astype("<i2").tobytes())

    def end_stream(self, utterance: int) -> None:
        with self._streams_lock:
            stream = self._streams.get(utterance)
        if stream is not None:
            stream.end()

    def finish_stream(self, utterance: int, audio: np.ndarray) -> str:
        with self._streams_lock:
            stream = self._streams.get(utterance)
        if stream is None:
            raise RuntimeError(f"Gemini stream unavailable for utterance {utterance}")
        stream.end()
        try:
            return stream.result(self.timeout_s)
        finally:
            with self._streams_lock:
                if self._streams.get(utterance) is stream:
                    self._streams.pop(utterance, None)

    def abort_stream(self, utterance: int) -> None:
        with self._streams_lock:
            stream = self._streams.pop(utterance, None)
        if stream is not None:
            stream.abort()

    def abort_all_streams(self) -> None:
        with self._streams_lock:
            streams = list(self._streams.values())
            self._streams.clear()
        for stream in streams:
            stream.abort()

    # ---------- 実装 ----------

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._client is None:
            raise RuntimeError("ASR not loaded")
        pcm = np.clip(audio, -1.0, 1.0)
        raw = (pcm * 32767.0).astype("<i2").tobytes()
        # AsrWorker スレッドから呼ばれるため、スレッドローカルのイベントループで回す
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # 既にループが動いているケース(通常ない)は新しいスレッドで実行
            import concurrent.futures  # noqa: PLC0415

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, self._run(raw)).result()
        return asyncio.run(self._run(raw))

    async def _run(self, raw_pcm: bytes) -> str:
        from google.genai import types  # noqa: PLC0415

        config = _live_config(types, self.language, self.mode)
        async with self._client.aio.live.connect(model=self.model_id, config=config) as session:
            await session.send_realtime_input(activity_start=types.ActivityStart())
            await session.send_realtime_input(
                audio=types.Blob(data=raw_pcm, mime_type="audio/pcm;rate=16000")
            )
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            final_text = ""
            async for response in session.receive():
                sc = response.server_content
                if sc is None:
                    continue
                if sc.input_transcription and sc.input_transcription.text:
                    final_text = sc.input_transcription.text
                if sc.turn_complete or sc.generation_complete:
                    break
            return final_text.strip()


class _LiveUtterance:
    """1発話の Gemini 接続。VAD 側からの feed/end は待機しない。"""

    def __init__(self, asr: GeminiLiveAsr, utterance: int, on_partial: Callable[[int, str], None]) -> None:
        self.asr = asr
        self.utterance = utterance
        self.on_partial = on_partial
        self._queue: deque[bytes | None] = deque()
        self._lock = threading.Lock()
        self._ended = False
        self._aborted = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, name=f"gemini-live-{utterance}", daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._text = ""
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def feed(self, raw: bytes) -> None:
        with self._lock:
            if not self._ended and not self._aborted:
                self._queue.append(raw)

    def end(self) -> None:
        with self._lock:
            if not self._ended:
                self._ended = True
                self._queue.append(None)

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            self._queue.clear()
            self._queue.append(None)
            loop, task = self._loop, self._task
        if loop is not None and task is not None and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def result(self, timeout_s: float) -> str:
        if not self._done.wait(timeout_s):
            self.abort()
            if self._text:
                # 確定シグナルは来なかったが確定テキストは受け取っている。捨てずに使う
                log.warning("Gemini Live: %.0fs 内に完了シグナルなし。受信済みの確定テキストを使う", timeout_s)
                return self._text.strip()
            raise TimeoutError("Gemini Live transcription timed out")
        self._thread.join(timeout=1)
        if self._error is not None:
            raise RuntimeError(f"Gemini Live transcription failed: {self._error}") from self._error
        return self._text.strip()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except asyncio.CancelledError:
            pass
        except BaseException as e:
            self._error = e
        finally:
            self._done.set()

    async def _run(self) -> None:
        from google.genai import types  # noqa: PLC0415

        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        if self._aborted:
            return
        config = _live_config(types, self.asr.language, self.asr.mode)
        async with self.asr._client.aio.live.connect(model=self.asr.model_id, config=config) as session:
            sender_done = asyncio.Event()

            async def sender() -> None:
                await session.send_realtime_input(activity_start=types.ActivityStart())
                pending = bytearray()
                last_send = time.monotonic()
                while True:
                    with self._lock:
                        item = self._queue.popleft() if self._queue else ...
                    if item is None:
                        if pending:
                            await session.send_realtime_input(
                                audio=types.Blob(data=bytes(pending), mime_type="audio/pcm;rate=16000")
                            )
                        await session.send_realtime_input(activity_end=types.ActivityEnd())
                        sender_done.set()
                        return
                    if item is not ...:
                        pending.extend(item)
                    if len(pending) >= 3200 or (pending and time.monotonic() - last_send >= 0.1):
                        await session.send_realtime_input(
                            audio=types.Blob(data=bytes(pending), mime_type="audio/pcm;rate=16000")
                        )
                        pending.clear()
                        last_send = time.monotonic()
                    if item is ...:
                        await asyncio.sleep(0.01)

            async def receiver() -> None:
                while True:
                    async for response in session.receive():
                        sc = response.server_content
                        if sc is None:
                            continue
                        interim = getattr(sc, "interim_input_transcription", None)
                        if interim is not None and interim.text:
                            try:
                                self.on_partial(self.utterance, interim.text)
                            except Exception:
                                log.exception("Gemini interim callback failed")
                        final = sc.input_transcription
                        if final is not None and final.text:
                            fragment = final.text.strip()
                            if fragment and fragment != self._text:
                                if self._text and fragment.startswith(self._text):
                                    self._text = fragment
                                else:
                                    self._text = f"{self._text} {fragment}".strip() if self._text else fragment
                        if (sc.turn_complete or sc.generation_complete) and sender_done.is_set():
                            return
                    if sender_done.is_set():
                        return
                    await asyncio.sleep(0.01)

            await asyncio.gather(sender(), receiver())
