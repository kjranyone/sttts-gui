"""ライブセッション: マイク → silero VAD → ストリーミングASR → 確定文を app へ通知。

- 部分文字起こし: 発話中バッファを partial_interval_ms ごとに再デコード(表示専用、
  長すぎる場合は直近 max_partial_seconds のみ)。
- 確定: VAD が発話終了を検出した時点でバッファ全体を高品質デコードして asr_final。
  確定文は app.on_asr_final 経由で自動発話(auto_speak)に回る。
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from .app import _engine_quiet_stdout
from .engines.asr_whisper import StreamingAsr
from .engines.mic import MicStream
from .engines.vad_silero import FRAME, SileroVad

log = logging.getLogger("sttts.session")

MIN_UTTERANCE_SECONDS = 0.25
MAX_PARTIAL_SECONDS = 12.0
LEVEL_INTERVAL = 0.1


class LiveSession:
    def __init__(self, app) -> None:
        self.app = app
        cfg = app.config["asr"]
        self.asr = StreamingAsr(
            model_id=cfg["model"],
            compute_type=cfg["compute_type"],
            language=cfg["language"],
        )
        self.partial_interval = max(0.4, cfg["partial_interval_ms"] / 1000.0)
        self.audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mic: MicStream | None = None

    # ---------- 生存期間 ----------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="asr-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.audio_q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---------- 実装 ----------

    def _on_mic_block(self, block: np.ndarray) -> None:
        try:
            self.audio_q.put_nowait(block)
        except queue.Full:
            pass  # ASR が追いつかない場合は最古を捨てるより新規を捨てる

    def _run(self) -> None:
        try:
            with _engine_quiet_stdout():
                self.asr.load(lambda m, f=None: self.app._set_asr("loading", m))
        except Exception as e:
            log.exception("asr load failed")
            self.app.on_asr_error(f"ASRモデルのロードに失敗: {e}")
            return
        self.app.on_asr_model_ready(self.asr.model_id)

        try:
            self._mic = MicStream(self.app.config["audio"].get("input_device_index"), self._on_mic_block)
            self._mic.start()
        except Exception as e:
            log.exception("mic open failed")
            self.app.on_asr_error(f"マイクを開けませんでした: {e}")
            return

        vad = SileroVad()
        utterance: list[np.ndarray] = []
        speaking = False
        last_partial = 0.0
        last_level = 0.0
        utterance_id = 1
        tail: np.ndarray = np.zeros(0, dtype=np.float32)  # 512フレーム整列用

        while not self._stop.is_set():
            try:
                block = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if block is None:
                break

            # レベルメータ(スロットル付き)
            now = time.monotonic()
            if now - last_level > LEVEL_INTERVAL:
                rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
                db = 20.0 * float(np.log10(max(rms, 1e-6)))
                self.app.on_mic_level(rms, db)
                last_level = now

            tail = np.concatenate([tail, block])
            n_frames = len(tail) // FRAME
            if n_frames == 0:
                continue
            for i in range(n_frames):
                frame = tail[i * FRAME : (i + 1) * FRAME]
                event = vad.process(frame)
                if event and "start" in event:
                    speaking = True
                    utterance = []
                    last_partial = now
                if speaking:
                    utterance.append(frame)
                if event and "end" in event:
                    speaking = False
                    audio = np.concatenate(utterance) if utterance else np.zeros(0, dtype=np.float32)
                    utterance = []
                    if audio.size >= 16000 * MIN_UTTERANCE_SECONDS:
                        text = self.asr.transcribe_utterance(audio)
                        if text:
                            self.app.on_asr_final(utterance_id, text)
                        else:
                            self.app.on_asr_partial(utterance_id, "")
                    utterance_id += 1
                    vad.reset()
                elif speaking and now - last_partial > self.partial_interval:
                    buf = np.concatenate(utterance) if utterance else None
                    if buf is not None and buf.size >= 16000 * 0.6:
                        partial_audio = buf[-16000 * MAX_PARTIAL_SECONDS :]
                        text = self.asr.transcribe_partial(partial_audio)
                        self.app.on_asr_partial(utterance_id, text)
                    last_partial = now
            tail = tail[n_frames * FRAME :]

        if self._mic is not None:
            self._mic.stop()
        log.info("live session stopped")
