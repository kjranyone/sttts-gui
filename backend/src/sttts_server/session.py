"""ライブセッション: マイク → silero VAD → ASR ワーカー → 確定文を app へ通知。

スレッド構成(VAD をリアルタイムに保ち、音声を一切捨てない):

- 音声ソース(PortAudio コールバック等)→ 無制限キュー audio_q へ (到着時刻, ブロック) を積む。
- VAD スレッド(asr-vad): レベルメータ・silero VAD・発話バッファ管理のみを行う。
  重いデコードは行わず、AsrWorker へジョブを投げるだけなので常に実時間で回る。
- ASR ワーカー(asr-worker): デコード専用。確定(final)ジョブを最優先で FIFO 処理し、
  部分(partial)ジョブは「最新の1件」だけを保持(古いものは上書き=合体)。
  確定が投入された発話の partial は、待機中のものも処理中のものも結果を破棄する。

発話終了時刻(speech_end)は VAD の end サンプル位置とブロック到着時刻から逆算する
(min_silence の待ち時間も含めた「ユーザーが話し終えた瞬間」を基準に計測するため)。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .app import _engine_quiet_stdout
from .engines.asr import create_asr
from .engines.vad_silero import FRAME, SileroVad

log = logging.getLogger("sttts.session")

SAMPLE_RATE = 16000
MIN_UTTERANCE_SECONDS = 0.25
MIN_PARTIAL_SECONDS = 0.6
MAX_PARTIAL_SECONDS = 12.0
LEVEL_INTERVAL = 0.1


@dataclass
class PartialJob:
    utterance: int
    audio: np.ndarray


@dataclass
class FinalJob:
    utterance: int
    audio: np.ndarray
    speech_end: float  # time.monotonic() 基準の発話終了推定時刻
    vad_end: float  # VAD が終了を検出した時刻(monotonic)
    meta: dict = field(default_factory=dict)


class AsrWorker:
    """ASR デコード専用ワーカー。final を優先し、partial は最新1件に合体する。"""

    def __init__(self, asr, on_partial, on_final, on_error=None) -> None:
        self.asr = asr
        self.on_partial = on_partial  # (utterance, text, asr_ms)
        self.on_final = on_final  # (FinalJob, text, asr_ms)
        self.on_error = on_error
        self._cv = threading.Condition()
        self._finals: deque[FinalJob] = deque()
        self._partial: PartialJob | None = None
        self._finalized_upto = 0
        self._stop = False
        self._thread: threading.Thread | None = None
        self.stats = {"partials_done": 0, "partials_dropped": 0, "finals_done": 0}

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="asr-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit_partial(self, job: PartialJob) -> None:
        with self._cv:
            if job.utterance <= self._finalized_upto:
                self.stats["partials_dropped"] += 1
                return
            if self._partial is not None:
                self.stats["partials_dropped"] += 1  # 古い partial は上書き
            self._partial = job
            self._cv.notify()

    def submit_final(self, job: FinalJob) -> None:
        with self._cv:
            self._finals.append(job)
            self._finalized_upto = max(self._finalized_upto, job.utterance)
            if self._partial is not None and self._partial.utterance <= self._finalized_upto:
                self._partial = None
                self.stats["partials_dropped"] += 1
            self._cv.notify()

    def pending(self) -> int:
        with self._cv:
            return len(self._finals) + (1 if self._partial is not None else 0)

    def _next_job(self):
        with self._cv:
            while not self._finals and self._partial is None and not self._stop:
                self._cv.wait()
            if self._finals:
                return "final", self._finals.popleft()
            if self._stop:
                return None, None
            job, self._partial = self._partial, None
            return "partial", job

    def run(self) -> None:
        while True:
            kind, job = self._next_job()
            if kind is None:
                break
            t0 = time.perf_counter()
            try:
                if kind == "final":
                    text = self.asr.transcribe_utterance(job.audio)
                else:
                    text = self.asr.transcribe_partial(job.audio)
            except Exception as e:  # 1件の失敗でワーカーを止めない
                log.exception("asr decode failed")
                if self.on_error is not None:
                    self.on_error(f"ASR デコード失敗: {e}")
                continue
            asr_ms = int((time.perf_counter() - t0) * 1000)
            if kind == "final":
                self.stats["finals_done"] += 1
                self.on_final(job, text, asr_ms)
            else:
                with self._cv:
                    stale = job.utterance <= self._finalized_upto
                if stale:
                    self.stats["partials_dropped"] += 1  # 処理中に確定が来た partial は捨てる
                    continue
                self.stats["partials_done"] += 1
                self.on_partial(job.utterance, text, asr_ms)


class VadSegmenter:
    """VAD + 発話バッファ管理(デコードはしない)。feed() を実時間で呼ぶ。"""

    def __init__(
        self,
        vad,
        worker: AsrWorker,
        *,
        partial_interval: float,
        on_level=None,
        clock=time.monotonic,
    ) -> None:
        self.vad = vad
        self.worker = worker
        self.partial_interval = partial_interval  # <=0 で partial 無効
        self.on_level = on_level
        self.clock = clock
        self.utterance_id = 1
        self._utterance: list[np.ndarray] = []
        self._speaking = False
        self._last_partial = 0.0
        self._last_level = 0.0
        self._tail = np.zeros(0, dtype=np.float32)  # 512 フレーム整列用
        self._samples_fed = 0  # VAD に与えた総サンプル数
        self._vad_base = 0  # 直近 vad.reset() 時点の _samples_fed

    def feed(self, block: np.ndarray, arrival: float | None = None) -> None:
        now = self.clock()
        arrival = now if arrival is None else arrival
        if self.on_level is not None and now - self._last_level > LEVEL_INTERVAL:
            rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
            db = 20.0 * float(np.log10(max(rms, 1e-6)))
            self.on_level(rms, db)
            self._last_level = now

        tail = np.concatenate([self._tail, block]) if self._tail.size else block
        n_frames = len(tail) // FRAME
        # このブロック末尾サンプルの絶対位置(到着時刻 arrival に対応)
        block_end_abs = self._samples_fed + len(tail)
        for i in range(n_frames):
            frame = tail[i * FRAME : (i + 1) * FRAME]
            event = self.vad.process(frame)
            self._samples_fed += FRAME
            if event and "start" in event:
                self._speaking = True
                self._utterance = []
                self._last_partial = now
            if self._speaking:
                self._utterance.append(frame)
            if event and "end" in event:
                end_abs = self._vad_base + int(event["end"])
                speech_end = arrival - max(0, block_end_abs - end_abs) / SAMPLE_RATE
                self._finish_utterance(speech_end, now)
            elif (
                self._speaking
                and self.partial_interval > 0
                and now - self._last_partial > self.partial_interval
            ):
                buf = np.concatenate(self._utterance) if self._utterance else None
                if buf is not None and buf.size >= SAMPLE_RATE * MIN_PARTIAL_SECONDS:
                    tail_audio = buf[-int(SAMPLE_RATE * MAX_PARTIAL_SECONDS) :]
                    self.worker.submit_partial(PartialJob(self.utterance_id, tail_audio))
                self._last_partial = now
        self._tail = tail[n_frames * FRAME :].copy()

    def _finish_utterance(self, speech_end: float, now: float) -> None:
        self._speaking = False
        audio = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.float32)
        self._utterance = []
        if audio.size >= SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
            self.worker.submit_final(
                FinalJob(
                    utterance=self.utterance_id,
                    audio=audio,
                    speech_end=speech_end,
                    vad_end=now,
                    meta={"audio_ms": int(audio.size / SAMPLE_RATE * 1000)},
                )
            )
        self.utterance_id += 1
        self.vad.reset()
        self._vad_base = self._samples_fed


class LiveSession:
    def __init__(self, app, *, source_factory=None, vad_factory=None) -> None:
        self.app = app
        cfg = app.config["asr"]
        self.asr = create_asr(cfg)
        interval_ms = int(cfg.get("partial_interval_ms") or 0)
        self.partial_interval = max(0.4, interval_ms / 1000.0) if interval_ms > 0 else 0.0
        self.vad_threshold = float(cfg.get("vad_threshold") or 0.5)
        self.vad_min_silence_ms = int(cfg.get("vad_min_silence_ms") or 400)
        # 無制限キュー: ASR が遅くても音声は捨てない(VAD スレッドは軽量なので溜まらない)
        self.audio_q: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source = None
        self._source_factory = source_factory
        self._vad_factory = vad_factory
        self.worker: AsrWorker | None = None

    # ---------- 生存期間 ----------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="asr-vad", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.audio_q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---------- 実装 ----------

    def _on_block(self, block: np.ndarray) -> None:
        self.audio_q.put((time.monotonic(), block))

    def _make_source(self):
        if self._source_factory is not None:
            return self._source_factory(self._on_block)
        from .engines.mic import MicStream  # noqa: PLC0415  (PortAudio はここで初めて読む)

        return MicStream(self.app.config["audio"].get("input_device_index"), self._on_block)

    def _on_final(self, job: FinalJob, text: str, asr_ms: int) -> None:
        timing = {
            "speech_end": job.speech_end,
            "vad_end": job.vad_end,
            "asr_ms": asr_ms,
            "audio_ms": job.meta.get("audio_ms"),
        }
        if text:
            self.app.on_asr_final(job.utterance, text, timing=timing)
        else:
            self.app.on_asr_partial(job.utterance, "", asr_ms=asr_ms)

    def _on_partial(self, utterance: int, text: str, asr_ms: int) -> None:
        self.app.on_asr_partial(utterance, text, asr_ms=asr_ms)

    def _run(self) -> None:
        try:
            with _engine_quiet_stdout():
                self.asr.load(lambda m, f=None: self.app._set_asr("loading", m))
        except Exception as e:
            log.exception("asr load failed")
            self.app.on_asr_error(f"ASRモデルのロードに失敗: {e}")
            return
        self.app.on_asr_model_ready(self.asr.model_id)

        self.worker = AsrWorker(self.asr, self._on_partial, self._on_final, on_error=self.app.on_asr_error)
        self.worker.start()
        vad = (
            self._vad_factory()
            if self._vad_factory is not None
            else SileroVad(threshold=self.vad_threshold, min_silence_ms=self.vad_min_silence_ms)
        )
        seg = VadSegmenter(
            vad,
            self.worker,
            partial_interval=self.partial_interval,
            on_level=self.app.on_mic_level,
        )

        try:
            self._source = self._make_source()
            self._source.start()
        except Exception as e:
            log.exception("audio source open failed")
            self.app.on_asr_error(f"マイクを開けませんでした: {e}")
            self.worker.stop()
            return

        while not self._stop.is_set():
            try:
                item = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            arrival, block = item
            seg.feed(block, arrival)

        if self._source is not None:
            self._source.stop()
        self.worker.stop()
        log.info("live session stopped (asr stats: %s)", self.worker.stats)
