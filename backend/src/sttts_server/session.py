"""ライブセッション: マイク → silero VAD → ストリーミングASR → 確定文を app へ通知。

- マイクは最優先で開くため、ASR モデルのロード中でもレベルメータは即座に動く。
  ASR の準備ができるまでの発話は文字起こしせず破棄する。
- レベルメータは ASR/VAD と別スレッド(ポンプ)で処理する。whisper のデコードは
  1発話数秒かかるが、メータも VAD ループも止まらない。
- ASR デコード(部分/確定)は専用ワーカーへ投げる。ループがデコードでブロック
  すると VAD の発話終了イベントの処理が部分的な再デコードに埋もれて確定文が
  出なくなるため、ループは一切デコードしない。
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
        # マイク直受け(raw_q) → レベルポンプ → ASR 処理(audio_q) の2段構成。
        # デコードで audio_q の消費が止まってもレベルメータは生き続ける。
        self.raw_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=200)
        self.audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=200)
        # ("final" | "partial", audio, utterance_id) のデコードジョブ
        self._decode_q: queue.Queue[tuple[str, np.ndarray, int] | None] = queue.Queue()
        self._stop = threading.Event()
        self._asr_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._mic: MicStream | None = None

    # ---------- 生存期間 ----------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="asr-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # 終端通知は満杯時に備えてタイムアウト付き(各消費者は停止フラグでも抜ける)
        try:
            self.raw_q.put(None, timeout=2)
        except queue.Full:
            pass
        try:
            self.audio_q.put(None, timeout=2)
        except queue.Full:
            pass
        self._decode_q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---------- 実装 ----------

    def _on_mic_block(self, block: np.ndarray) -> None:
        try:
            self.raw_q.put_nowait(block)
        except queue.Full:
            pass  # ポンプが追いつかない場合(通常起こらない)は新規を捨てる

    def _pump_level(self) -> None:
        """マイクブロックのレベル計測と ASR キューへの振り分け専用スレッド。

        ASR のデコード(1発話数秒)がセッションループをブロックしても、
        レベルメータはこのスレッドで常に 10Hz で流れ続ける。
        """
        last_level = 0.0
        while True:
            try:
                block = self.raw_q.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if block is None:
                break
            now = time.monotonic()
            if now - last_level > LEVEL_INTERVAL:
                rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
                db = 20.0 * float(np.log10(max(rms, 1e-6)))
                self.app.on_mic_level(rms, db)
                last_level = now
            try:
                self.audio_q.put_nowait(block)
            except queue.Full:
                pass  # ASR が追いつかない間は新規を捨てる(従来同様)

    def _submit_decode(self, kind: str, audio: np.ndarray, utterance_id: int) -> None:
        self._decode_q.put((kind, audio, utterance_id))

    def _decode_worker(self) -> None:
        """ASR デコード専用ワーカー。ループを塞がないため VAD イベントは常に即処理される。"""
        while True:
            job = self._decode_q.get()
            if job is None:
                break
            kind, audio, utterance_id = job
            try:
                if kind == "final":
                    text = self.asr.transcribe_utterance(audio)
                    if text:
                        self.app.on_asr_final(utterance_id, text)
                    else:
                        self.app.on_asr_partial(utterance_id, "")
                else:
                    text = self.asr.transcribe_partial(audio)
                    self.app.on_asr_partial(utterance_id, text)
            except Exception:
                # ワーカーが死ぬと以後の確定文が全て消えるため、例外を呑んで継続する
                log.exception("decode failed (%s)", kind)

    def _load_asr(self) -> None:
        """ASR モデルのロード。失敗してもセッション(マイク/レベルメータ)は生かす。"""
        try:
            with _engine_quiet_stdout():
                self.asr.load(lambda m, f=None: self.app._set_asr("loading", m))
        except Exception as e:
            log.exception("asr load failed")
            self.app.on_asr_error(f"ASRモデルのロードに失敗: {e}")
            return
        self._asr_ready.set()
        self.app.on_asr_model_ready(self.asr.model_id)

    def _run(self) -> None:
        # マイクを最優先で開く。ASR のロード完了を待つとその間レベルメータが
        # 完全に死ぬ(初回はモデル取得に数十秒〜分単位)ため、並行して進める。
        try:
            self._mic = MicStream(self.app.config["audio"].get("input_device_index"), self._on_mic_block)
            self._mic.start()
        except Exception as e:
            log.exception("mic open failed")
            self.app.on_asr_error(f"マイクを開けませんでした: {e}")
            return
        threading.Thread(target=self._pump_level, name="mic-level-pump", daemon=True).start()
        threading.Thread(target=self._load_asr, name="asr-loader", daemon=True).start()
        threading.Thread(target=self._decode_worker, name="asr-decode", daemon=True).start()

        vad = SileroVad()
        utterance: list[np.ndarray] = []
        speaking = False
        last_partial = 0.0
        utterance_id = 1
        tail: np.ndarray = np.zeros(0, dtype=np.float32)  # 512フレーム整列用

        while not self._stop.is_set():
            try:
                block = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if block is None:
                break

            now = time.monotonic()
            tail = np.concatenate([tail, block])
            try:
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
                            if self._asr_ready.is_set():
                                self._submit_decode("final", audio, utterance_id)
                            else:
                                log.info("utterance dropped: ASR not ready yet")
                        utterance_id += 1
                        vad.reset()
                    elif (
                        speaking
                        and self._asr_ready.is_set()
                        and now - last_partial > self.partial_interval
                    ):
                        buf = np.concatenate(utterance) if utterance else None
                        if buf is not None and buf.size >= 16000 * 0.6:
                            # デコード待ちが既にあるなら今のバッファは古くなるので捨てる
                            if self._decode_q.empty():
                                partial_audio = buf[-int(16000 * MAX_PARTIAL_SECONDS) :]
                                self._submit_decode("partial", partial_audio, utterance_id)
                        last_partial = now
                tail = tail[n_frames * FRAME :]
            except Exception:
                # ループ内の例外でスレッドが死ぬとマイクを開いたまま誰も消費せず、
                # レベルメータも固まるため、エラーを通知して継続する。
                # ハンドラ内の例外で死ぬと本末転倒なので reset も防護する。
                log.exception("session loop error")
                self.app.log("セッション処理でエラーが発生しました(継続します)", "warn")
                tail = np.zeros(0, dtype=np.float32)
                speaking = False
                utterance = []
                try:
                    vad.reset()
                except Exception:
                    log.exception("vad reset failed")

        if self._mic is not None:
            self._mic.stop()
        log.info("live session stopped")
