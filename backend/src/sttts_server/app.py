"""NDJSON stdio バックエンド本体。

- stdout はプロトコルメッセージ専用(1行1JSON)。人間可読なログはすべて stderr へ。
- stdin から GUI→backend メッセージを受け取り dispatch する。
- TTS は専用ワーカースレッド + キュー(チャンク逐次合成=疑似ストリーミング)。
- マイク+ASR はセッションスレッド(start_session / stop_session)。
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import queue
import secrets
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import BACKEND_VERSION
from .chunker import split_chunks
from .config import default_config, merge_config
from .protocol import ERROR, IDLE, LOADING, MODEL_CATALOG, PROTOCOL_VERSION, READY

# stderr 用ロガー(stdout はプロトコル専用のため)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sttts")

# irodori / dacvae / silentcipher は print() でプロセスの stdout へ直接書くため、
# エンジン呼び出し中は fd レベルで stdout→stderr へリダイレクトする。
# プロトコル出力は起動時に複製した専用 fd へ書くため影響を受けない。
_FD_REDIRECT_LOCK = threading.Lock()


@contextlib.contextmanager
def _engine_quiet_stdout():
    with _FD_REDIRECT_LOCK:
        saved = os.dup(1)
        try:
            sys.stdout.flush()
            os.dup2(2, 1)
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


@dataclass
class SynthResult:
    wav_bytes: bytes
    sample_rate: int
    duration_ms: int
    gen_ms: int
    used_seed: int | None
    path: str | None = None


@dataclass
class TtsJob:
    request: int
    chunk: int
    text: str
    caption: str | None
    ref_wavs: list[str] | None
    seed: int | None


@dataclass
class ProgressFn:
    """モデルDL/ロード進捗を state メッセージへ変換するコールバック。"""

    app: "BackendApp"
    engine: str = "tts"

    def __call__(self, message: str, frac: float | None = None) -> None:
        detail = f"{message} ({frac * 100:.0f}%)" if frac is not None else message
        if self.engine == "tts":
            self.app._set_tts(LOADING, detail)
        else:
            self.app._set_asr(LOADING, detail)


class BackendApp:
    def __init__(self, *, mock: bool = False, output_dir: str = "output") -> None:
        self.mock = mock
        self.output_dir = output_dir
        self.config = default_config()

        # プロトコル専用ストリーム(元の fd 1 の複製。エンジンの fd リダイレクトの影響を受けない)
        self._proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)

        self._out_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._tts_phase = IDLE
        self._tts_detail: str | None = None
        self._tts_loaded_model: str | None = None
        self._asr_phase = IDLE
        self._asr_detail: str | None = None
        self._asr_loaded_model: str | None = None
        self._mic_running = False

        self._engine = None  # TtsEngine
        self._engine_lock = threading.Lock()

        self._tts_queue: queue.Queue[TtsJob | None] = queue.Queue()
        self._tts_thread: threading.Thread | None = None
        self._pending: dict[int, dict] = {}  # request -> {total, done, cancelled}
        self._pending_lock = threading.Lock()
        self._request_seq = 0

        self._session = None  # SessionRunner
        self._utterance_seq = 0

        self._stop = threading.Event()

    # ---------- 出力系 ----------

    def send(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False)
        with self._out_lock:
            self._proto.write(line + "\n")
            self._proto.flush()

    def log(self, message: str, level: str = "info") -> None:
        self.send({"type": "log", "level": level, "message": message})

    def send_error(self, scope: str, message: str, recoverable: bool = True) -> None:
        self.send({"type": "error", "scope": scope, "message": message, "recoverable": recoverable})

    def send_state(self) -> None:
        with self._state_lock:
            msg = {
                "type": "state",
                "tts": {
                    "phase": self._tts_phase,
                    "detail": self._tts_detail,
                    "model": self._tts_loaded_model,
                },
                "asr": {
                    "phase": self._asr_phase,
                    "detail": self._asr_detail,
                    "model": self._asr_loaded_model,
                },
                "mic_running": self._mic_running,
            }
        self.send(msg)

    def _set_tts(self, phase: str, detail: str | None) -> None:
        with self._state_lock:
            self._tts_phase = phase
            self._tts_detail = detail
        self.send_state()

    def _set_asr(self, phase: str, detail: str | None) -> None:
        with self._state_lock:
            self._asr_phase = phase
            self._asr_detail = detail
        self.send_state()

    def send_devices(self) -> None:
        """入出力デバイス一覧を送る。mock モードでも列挙を試みる(失敗時は空リスト)。"""
        inputs: list[dict] = []
        outputs: list[dict] = []
        try:
            import sounddevice as sd

            default_in = sd.default.device[0]
            for i, d in enumerate(sd.query_devices()):
                info = {
                    "index": i,
                    "name": d["name"],
                    "default_rate": int(d["default_samplerate"]),
                    "is_default": i == default_in,
                }
                if d.get("max_input_channels", 0) > 0:
                    inputs.append(info)
                if d.get("max_output_channels", 0) > 0:
                    outputs.append(info)
        except Exception as e:  # PortAudio が無くても致命傷にしない
            log.warning("device enumeration failed: %s", e)
        self.send({"type": "devices", "inputs": inputs, "outputs": outputs})

    # ---------- stdin ループ ----------

    def _preimport_heavy_deps(self) -> None:
        """scipy を他エンジン(ctranslate2/onnxruntime)より先に読み込む。

        irodori(silentcipher) は from_key 中に scipy を import するが、
        faster-whisper 等を先にロードしたプロセスで scipy の OpenBLAS 系 DLL の
        ロードがデッドロックすることがある(Windows)。起動時に完了させておけば
        後続の import はキャッシュヒットになり競合が起きない。
        """
        t0 = time.perf_counter()
        try:
            import scipy.linalg  # noqa: F401, PLC0415
            import scipy.signal  # noqa: F401, PLC0415

            log.info("pre-import scipy ok in %.1fs", time.perf_counter() - t0)
        except Exception:
            log.exception("scipy pre-import failed (続行します)")

    def run_stdio(self) -> None:
        self._preimport_heavy_deps()
        self.send(
            {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "mock": self.mock,
                "python": sys.version.split()[0],
                "backend_version": BACKEND_VERSION,
                "models": MODEL_CATALOG,
            }
        )
        self.send_state()
        self.send_devices()

        self._tts_thread = threading.Thread(
            target=self._tts_worker, name="tts-worker", daemon=True
        )
        self._tts_thread.start()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("unparseable input: %s (%s)", line[:120], e)
                continue
            try:
                self.dispatch(msg)
            except Exception:
                log.exception("dispatch failed for %s", msg.get("type"))
            if self._stop.is_set():
                break

        self._teardown()

    def dispatch(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "configure":
            self.configure(msg)
        elif mtype == "start_session":
            self.start_session()
        elif mtype == "stop_session":
            self.stop_session()
        elif mtype == "speak":
            self.speak(msg)
        elif mtype == "cancel_speak":
            self.cancel_speak()
        elif mtype == "ping":
            self.send({"type": "pong", "nonce": msg.get("nonce", 0)})
        elif mtype == "shutdown":
            self._stop.set()
        else:
            self.log(f"未知のメッセージ型: {mtype}", "warn")

    # ---------- 設定 ----------

    def configure(self, msg: dict) -> None:
        patch = {
            k: v
            for k, v in msg.items()
            if k in ("tts", "asr", "audio", "voice", "pipeline") and v is not None
        }
        self.config = merge_config(self.config, patch)
        self.log(f"configure 適用: {sorted(patch.keys())}", "debug")

    # ---------- TTS ----------

    def speak(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            self.send_error("tts", "空のテキストです")
            return
        chunks = self._split(text)
        if not chunks:
            self.send_error("tts", "チャンクに分割できませんでした")
            return

        voice = self.config["voice"]
        caption = msg.get("caption")
        ref_wavs = msg.get("ref_wavs")
        if caption is None and voice.get("caption"):
            caption = voice["caption"]
        if ref_wavs is None and voice.get("ref_wavs"):
            ref_wavs = list(voice["ref_wavs"])

        with self._pending_lock:
            self._request_seq += 1
            request = self._request_seq
            self._pending[request] = {
                "total": len(chunks),
                "done": 0,
                "cancelled": False,
                "failed": False,
            }

        self.send(
            {
                "type": "speak_accepted",
                "request": request,
                "origin": msg.get("origin", "manual"),
                "tag": msg.get("tag"),
            }
        )
        seed = self._request_seed(msg.get("seed"))
        for i, chunk_text in enumerate(chunks):
            self._tts_queue.put(
                TtsJob(
                    request=request,
                    chunk=i,
                    text=chunk_text,
                    caption=caption,
                    ref_wavs=ref_wavs,
                    seed=seed,
                )
            )

    def _split(self, text: str) -> list[str]:
        pipeline = self.config["pipeline"]
        return split_chunks(
            text,
            min_chars=int(pipeline.get("chunk_min_chars", 16)),
            first_min_chars=int(pipeline.get("first_chunk_min_chars", 1)),
            max_chars=int(pipeline.get("chunk_max_chars", 80) or 0),
            first_mora_min=float(pipeline.get("first_chunk_mora_min", 8) or 0),
            first_mora_max=float(pipeline.get("first_chunk_mora_max", 12) or 0),
        )

    @staticmethod
    def _request_seed(seed) -> int:
        """リクエスト単位の seed。未指定(ランダム)ならここで1回だけ決めて全チャンクで共有する
        (チャンクごとに別 seed だと、参照音声なしの声質がチャンク間で変わるため)。"""
        if seed is None:
            return secrets.randbits(31)
        return int(seed)

    def _save_wav(self, wav_bytes: bytes) -> str | None:
        """生成WAVを output/ へ保存し、パスを返す(失敗時は None、致命傷にしない)。"""
        try:
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"tts_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
            path.write_bytes(wav_bytes)
            return str(path)
        except OSError:
            log.exception("failed to save wav")
            return None

    def cancel_speak(self) -> None:
        drained: list[TtsJob] = []
        while True:
            try:
                job = self._tts_queue.get_nowait()
            except queue.Empty:
                break
            drained.append(job)
        affected = sorted({j.request for j in drained})
        with self._pending_lock:
            for req in affected:
                info = self._pending.get(req)
                if info is None:
                    continue
                info["cancelled"] = True
                self._send_speak_done_locked(req)
        # 合成中の1チャンクは中断不可(irodori 仕様)
        self.log("キューを破棄しました(合成中の1チャンクは中断できません)", "warn")

    def _send_speak_done_locked(self, request: int) -> None:
        """_pending_lock 保持中に呼ぶこと。done==total か cancelled で speak_done を送る。"""
        info = self._pending.get(request)
        if info is None:
            return
        if info["done"] >= info["total"] or info["cancelled"]:
            del self._pending[request]
            self.send(
                {
                    "type": "speak_done",
                    "request": request,
                    "chunks": info["done"],
                    "cancelled": info["cancelled"],
                    "failed": info["failed"],
                }
            )

    def _load_engine_locked(self) -> None:
        """呼び出しは TTS ワーカースレッドからのみ。必要ならモデルをロードする。"""
        cfg = self.config["tts"]
        model = cfg["model"]
        if (
            self._engine is not None
            and self._tts_loaded_model == model
            and self._tts_phase == READY
        ):
            return
        if self._engine is not None:
            self.log(f"旧モデルを解放: {self._tts_loaded_model}")
            try:
                self._engine.unload()
            except Exception:
                log.exception("unload failed")
            self._engine = None

        self._set_tts(LOADING, f"loading {model}")
        progress = ProgressFn(self, "tts")
        if self.mock:
            from .engines.mock import MockTts

            engine = MockTts(model_id=model)
        else:
            from .engines.tts_irodori import IrodoriTts

            engine = IrodoriTts(
                model_id=model,
                device=str(cfg.get("device") or "auto"),
                num_steps=cfg.get("num_steps"),
                decode_mode=str(cfg.get("decode_mode") or "sequential"),
            )
        engine.load(progress)
        self._engine = engine
        self._tts_loaded_model = model
        self._set_tts(READY, None)

    def _tts_worker(self) -> None:
        while True:
            job = self._tts_queue.get()
            if job is None or self._stop.is_set():
                break
            with self._pending_lock:
                info = self._pending.get(job.request)
                cancelled = bool(info and info["cancelled"])
            if cancelled:
                # キャンセル済みリクエストの残チャンクはスキップ
                continue
            try:
                with self._engine_lock:
                    with _engine_quiet_stdout():
                        self._load_engine_locked()
                self.send(
                    {
                        "type": "tts_chunk_start",
                        "request": job.request,
                        "chunk": job.chunk,
                        "text": job.text,
                    }
                )
                with _engine_quiet_stdout():
                    result = self._engine.synthesize(
                        job.text,
                        caption=job.caption,
                        ref_wavs=job.ref_wavs,
                        seed=job.seed,
                        progress=ProgressFn(self, "tts"),
                    )
                path = self._save_wav(result.wav_bytes)
                self.send(
                    {
                        "type": "tts_audio",
                        "request": job.request,
                        "chunk": job.chunk,
                        "wav_base64": base64.b64encode(result.wav_bytes).decode("ascii"),
                        "sample_rate": result.sample_rate,
                        "duration_ms": result.duration_ms,
                        "gen_ms": result.gen_ms,
                        "path": path,
                        "seed": result.used_seed,
                    }
                )
                self.send(
                    {
                        "type": "tts_chunk_done",
                        "request": job.request,
                        "chunk": job.chunk,
                        "gen_ms": result.gen_ms,
                    }
                )
                with self._pending_lock:
                    info = self._pending.get(job.request)
                    if info is not None:
                        info["done"] += 1
                        self._send_speak_done_locked(job.request)
            except Exception as e:
                log.exception("synthesis failed")
                self.send_error("tts", f"合成失敗: {e}", recoverable=True)
                with self._pending_lock:
                    info = self._pending.get(job.request)
                    if info is not None:
                        info["done"] += 1
                        info["failed"] = True
                        self._send_speak_done_locked(job.request)

    # ---------- セッション(マイク+ASR) ----------

    def start_session(self) -> None:
        if self._session is not None:
            self.log("セッションは既に実行中です", "warn")
            return
        self._utterance_seq += 1
        asr_cfg = self.config["asr"]
        try:
            if self.mock:
                from .engines.mock import MockSession

                session = MockSession(self)
            else:
                from .session import LiveSession

                session = LiveSession(self)
        except Exception as e:
            log.exception("session init failed")
            self.send_error("asr", f"セッション初期化失敗: {e}")
            return
        self._session = session
        session.start()
        with self._state_lock:
            self._mic_running = True
        label = asr_cfg["model"] if asr_cfg.get("engine", "kotoba") == "kotoba" else asr_cfg.get("engine")
        self._set_asr(LOADING, f"loading {label}")
        self.log("マイクセッション開始")

    def stop_session(self) -> None:
        if self._session is None:
            return
        self._session.stop()
        self._session = None
        with self._state_lock:
            self._mic_running = False
        self._set_asr(IDLE if not self._asr_loaded_model else READY, None)
        self.log("マイクセッション停止")

    # ---------- セッション→アプリ コールバック ----------

    def on_asr_model_ready(self, model_id: str) -> None:
        self._asr_loaded_model = model_id
        self._set_asr(READY, None)

    def on_asr_error(self, message: str) -> None:
        self._set_asr(ERROR, message)
        self.send_error("asr", message)

    def on_mic_level(self, rms: float, db: float) -> None:
        self.send({"type": "mic_level", "rms": rms, "db": db})

    def on_asr_partial(self, utterance: int, text: str, asr_ms: int | None = None) -> None:
        self.send({"type": "asr_partial", "utterance": utterance, "text": text})

    def on_asr_final(self, utterance: int, text: str, timing: dict | None = None) -> None:
        self.send({"type": "asr_final", "utterance": utterance, "text": text})
        if self.config["pipeline"]["auto_speak"] and text.strip():
            self.speak({"text": text, "origin": "auto"})

    # ---------- 終了 ----------

    def _teardown(self) -> None:
        self._stop.set()
        try:
            if self._session is not None:
                self._session.stop()
        except Exception:
            log.exception("session stop failed")
        self._tts_queue.put(None)  # ワーカー起床用
        if self._tts_thread is not None:
            # 合成中の1チャンクは中断できないため、完了を待ってから解放する
            # (解放を先に行うと synthesize 中に del self.model で AttributeError になる)
            self._tts_thread.join(timeout=300)
        # 未完了リクエスト(未合成チャンクは破棄)に speak_done を返してプロトコルを閉じる
        with self._pending_lock:
            for request in sorted(self._pending.keys()):
                info = self._pending[request]
                info["cancelled"] = True
                self._send_speak_done_locked(request)
        with self._engine_lock:
            if self._engine is not None:
                try:
                    with _engine_quiet_stdout():
                        self._engine.unload()
                except Exception:
                    log.exception("engine unload failed")
                self._engine = None
        log.info("backend stopped")
