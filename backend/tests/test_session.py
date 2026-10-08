"""ASR ワーカー / VAD セグメンタのテスト(実モデル不要)。"""

import threading
import time

import numpy as np

from sttts_server.engines.vad_silero import FRAME
from sttts_server.session import AsrWorker, FinalJob, LiveSession, PartialJob, VadSegmenter


class GateAsr:
    """呼び出しごとにゲートで止められる ASR。呼ばれた順序を記録する。"""

    def __init__(self, block_first: bool = True) -> None:
        self.calls: list[tuple[str, int]] = []
        self.gate = threading.Event()
        self.entered = threading.Event()
        if not block_first:
            self.gate.set()

    def _run(self, kind, audio):
        self.calls.append((kind, int(audio[0]) if audio.size else -1))
        self.entered.set()
        self.gate.wait(5)
        return f"{kind}:{int(audio[0])}"

    def transcribe_partial(self, audio):
        return self._run("partial", audio)

    def transcribe_utterance(self, audio):
        return self._run("final", audio)


def _audio(tag: int) -> np.ndarray:
    return np.full(1600, tag, dtype=np.float32)


def _final(utt: int, tag: int) -> FinalJob:
    return FinalJob(utterance=utt, audio=_audio(tag), speech_end=0.0, vad_end=0.0)


def test_inflight_partial_is_discarded_when_final_arrives():
    asr = GateAsr()
    partials, finals = [], []
    w = AsrWorker(asr, lambda u, t, ms: partials.append((u, t)), lambda j, t, ms: finals.append((j.utterance, t)))
    w.start()
    w.submit_partial(PartialJob(1, _audio(10)))
    assert asr.entered.wait(2)  # partial 処理中
    w.submit_partial(PartialJob(1, _audio(11)))  # 待機中 partial
    w.submit_final(_final(1, 99))  # 確定 → 待機中/処理中 partial は無効
    asr.gate.set()
    deadline = time.time() + 3
    while not finals and time.time() < deadline:
        time.sleep(0.01)
    w.stop()
    assert finals == [(1, "final:99")]
    assert partials == []  # 処理中だった partial の結果も捨てる
    assert ("partial", 11) not in asr.calls  # 待機中 partial はデコードすらしない


def test_partials_are_coalesced_to_latest():
    asr = GateAsr()
    partials, finals = [], []
    w = AsrWorker(asr, lambda u, t, ms: partials.append((u, t)), lambda j, t, ms: finals.append(j.utterance))
    w.start()
    w.submit_final(_final(1, 1))
    assert asr.entered.wait(2)  # final 処理中に partial が3つ届く
    for tag in (20, 21, 22):
        w.submit_partial(PartialJob(2, _audio(tag)))
    asr.gate.set()
    deadline = time.time() + 3
    while not partials and time.time() < deadline:
        time.sleep(0.01)
    w.stop()
    assert finals == [1]
    assert partials == [(2, "partial:22")]
    assert w.stats["partials_dropped"] == 2


def test_finals_are_never_dropped_and_keep_order():
    asr = GateAsr(block_first=False)
    finals = []
    w = AsrWorker(asr, lambda *a: None, lambda j, t, ms: finals.append(j.utterance))
    w.start()
    for utt in range(1, 8):
        w.submit_partial(PartialJob(utt, _audio(utt)))
        w.submit_final(_final(utt, utt))
    deadline = time.time() + 3
    while len(finals) < 7 and time.time() < deadline:
        time.sleep(0.01)
    w.stop()
    assert finals == list(range(1, 8))


def test_final_has_priority_over_pending_partial():
    asr = GateAsr()
    w = AsrWorker(asr, lambda *a: None, lambda *a: None)
    w.start()
    w.submit_final(_final(1, 1))
    assert asr.entered.wait(2)
    w.submit_partial(PartialJob(2, _audio(2)))
    w.submit_final(_final(2, 3))
    asr.gate.set()
    deadline = time.time() + 3
    while w.stats["finals_done"] < 2 and time.time() < deadline:
        time.sleep(0.01)
    w.stop()
    assert asr.calls == [("final", 1), ("final", 3)]


class ScriptVad:
    """フレーム番号 → イベント の台本で動く VAD。"""

    def __init__(self, script: dict[int, dict]) -> None:
        self.script = script
        self.frame = 0  # reset 後の相対フレーム
        self.resets = 0

    def process(self, frame):
        ev = self.script.get((self.resets, self.frame))
        self.frame += 1
        return ev

    def reset(self):
        self.frame = 0
        self.resets += 1


class RecordingWorker:
    def __init__(self):
        self.partials: list[PartialJob] = []
        self.finals: list[FinalJob] = []

    def submit_partial(self, job):
        self.partials.append(job)

    def submit_final(self, job):
        self.finals.append(job)


def test_segmenter_emits_final_with_speech_end_timestamp():
    # 発話: 2 フレーム目で開始、40 フレーム目で終了検出。end サンプル = 30 フレーム目
    vad = ScriptVad({(0, 2): {"start": 2 * FRAME}, (0, 40): {"end": 30 * FRAME}})
    worker = RecordingWorker()
    clock = [100.0]
    seg = VadSegmenter(vad, worker, partial_interval=0.0, clock=lambda: clock[0])
    block = np.zeros(FRAME * 41, dtype=np.float32)
    seg.feed(block, arrival=100.0)
    assert len(worker.finals) == 1
    job = worker.finals[0]
    assert job.utterance == 1
    assert job.audio.size == FRAME * 39  # frame 2..40
    # ブロック末尾(41 フレーム)が arrival。end は 30 フレーム → 11 フレーム前
    assert abs(job.speech_end - (100.0 - 11 * FRAME / 16000)) < 1e-6
    assert seg.utterance_id == 2
    assert vad.resets == 1


def test_segmenter_speech_end_after_reset_uses_absolute_position():
    script = {
        (0, 0): {"start": 0},
        (0, 20): {"end": 10 * FRAME},
        (1, 5): {"start": 5 * FRAME},
        (1, 30): {"end": 25 * FRAME},
    }
    vad = ScriptVad(script)
    worker = RecordingWorker()
    seg = VadSegmenter(vad, worker, partial_interval=0.0, clock=lambda: 0.0)
    seg.feed(np.zeros(FRAME * 21, dtype=np.float32), arrival=10.0)  # 1 発話目
    seg.feed(np.zeros(FRAME * 40, dtype=np.float32), arrival=20.0)  # 2 発話目(31 フレームで終了)
    assert [j.utterance for j in worker.finals] == [1, 2]
    # 2 発話目: reset は絶対 21 フレーム目。end = 21 + 25 = 46、ブロック末尾 = 61
    assert abs(worker.finals[1].speech_end - (20.0 - 15 * FRAME / 16000)) < 1e-6


def test_segmenter_submits_partials_on_interval_and_handles_unaligned_blocks():
    vad = ScriptVad({(0, 0): {"start": 0}})
    worker = RecordingWorker()
    clock = [0.0]
    seg = VadSegmenter(vad, worker, partial_interval=0.5, clock=lambda: clock[0])
    for i in range(100):  # 30ms 相当(480 サンプル)ずつ、512 に揃わないブロック
        clock[0] = i * 0.03
        seg.feed(np.ones(480, dtype=np.float32))
    assert worker.partials, "partial が投げられていない"
    assert all(p.utterance == 1 for p in worker.partials)
    assert len(worker.partials) <= 3 / 0.5 + 1
    # 端数は持ち越され、音声は欠けない
    total = sum(1 for _ in seg._utterance) * FRAME + seg._tail.size
    assert total == 100 * 480


class _App:
    def __init__(self, cfg):
        self.config = cfg
        self.events = []

    def _set_asr(self, *a):
        pass

    def on_asr_model_ready(self, m):
        self.events.append(("ready", m))

    def on_asr_error(self, m):
        self.events.append(("error", m))

    def on_mic_level(self, rms, db):
        self.events.append(("level", db))

    def on_asr_partial(self, u, t, asr_ms=None):
        self.events.append(("partial", u, t))

    def on_asr_final(self, u, t, timing=None):
        self.events.append(("final", u, t, timing))


class _ListSource:
    def __init__(self, on_block, blocks):
        self.on_block = on_block
        self.blocks = blocks

    def start(self):
        for b in self.blocks:
            self.on_block(b)

    def stop(self):
        pass


def test_live_session_end_to_end_with_mock_asr():
    from sttts_server.config import default_config, merge_config

    cfg = merge_config(default_config(), {"asr": {"engine": "mock", "mock_texts": ["テストです"], "mock_latency_ms": 50}})
    app = _App(cfg)
    blocks = [np.zeros(FRAME, dtype=np.float32) for _ in range(60)]
    vad = ScriptVad({(0, 5): {"start": 5 * FRAME}, (0, 50): {"end": 40 * FRAME}})
    s = LiveSession(app, source_factory=lambda cb: _ListSource(cb, blocks), vad_factory=lambda: vad)
    s.start()
    deadline = time.time() + 5
    while not any(e[0] == "final" for e in app.events) and time.time() < deadline:
        time.sleep(0.02)
    s.stop()
    finals = [e for e in app.events if e[0] == "final"]
    assert len(finals) == 1
    _, utt, text, timing = finals[0]
    assert (utt, text) == (1, "テストです")
    assert timing["asr_ms"] >= 40
    assert timing["speech_end"] <= timing["vad_end"]


def test_mic_first_level_flows_before_asr_ready_and_audio_is_kept():
    """ASR ロード中でもレベルメータが即座に動き、ロード前の音声は捨てられない。"""
    from sttts_server.config import default_config, merge_config

    cfg = merge_config(
        default_config(),
        {"asr": {"engine": "mock", "mock_texts": ["ロード中の発話"], "mock_load_delay_ms": 400}},
    )
    app = _App(cfg)
    blocks = [np.full(FRAME, 0.1, dtype=np.float32) for _ in range(60)]
    vad = ScriptVad({(0, 5): {"start": 5 * FRAME}, (0, 50): {"end": 40 * FRAME}})
    s = LiveSession(app, source_factory=lambda cb: _ListSource(cb, blocks), vad_factory=lambda: vad)
    s.start()

    # ロード遅延(400ms)より早い時点でレベルが届いていること
    deadline = time.time() + 2
    while not any(e[0] == "level" for e in app.events) and time.time() < deadline:
        time.sleep(0.01)
    assert any(e[0] == "level" for e in app.events), "level must flow before ASR load completes"
    assert not any(e[0] == "ready" for e in app.events), "ASR must still be loading here"

    deadline = time.time() + 5
    while not any(e[0] == "final" for e in app.events) and time.time() < deadline:
        time.sleep(0.02)
    s.stop()
    # ロード前に投入された音声も pending 経由で発話として処理される
    finals = [e for e in app.events if e[0] == "final"]
    assert len(finals) == 1
    assert finals[0][2] == "ロード中の発話"


def test_vad_min_silence_default_is_280ms():
    from sttts_server.config import default_config
    from sttts_server.engines.vad_silero import SileroVad

    assert default_config()["asr"]["vad_min_silence_ms"] == 280
    vad = SileroVad()
    assert vad._iter.min_silence_samples == 16000 * 280 / 1000
    assert SileroVad(min_silence_ms=400)._iter.min_silence_samples == 6400


def test_real_silero_reset_and_multiple_utterances():
    """実 silero で reset が例外にならず、複数発話を検出できること(以前は reset() で落ちていた)。"""
    import pytest

    pytest.importorskip("scipy")
    from sttts_server.engines.vad_silero import SileroVad
    from sttts_server.testsignal import speechlike

    worker = RecordingWorker()
    seg = VadSegmenter(SileroVad(), worker, partial_interval=0.0)
    seg.feed(np.zeros(8000, dtype=np.float32), arrival=0.0)
    for k, sec in enumerate((1.2, 1.5, 1.8)):
        seg.feed(speechlike(sec, seed=k))
        seg.feed(np.zeros(32000, dtype=np.float32))
    assert len(worker.finals) >= 3
    assert [j.utterance for j in worker.finals] == list(range(1, len(worker.finals) + 1))


def test_wait_idle_waits_for_inflight_final():
    asr = GateAsr()
    finals = []
    w = AsrWorker(asr, lambda *a: None, lambda j, t, ms: finals.append(j.utterance))
    w.start()
    w.submit_final(_final(1, 1))
    assert asr.entered.wait(2)
    assert w.wait_idle(timeout=0.2) is False  # デコード中は idle ではない
    asr.gate.set()
    assert w.wait_idle(timeout=3) is True
    assert finals == [1]
    w.stop()
