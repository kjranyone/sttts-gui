"""レイテンシ計測ハーネス: 固定 WAV を実時間ペースで backend に流し、P50/P90 を出す。

backend を `--stdio --input-wav ...` で起動し、マイクの代わりに WAV を 30ms ブロックで
実時間ペース投入する(VAD・ASR ワーカー・チャンク分割・TTS ワーカーは本番と同じ経路)。

モード:
  mock: ASR = MockAsr(固定テキスト)、TTS = MockTts。VAD は本物(silero)。
        パイプライン自体のオーバーヘッドと VAD 待ちを測る。モデル DL 不要。
        --wav 省略時は合成した「音声っぽい」信号を使う(silero が発話と判定する)。
  real: ASR = 実エンジン(--asr-engine)、TTS = 実 Irodori(--mock-tts でモックに置換可)。

例:
  uv run python scripts/bench_latency.py --mode mock
  uv run python scripts/bench_latency.py --mode real --mock-tts --asr-engine kotoba --wav a.wav --wav b.wav
  uv run python scripts/bench_latency.py --mode real --asr-engine reazonspeech --wav a.wav --repeat 3

指標(すべて ms。backend の time.monotonic() 基準で算出):
  vad_wait        話し終わり → VAD が発話終了を確定するまで(≒ vad_min_silence_ms)
  asr_final       確定デコード時間
  tts_first_chunk 発話受付 → 先頭チャンク送出(キュー待ち + 合成)
  e2e             話し終わり → 先頭チャンク送出(GUI ヘッダの「発話終了→初音」の backend 分)
  chunk_gen       チャンクごとの合成時間 / rtf = 合成時間 / 音声長
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"


def synth_speechlike_wavs(out_dir: Path, seconds=(1.2, 1.8, 2.4, 1.5, 2.0)) -> list[str]:
    """silero VAD が発話と判定する合成信号の WAV を作る(sttts_server.testsignal)。"""
    import soundfile as sf

    sys.path.insert(0, str(BACKEND_SRC))
    from sttts_server.testsignal import SAMPLE_RATE, speechlike

    paths = []
    for k, sec in enumerate(seconds):
        p = out_dir / f"speechlike_{k}.wav"
        sf.write(p, speechlike(sec, seed=k), SAMPLE_RATE)
        paths.append(str(p))
    return paths


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_once(args, wavs: list[str]) -> dict:
    cmd = [args.python, "-m", "sttts_server", "--stdio", "--no-save-wav", "--output-dir", tempfile.gettempdir()]
    for w in wavs:
        cmd += ["--input-wav", w]
    if args.mode == "mock" or args.mock_tts:
        cmd.append("--mock-tts")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    stderr = open(args.stderr_log, "a", encoding="utf-8") if args.stderr_log else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
        encoding="utf-8",
        env=env,
    )
    msgs: queue.Queue = queue.Queue()

    def reader():
        for line in proc.stdout:
            line = line.strip()
            if line:
                try:
                    msgs.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        msgs.put(None)

    threading.Thread(target=reader, daemon=True).start()

    def send(m):
        proc.stdin.write(json.dumps(m, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    asr = {"partial_interval_ms": args.partial_interval_ms, "vad_min_silence_ms": args.vad_min_silence_ms}
    tts = {"model": args.tts_model, "warmup": not args.no_warmup}
    if args.mode == "mock":
        asr.update({"engine": "mock", "mock_latency_ms": args.mock_asr_ms})
    else:
        asr.update({"engine": args.asr_engine, "device": args.asr_device})
        if args.reazon_model_dir:
            asr["reazon_model_dir"] = args.reazon_model_dir
    if args.mode == "mock" or args.mock_tts:
        tts.update({"mock_rtf": args.mock_tts_rtf, "mock_delay_ms": args.mock_tts_delay_ms})
    pipeline = {"auto_speak": True, "speculative_tts": args.speculative}
    if args.first_mora_max is not None:
        pipeline["first_chunk_mora_max"] = args.first_mora_max
    send({"type": "configure", "tts": tts, "asr": asr, "pipeline": pipeline})

    # TTS のウォームアップ/ロード完了を待ってから入力を流す(実機の「起動後に話し始める」に相当)
    deadline = time.time() + args.load_timeout
    started = False
    eof = False
    accepted: set[int] = set()
    done: set[int] = set()
    log_lines = []
    records = {"asr_final": [], "tts_audio": [], "speak_accepted": [], "asr_partial": []}
    tts_ready = args.no_warmup
    while time.time() < deadline:
        try:
            m = msgs.get(timeout=0.5)
        except queue.Empty:
            m = {}
        if m is None:
            break
        t = m.get("type")
        if t == "log":
            log_lines.append(m.get("message", ""))
            if "ウォームアップ" in m.get("message", ""):
                tts_ready = True
            if m.get("message") == "input_eof":
                eof = True
                deadline = time.time() + args.drain_timeout
        elif t == "error":
            print(f"[backend error] {m.get('scope')}: {m.get('message')}", file=sys.stderr)
        elif t in records:
            records[t].append(m)
            if t == "speak_accepted":
                accepted.add(m["request"])
        elif t == "speak_done":
            done.add(m["request"])
        if tts_ready and not started:
            send({"type": "start_session"})
            started = True
            deadline = time.time() + args.load_timeout + sum(_dur(w) for w in wavs) + 30
        if eof and accepted and accepted <= done and time.time() > 0:
            # 最後の発話の後に追加の final が来ないことを少し待って確認
            time.sleep(0.3)
            if msgs.empty():
                break
    try:
        send({"type": "shutdown"})
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    if not eof:
        print("[warn] input_eof を受信できませんでした(タイムアウト)", file=sys.stderr)
    return records


def _dur(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate + 2.5


def summarize(all_records: list[dict]) -> dict:
    vals: dict[str, list[float]] = {
        k: [] for k in ("vad_wait", "asr_final", "asr_partial", "tts_first_chunk", "e2e", "chunk_gen", "rtf", "queue_wait")
    }
    stages: dict[str, list[float]] = {}
    texts = []
    spec_hits = 0
    for rec in all_records:
        for m in rec["asr_final"]:
            if m.get("vad_wait_ms") is not None:
                vals["vad_wait"].append(m["vad_wait_ms"])
            if m.get("asr_ms") is not None:
                vals["asr_final"].append(m["asr_ms"])
            texts.append(m.get("text", ""))
        for m in rec["asr_partial"]:
            if m.get("asr_ms") is not None:
                vals["asr_partial"].append(m["asr_ms"])
        for m in rec["tts_audio"]:
            vals["chunk_gen"].append(m["gen_ms"])
            if m.get("rtf") is not None:
                vals["rtf"].append(m["rtf"])
            if m.get("queue_wait_ms") is not None:
                vals["queue_wait"].append(m["queue_wait_ms"])
            if m.get("chunk") == 0:
                if m.get("first_chunk_ms") is not None:
                    vals["tts_first_chunk"].append(m["first_chunk_ms"])
                if m.get("e2e_ms") is not None:
                    vals["e2e"].append(m["e2e_ms"])
                spec_hits += 1 if m.get("speculative") else 0
            for k, v in (m.get("stages") or {}).items():
                stages.setdefault(k, []).append(v)
    out = {k: {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9), "mean": statistics.fmean(v) if v else float("nan")} for k, v in vals.items()}
    out["stages"] = {k: {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9)} for k, v in stages.items()}
    out["speculative_hits"] = spec_hits
    out["asr_texts"] = texts
    return out


def print_table(summary: dict) -> None:
    print(f"{'metric':<18}{'n':>5}{'P50':>10}{'P90':>10}{'mean':>10}")
    for k in ("vad_wait", "asr_partial", "asr_final", "queue_wait", "tts_first_chunk", "chunk_gen", "rtf", "e2e"):
        s = summary[k]
        fmt = "{:>10.3f}" if k == "rtf" else "{:>10.0f}"
        print(f"{k:<18}{s['n']:>5}" + (fmt * 3).format(s["p50"], s["p90"], s["mean"]))
    if summary["stages"]:
        print("irodori stages (ms):")
        for k, s in summary["stages"].items():
            print(f"  {k:<24}{s['n']:>5}{s['p50']:>10.1f}{s['p90']:>10.1f}")
    print(f"speculative chunk0 hits: {summary['speculative_hits']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("mock", "real"), default="mock")
    p.add_argument("--wav", action="append", default=[], help="入力 WAV(複数可)。mock で省略時は合成信号")
    p.add_argument("--repeat", type=int, default=1, help="同じ入力を何回流すか(backend は毎回起動し直す)")
    p.add_argument("--python", default=sys.executable, help="backend を動かす python")
    p.add_argument("--asr-engine", default="kotoba", choices=("kotoba", "reazonspeech"))
    p.add_argument("--asr-device", default="auto")
    p.add_argument("--reazon-model-dir", default=None)
    p.add_argument("--tts-model", default="v4.1-small-mf")
    p.add_argument("--mock-tts", action="store_true", help="real モードでも TTS をモックにする")
    p.add_argument("--mock-asr-ms", type=int, default=0, help="mock ASR の擬似デコード時間")
    p.add_argument("--mock-tts-rtf", type=float, default=0.0, help="mock TTS の擬似 RTF")
    p.add_argument("--mock-tts-delay-ms", type=float, default=50.0, help="mock TTS の固定合成時間")
    p.add_argument("--partial-interval-ms", type=int, default=800)
    p.add_argument("--vad-min-silence-ms", type=int, default=280)
    p.add_argument("--first-mora-max", type=int, default=None)
    p.add_argument("--speculative", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--load-timeout", type=float, default=900.0)
    p.add_argument("--drain-timeout", type=float, default=120.0)
    p.add_argument("--stderr-log", default=None, help="backend の stderr をこのファイルへ追記")
    p.add_argument("--json", default=None, help="結果 JSON の保存先")
    args = p.parse_args(argv)

    tmp = None
    wavs = list(args.wav)
    if not wavs:
        if args.mode == "real":
            p.error("real モードでは --wav が必要です")
        tmp = tempfile.TemporaryDirectory()
        wavs = synth_speechlike_wavs(Path(tmp.name))

    all_records = []
    for i in range(args.repeat):
        t0 = time.time()
        rec = run_once(args, wavs)
        all_records.append(rec)
        print(f"run {i + 1}/{args.repeat}: finals={len(rec['asr_final'])} chunks={len(rec['tts_audio'])} ({time.time() - t0:.1f}s)", file=sys.stderr)
    summary = summarize(all_records)
    summary["config"] = {k: v for k, v in vars(args).items() if k not in ("python",)}
    print_table(summary)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
