"""エントリポイント。

使い方:
    uv run python -m sttts_server --stdio [--mock] [--output-dir DIR]
    uv run python -m sttts_server --self-check-tts "テキスト" [--mock]
    uv run python -m sttts_server --self-check-asr
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sttts_server")
    p.add_argument("--stdio", action="store_true", help="NDJSON stdio サーバを起動(既定動作)")
    p.add_argument("--mock", action="store_true", help="モックエンジンを使用(モデルDLなし)")
    p.add_argument("--output-dir", default="output", help="生成WAVの保存先")
    p.add_argument("--self-check-tts", metavar="TEXT", help="TEXT を合成して WAV を保存し結果をJSONで出力して終了")
    p.add_argument("--self-check-asr", action="store_true", help="マイクから5秒録音して文字起こしし結果をJSONで出力して終了")
    p.add_argument("--model", default=None, help="self-check 用モデルエイリアス")
    return p


def _self_check_tts(args: argparse.Namespace) -> int:
    from sttts_server.app import BackendApp  # ProgressFn を共有

    if args.mock:
        from sttts_server.engines.mock import MockTts as Engine
    else:
        from sttts_server.engines.tts_irodori import IrodoriTts as Engine

    engine = Engine(
        model_id=args.model or "v4.1-small-mf",
        device="auto",
        num_steps=None,
        decode_mode="sequential",
    )

    def progress(message, frac=None):
        print(f"[load] {message}" + (f" ({frac * 100:.0f}%)" if frac is not None else ""), file=sys.stderr)

    print("loading engine...", file=sys.stderr)
    engine.load(progress)
    print("synthesizing...", file=sys.stderr)
    result = engine.synthesize(args.self_check_tts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"selfcheck_{uuid.uuid4().hex[:8]}.wav"
    out_path.write_bytes(result.wav_bytes)
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(out_path),
                "sample_rate": result.sample_rate,
                "duration_ms": result.duration_ms,
                "gen_ms": result.gen_ms,
                "used_seed": result.used_seed,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _self_check_asr(args: argparse.Namespace) -> int:
    from sttts_server.selfcheck import self_check_asr

    return self_check_asr(seconds=5.0, model=args.model, mock=args.mock)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.self_check_tts is not None:
        return _self_check_tts(args)
    if args.self_check_asr:
        return _self_check_asr(args)

    from sttts_server.app import BackendApp

    app = BackendApp(mock=args.mock, output_dir=args.output_dir)
    app.run_stdio()
    return 0


if __name__ == "__main__":
    sys.exit(main())
