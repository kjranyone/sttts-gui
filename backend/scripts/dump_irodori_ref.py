"""Irodori-TTS の参照出力を CPU で書き出す(純 Rust 実装の数値一致テスト用)。

PyTorch の実行は CPU・fp32 のみ(XPU / GPU には触れない)。各ケースについて、推論の途中経過
(トークナイザ、ModernBERT、条件エンコーダ、長さ予測、DiT の各ステップ、codec のデコード、透かし)を
safetensors に保存する。Rust 側のテストはこれを読んで段階ごとに突き合わせる。

使い方:  cd backend && uv run --no-sync python scripts/dump_irodori_ref.py [出力ディレクトリ]
既定の出力先: ../target/irodori-ref
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

REPO = "Aratako/Irodori-TTS-v4.1-Small-MF"
TEXT = "こんにちは、よろしくお願いします。"
LONG_TEXT = "今日は朝から小雨が降っていましたが、午後にはすっかり晴れて、夕方の空がとてもきれいでした。"
CAPTION = "落ち着いた女性の声で、ゆっくり話す。"

CASES = {
    # name: SamplingRequest kwargs
    "A": dict(text=TEXT, no_ref=True, seed=0),
    "B": dict(text=TEXT, caption=CAPTION, no_ref=True, seed=1),
    "C": dict(text=TEXT, no_ref=False, seed=2),  # ref_wav は後で差し込む
    "D": dict(text=LONG_TEXT, no_ref=True, seed=3),
}


def make_ref_wav(path: Path, sr: int = 24000, seconds: float = 3.0) -> None:
    """再現可能な合成「声っぽい」信号(基本周波数が揺れる倍音 + 少量のノイズ)。"""
    rng = np.random.default_rng(1234)
    t = np.arange(int(sr * seconds)) / sr
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 12))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t) ** 2
    sig = 0.2 * sig * env + 0.002 * rng.standard_normal(t.shape)
    sf.write(path, sig.astype(np.float32), sr, subtype="PCM_16")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "target" / "irodori-ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    ckpt_dir = Path(snapshot_download(REPO))
    checkpoint = str(ckpt_dir / "model.safetensors")
    rt = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint,
            model_device="cpu",
            model_precision="fp32",
            codec_device="cpu",
            codec_precision="fp32",
        )
    )
    print("sample_rate", rt.codec.sample_rate, "hop", rt.codec.model.hop_length, flush=True)

    ref_wav = out_dir / "ref.wav"
    make_ref_wav(ref_wav)

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {"repo": REPO, "cases": {}}
    state = {"case": "", "calls": {}}

    def put(stage: str, value) -> None:
        n = state["calls"].get(stage, 0)
        state["calls"][stage] = n + 1
        key = f"{state['case']}.{stage}.{n}"
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().to("cpu").contiguous().clone()

    def wrap(obj, name: str, stage: str, pack_args=None):
        orig = getattr(obj, name)

        def inner(*args, **kwargs):
            if pack_args:
                pack_args(stage, args, kwargs)
            out = orig(*args, **kwargs)
            if isinstance(out, (tuple, list)):
                for i, o in enumerate(out):
                    put(f"{stage}.out{i}", o) if isinstance(o, torch.Tensor) else None
            else:
                put(f"{stage}.out", out)
            return out

        setattr(obj, name, inner)

    def pack_tok(stage, args, kwargs):
        meta["cases"][state["case"]].setdefault("tokenizer_calls", []).append(
            {"stage": stage, "texts": list(args[0]), "max_length": kwargs.get("max_length")}
        )

    def pack_dit(stage, args, kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                put(f"{stage}.in.{k}", v)

    def pack_kv(stage, args, kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                put(f"{stage}.in.{k}", v)

    wrap(rt.tokenizer, "batch_encode", "tok_text", pack_tok)
    if rt.caption_tokenizer is not None:
        wrap(rt.caption_tokenizer, "batch_encode", "tok_caption", pack_tok)
    wrap(rt.model, "encode_conditions", "encode_conditions")
    wrap(rt.model, "forward_with_encoded_conditions", "dit", pack_dit)
    wrap(rt.model, "predict_duration_log_frames", "duration", pack_dit)
    wrap(rt.model, "build_context_kv_cache", "kv_cache")
    wrap(rt.codec, "decode_latent", "codec_decode", lambda s, a, k: put("codec_decode.in", a[0]))
    wrap(rt.codec, "encode_waveform", "codec_encode", lambda s, a, k: put("codec_encode.in", a[0]))
    if rt.watermarker.ready:
        orig_wm = rt.watermarker.encode_batch

        def wm(audios, sample_rate):
            for i, a in enumerate(audios):
                put(f"watermark.in{i}", a)
            out = orig_wm(audios, sample_rate=sample_rate)
            for i, a in enumerate(out):
                put(f"watermark.out{i}", a)
            return out

        rt.watermarker.encode_batch = wm

    # ModernBERT(テキスト・キャプション共用の backbone)の入出力
    def backbone_hook(module, args, kwargs, output):
        put("backbone.in.ids", args[0])
        put("backbone.in.mask", args[1])
        put("backbone.out", output)

    rt.model.pretrained_text_backbone.register_forward_hook(backbone_hook, with_kwargs=True)

    for name, kw in CASES.items():
        state["case"] = name
        state["calls"] = {}
        meta["cases"][name] = {"request": {k: v for k, v in kw.items()}}
        kw = dict(kw)
        if kw.get("no_ref") is False:
            kw["ref_wav"] = str(ref_wav)
            meta["cases"][name]["request"]["ref_wav"] = "ref.wav"
        req = SamplingRequest(**kw)
        with torch.inference_mode():
            res = rt.synthesize(req)
        put("final_audio", res.audio)
        meta["cases"][name]["sample_rate"] = int(res.sample_rate)
        meta["cases"][name]["messages"] = list(res.messages)
        meta["cases"][name]["stage_timings"] = [(n, float(s)) for n, s in res.stage_timings]
        print(name, {k: v for k, v in state["calls"].items()}, flush=True)

    save_file(tensors, str(out_dir / "refs.safetensors"))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_dir, len(tensors), "tensors", flush=True)


if __name__ == "__main__":
    main()
