"""Irodori-TTS の参照出力を CPU で書き出す(純 Rust 実装の数値一致テスト用)。

PyTorch の実行は CPU・fp32 のみ(XPU / GPU には触れない)。各ケースについて、推論の途中経過
(トークナイザ、ModernBERT、条件エンコーダ、長さ予測、DiT の各ステップ、codec のデコード、透かし)を
safetensors に保存する。Rust 側のテストはこれを読んで段階ごとに突き合わせる。

使い方:  cd tools/reference && uv run python dump_irodori_ref.py [--rf | --int8 | --large | --large-int8] [出力ディレクトリ]
既定の出力先は <repo>/target/ の下:
  (なし)        irodori-ref            v4.1 Small MeanFlow
  --rf          irodori-ref-rf         v4.1 Small(RF + CFG)
  --int8        irodori-ref-int8       v4.1 Small の int8 weight-only
  --large       irodori-ref-large      v4 Large(RF + CFG、T5Gemma 2。重み 13GB、RAM 40GB 程度)
  --large-int8  irodori-ref-large-int8 v4 Large の int8 weight-only

int8 は torchao を使わず、int8 の重みを `qdata * scale` で fp32 に戻したチェックポイントを作って fp32 で動かす
(Rust 版も同じく、int8 の重みを fp32 に戻して fp32 で計算する。PyTorch の量子化版は bf16 で計算するので比べない)。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
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

# RF(v4.1 Small、CFG あり)。Rust の CPU テストが現実的な時間で済むよう、ステップ数は少なめにして
# CFG のかけ方と RF サンプラの各オプションを一通り通す。
RF_REPO = "Aratako/Irodori-TTS-v4.1-Small"
RF_CASES = {
    # text CFG のみ
    "RA": dict(text=TEXT, no_ref=True, seed=10, num_steps=8),
    # text + caption(independent: 3 本を 1 バッチ)
    "RB": dict(text=TEXT, caption=CAPTION, no_ref=True, seed=11, num_steps=8),
    # text + speaker
    "RC": dict(text=TEXT, no_ref=False, seed=12, num_steps=8),
    # alternating、sway、話者 K/V 強調、score rescale、truncation
    "RD": dict(
        text=TEXT,
        caption=CAPTION,
        no_ref=False,
        seed=13,
        num_steps=6,
        cfg_guidance_mode="alternating",
        t_schedule_mode="sway",
        sway_coeff=-0.5,
        speaker_kv_scale=1.3,
        speaker_kv_min_t=0.7,
        speaker_kv_max_layers=4,
        rescale_k=1.1,
        rescale_sigma=0.5,
        truncation_factor=0.9,
    ),
    # joint(倍率を cfg_scale でそろえる)、CFG をかける範囲を広げる
    "RE": dict(text=TEXT, caption=CAPTION, no_ref=True, seed=14, num_steps=6, cfg_guidance_mode="joint", cfg_scale=2.0, cfg_min_t=0.3),
}

# int8 weight-only(torchao で量子化したもの)。量子化した層(DiT・話者エンコーダ・テキストのバックボーン)を全部通る
INT8_SUBFOLDER = "int8-weight-only"
INT8_CASES = {
    "QA": dict(text=TEXT, caption=CAPTION, no_ref=True, seed=20, num_steps=6),
    "QC": dict(text=TEXT, no_ref=False, seed=21, num_steps=6),
}

# v4 Large(T5Gemma 2 のテキストエンコーダ、話者は 4 フレームずつまとめる)。CPU で重いのでステップは少なく
LARGE_CASES = {
    "LA": dict(text=TEXT, caption=CAPTION, no_ref=True, seed=30, num_steps=4),
    "LC": dict(text=TEXT, no_ref=False, seed=31, num_steps=4),
}

# モード → (リポジトリ, 量子化版のサブフォルダ, ケース, 既定の出力先)
MODES = {
    "mf": (REPO, None, CASES, "irodori-ref"),
    "rf": (RF_REPO, None, RF_CASES, "irodori-ref-rf"),
    "int8": ("Aratako/Irodori-TTS-v4.1-Small-Quantized", INT8_SUBFOLDER, INT8_CASES, "irodori-ref-int8"),
    "large": ("Aratako/Irodori-TTS-v4-Large", None, LARGE_CASES, "irodori-ref-large"),
    "large-int8": ("Aratako/Irodori-TTS-v4-Large-Quantized", INT8_SUBFOLDER, INT8_CASES, "irodori-ref-large-int8"),
}


def dequantized_checkpoint(snapshot: Path, out_dir: Path) -> Path:
    """int8 weight-only の量子化チェックポイントを fp32 に戻して保存する(トークナイザも隣に置く)。"""
    src = snapshot / INT8_SUBFOLDER / "model.safetensors"
    dst_dir = out_dir / "ckpt" / INT8_SUBFOLDER
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.safetensors"
    with safe_open(str(src), framework="pt", device="cpu") as f:
        meta = f.metadata()
        q = json.loads(meta["irodori_quantization_json"])
        assert q["quantization_type"] == "int8_weight_only", q
        state = {}
        for name, desc in meta.items():
            try:
                d = json.loads(desc)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict) or "_type" not in d:
                continue
            if d["_type"] == "Int8Tensor":
                parent, attr = name.rsplit(".", 1)
                qdata = f.get_tensor(f"{parent}._{attr}_qdata")
                scale = f.get_tensor(f"{parent}._{attr}_scale")
                state[name] = qdata.to(torch.float32) * scale.to(torch.float32)
            elif d["_type"] == "Tensor":
                state[name] = f.get_tensor(name).to(torch.float32)
            else:
                raise ValueError(f"unexpected tensor type {d['_type']} for {name}")
    keep = {k: meta[k] for k in ("config_json", "text_encoder_config_json") if k in meta}
    save_file(state, str(dst), metadata=keep)
    shutil.copytree(snapshot / "tokenizer", out_dir / "ckpt" / "tokenizer", dirs_exist_ok=True)
    return dst


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
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    args = [a for a in args if not a.startswith("--")]
    mode = flags[0].removeprefix("--") if flags else "mf"
    if len(flags) > 1 or mode not in MODES:
        raise SystemExit(f"usage: dump_irodori_ref.py [--{' | --'.join(m for m in MODES if m != 'mf')}] [out_dir]")
    repo, subfolder, cases, default_dir = MODES[mode]
    out_dir = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "target" / default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if subfolder is not None:
        snapshot = Path(snapshot_download(repo, allow_patterns=[f"{subfolder}/*", "tokenizer/*"]))
        checkpoint = str(dequantized_checkpoint(snapshot, out_dir))
    else:
        ckpt_dir = Path(snapshot_download(repo))
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
    meta: dict[str, object] = {"repo": repo, "cases": {}}
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

    for name, kw in cases.items():
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
