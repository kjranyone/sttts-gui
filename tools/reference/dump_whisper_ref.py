"""kotoba-whisper-v2.0 の参照出力を CPU・fp32 で書き出す(純 Rust 実装 crates/whisper の数値一致テスト用)。

PyTorch の実行は CPU のみ(XPU / GPU には触れない)。音声は output/*.wav と data/voices/* を 16kHz に変換して
連結し、1 / 10 / 25 / 35 秒(30 秒超)のケースを作る。各ケースについて、ログメル特徴・encoder 出力・
decoder の最初の数ステップの logits・greedy / beam(2) の転写を保存する(30 秒を超える音声は、
Rust 実装と同じく 30 秒ずつ順次デコードして連結する)。

使い方:  cd tools/reference && uv run python dump_whisper_ref.py [出力ディレクトリ]
既定の出力先: <repo>/target/whisper-ref  (refs.safetensors, meta.json)
"""

from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

REPO = "kotoba-tech/kotoba-whisper-v2.0"
SR = 16000
WINDOW = 30 * SR
CASE_SECONDS = {"s1": 1, "s10": 10, "s25": 25, "s35": 35}
LOGIT_STEPS = 6


def load_16k(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def build_pool(root: Path) -> np.ndarray:
    files = sorted((root / "output").glob("*.wav")) + sorted((root / "data" / "voices").rglob("*.wav"))
    parts = []
    for f in files:
        try:
            parts.append(load_16k(f))
        except Exception as e:  # 壊れたファイルは飛ばす
            print("skip", f, e)
        if sum(len(p) for p in parts) > 40 * SR:
            break
    if not parts:
        sys.exit("音声が見つかりません(output/*.wav, data/voices)")
    pool = np.concatenate(parts)
    while len(pool) < 36 * SR:  # 足りなければ繰り返す
        pool = np.concatenate([pool, pool])
    return pool


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "target" / "whisper-ref"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = snapshot_download(REPO)
    fe = WhisperFeatureExtractor.from_pretrained(ckpt)
    tok = WhisperTokenizerFast.from_pretrained(ckpt)
    model = WhisperForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
    prompt = [tok.convert_tokens_to_ids(t) for t in ("<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>")]

    pool = build_pool(root)
    tensors: dict[str, torch.Tensor] = {}
    meta: dict = {"prompt": prompt, "cases": {}}

    def transcribe(window: np.ndarray, beams: int) -> tuple[str, list[int]]:
        feats = fe(window, sampling_rate=SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(
                feats, language="ja", task="transcribe", return_timestamps=False,
                num_beams=beams, do_sample=False, max_new_tokens=440,
            )
        ids = ids[0].tolist()
        return tok.decode(ids, skip_special_tokens=True).strip(), ids

    for name, secs in CASE_SECONDS.items():
        # ケースごとに別の位置から切り出す(同じ音声ばかりにしない)
        start = list(CASE_SECONDS).index(name) * 3 * SR
        audio = np.ascontiguousarray(pool[start : start + secs * SR]).astype(np.float32)
        if len(audio) < secs * SR:
            audio = np.ascontiguousarray(pool[: secs * SR])
        tensors[f"{name}.audio"] = torch.from_numpy(audio)
        wins = [audio[i : i + WINDOW] for i in range(0, len(audio), WINDOW)]

        feats0 = fe(wins[0], sampling_rate=SR, return_tensors="pt").input_features
        tensors[f"{name}.mel"] = feats0[0].contiguous()
        with torch.no_grad():
            enc = model.model.encoder(feats0).last_hidden_state
        tensors[f"{name}.enc"] = enc[0].contiguous()

        greedy_texts, beam_texts, greedy_ids0 = [], [], None
        for k, w in enumerate(wins):
            t, ids = transcribe(w, 1)
            greedy_texts.append(t)
            if k == 0:
                greedy_ids0 = ids
            beam_texts.append(transcribe(w, 2)[0])

        # 先頭窓の greedy 出力で teacher forcing し、最初の数ステップの logits を保存する
        gen = [i for i in greedy_ids0 if i not in prompt]
        seq = prompt + gen[: LOGIT_STEPS - 1]
        with torch.no_grad():
            lg = model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([seq])).logits[0]
        tensors[f"{name}.logits"] = lg[len(prompt) - 1 : len(prompt) - 1 + LOGIT_STEPS].contiguous()
        meta["cases"][name] = {
            "seconds": secs,
            "greedy": "".join(greedy_texts),
            "beam2": "".join(beam_texts),
            "greedy_ids0": greedy_ids0,
        }
        print(name, meta["cases"][name]["greedy"][:60], "|", meta["cases"][name]["beam2"][:60])

    save_file(tensors, out_dir / "refs.safetensors")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
