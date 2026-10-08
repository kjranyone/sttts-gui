"""テスト・ベンチ用の合成「音声っぽい」信号(silero VAD が発話と判定する)。

声帯パルス列(f0 ≈ 140Hz、揺らぎあり)を 180ms ごとに母音フォルマントの異なる
共振器に通す。実音声ではないので ASR には使えない(VAD / 配線の検証専用)。
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
_VOWELS = [(800, 1200), (300, 2300), (500, 1800), (350, 800), (600, 1000)]


def _resonator(x: np.ndarray, f: float, bw: float) -> np.ndarray:
    from scipy.signal import lfilter  # noqa: PLC0415

    r = np.exp(-np.pi * bw / SAMPLE_RATE)
    th = 2 * np.pi * f / SAMPLE_RATE
    return lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 5 * t)
    src = (np.sin(np.cumsum(2 * np.pi * f0 / SAMPLE_RATE)) > 0.95).astype(float) - 0.05
    seg = int(0.18 * SAMPLE_RATE)
    out = np.zeros(n)
    for i in range(0, n, seg):
        f1, f2 = _VOWELS[rng.integers(len(_VOWELS))]
        s = src[i : i + seg]
        y = _resonator(s, f1, 90) + 0.6 * _resonator(s, f2, 120) + 0.2 * _resonator(s, 2800, 200)
        out[i : i + seg] = y * np.hanning(len(y)) ** 0.5
    out += 0.003 * rng.standard_normal(n)
    return (0.3 * out / np.max(np.abs(out))).astype(np.float32)
