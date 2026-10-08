"""Irodori-TTS InferenceRuntime ラッパ(実エンジン)。

重い import は load()/synthesize() 内で遅延行うため、モックモードや
protocol/chunker のテストでは本モジュールは import されない。
"""

from __future__ import annotations

import io
import time

from ..app import SynthResult

# モデルエイリアス → (checkpoint指定, 既定precision)
# checkpoint は repo_id または repo_id/subfolder
CATALOG: dict[str, tuple[str, str]] = {
    "v4.1-small-mf": ("Aratako/Irodori-TTS-v4.1-Small-MF", "bf16"),
    "v4.1-small": ("Aratako/Irodori-TTS-v4.1-Small", "bf16"),
    "v4.1-small-int8": (
        "Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only",
        "bf16",
    ),
    "v4-large": ("Aratako/Irodori-TTS-v4-Large", "bf16"),
    "v4-large-int8": (
        "Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only",
        "bf16",
    ),
}

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"

# 空きメモリ確保のため N 回ごとに empty_cache を呼ぶ(B570 10GB 前提の低VRAM対策)
EMPTY_CACHE_INTERVAL = 8


class IrodoriTts:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        num_steps: int | None = None,
        decode_mode: str = "sequential",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.num_steps = num_steps
        self.decode_mode = decode_mode
        self._runtime = None
        self._device_resolved: str | None = None
        self._synth_count = 0

    def load(self, progress=None) -> None:
        def report(message: str, frac: float | None = None) -> None:
            if progress is not None:
                progress(message, frac)

        from irodori_tts.inference_runtime import (  # noqa: PLC0415
            InferenceRuntime,
            RuntimeKey,
            default_runtime_device,
            download_hf_checkpoint,
        )

        checkpoint, precision = CATALOG.get(self.model_id, (self.model_id, "bf16"))
        device = self.device
        if device in (None, "", "auto"):
            device = default_runtime_device()
        # bf16 は CUDA/XPU のみ。CPU フォールバック時は fp32。
        model_precision = precision if device in ("cuda", "xpu") else "fp32"

        report(f"checkpoint 取得: {checkpoint}")
        checkpoint_path = download_hf_checkpoint(checkpoint)
        report("ランタイム構築中(初回は数分)", None)
        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=device,
                codec_repo=CODEC_REPO,
                model_precision=model_precision,
                codec_device=device,
                codec_precision="fp32",
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
                compile_model=False,
                compile_dynamic=False,
            )
        )
        self._device_resolved = device
        report(f"ロード完了: {checkpoint} @ {device}/{model_precision}")

    def synthesize(
        self,
        text: str,
        *,
        caption: str | None = None,
        ref_wavs: list[str] | None = None,
        seed: int | None = None,
        progress=None,
    ) -> SynthResult:
        import soundfile as sf  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from irodori_tts.inference_runtime import SamplingRequest  # noqa: PLC0415

        if self._runtime is None:
            raise RuntimeError("engine not loaded")

        no_ref = not ref_wavs and not caption
        req = SamplingRequest(
            text=text,
            caption=caption,
            ref_wavs=list(ref_wavs) if ref_wavs else None,
            no_ref=no_ref,
            num_candidates=1,
            decode_mode=self.decode_mode,  # 低VRAM既定
            num_steps=self.num_steps,  # None で checkpoint 既定(MF:4 / RF:40)
            seed=seed,
        )
        t0 = time.perf_counter()
        try:
            result = self._runtime.synthesize(req)
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                "XPUメモリ不足(INT8モデル切替 / codec を CPU 化 / sequential デコードを試してください)"
            ) from e
        gen_ms = int((time.perf_counter() - t0) * 1000)

        # result.audio は (1, N) のことがあるため 1次元に潰す(モノラル前提)
        audio_np = result.audio.detach().to("cpu").float().numpy().reshape(-1)
        sample_rate = int(result.sample_rate)
        buf = io.BytesIO()
        sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")

        self._synth_count += 1
        if self._synth_count % EMPTY_CACHE_INTERVAL == 0:
            if self._device_resolved == "xpu":
                torch.xpu.empty_cache()
            elif self._device_resolved == "cuda":
                torch.cuda.empty_cache()

        duration_ms = int(audio_np.shape[-1] / sample_rate * 1000) if audio_np.size else 0
        return SynthResult(
            wav_bytes=buf.getvalue(),
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            gen_ms=gen_ms,
            used_seed=result.used_seed,
        )

    def unload(self) -> None:
        if self._runtime is not None:
            self._runtime.unload()
            self._runtime = None
