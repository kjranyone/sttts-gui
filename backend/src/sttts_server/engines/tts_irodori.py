"""Irodori-TTS InferenceRuntime ラッパ(実エンジン)。

重い import は load()/synthesize() 内で遅延行うため、モックモードや
protocol/chunker のテストでは本モジュールは import されない。

オーバーヘッド削減(Irodori 本体は改変しない):
- precision "auto": CUDA の compute capability < 8.0(Turing: RTX 20xx 等)は bf16 の
  ハード支援が無いため fp32、Ampere 以降と XPU は bf16、CPU/MPS は fp32。
- 参照音声 latent キャッシュ: 参照 WAV を毎回 DACVAE でエンコードし直す代わりに
  初回のエンコード結果を .pt に保存し、ref_latents として渡す(Irodori の
  _load_reference_latent と同じ前処理: 単一 WAV の max_ref_seconds トリム / -16dB 正規化)。
- 条件エンコードのメモ化: text/caption/speaker エンコーダの forward を入力テンソルの
  内容をキーに LRU キャッシュする。1回の合成で encode_conditions が2回(尺予測と
  サンプラ)呼ばれる重複と、同じ caption / 参照話者の再計算を省く。
  ※ encode_conditions 自体の二重呼び出しをなくすには Irodori 本体の改変が必要なため対象外。
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

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

# 参照音声の前処理(SamplingRequest の既定値と同じ)
REF_NORMALIZE_DB = -16.0
REF_ENSURE_MAX = True
REF_CACHE_VERSION = 1

log = logging.getLogger("sttts.tts")


def resolve_precision(device: str, requested: str | None, cuda_capability: tuple[int, int] | None = None) -> str:
    """モデル precision を決める。Irodori は fp32 / bf16 のみ対応(bf16 は cuda/xpu のみ)。"""
    req = (requested or "auto").lower()
    if device not in ("cuda", "xpu"):
        return "fp32"
    if req in ("fp32", "bf16"):
        return req
    if device == "cuda" and cuda_capability is not None and tuple(cuda_capability) < (8, 0):
        return "fp32"  # Turing 以前は bf16 演算が遅い/非対応
    return "bf16"


def default_ref_cache_dir() -> Path:
    base = os.environ.get("STTTS_CACHE_DIR")
    if base:
        return Path(base) / "ref_latents"
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "sttts-gui" / "cache" / "ref_latents"
    return Path.home() / ".cache" / "sttts-gui" / "ref_latents"


def ref_cache_key(path: str, *, codec_repo: str, trim_seconds: float | None) -> str:
    st = os.stat(path)
    raw = "|".join(
        str(x)
        for x in (
            REF_CACHE_VERSION,
            os.path.abspath(path),
            st.st_size,
            st.st_mtime_ns,
            codec_repo,
            REF_NORMALIZE_DB,
            REF_ENSURE_MAX,
            trim_seconds,
        )
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _tensor_digest(t) -> tuple:
    import torch  # noqa: PLC0415

    data = t.detach()
    if data.device.type != "cpu":
        data = data.to("cpu")
    data = data.contiguous()
    raw = data.view(torch.uint8).numpy().tobytes() if data.numel() else b""
    return ("T", str(t.dtype), tuple(t.shape), t.device.type, hashlib.blake2b(raw, digest_size=16).digest())


class ModuleMemo:
    """nn.Module の forward を入力内容キーの LRU でメモ化する(推論専用・Irodori 非改変)。

    出力テンソルはキャッシュから同じオブジェクトを返す。Irodori の encode_conditions は
    出力をインプレース変更しない(マスク変更は clone 後に行う)ことを確認済み(89f9d8f)。
    """

    def __init__(self, module, name: str, maxsize: int = 8) -> None:
        self.module = module
        self.name = name
        self.maxsize = maxsize
        self.orig_forward = module.forward
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _key(self, args, kwargs) -> tuple:
        import torch  # noqa: PLC0415

        def one(v):
            if isinstance(v, torch.Tensor):
                return _tensor_digest(v)
            if isinstance(v, torch.nn.Module):
                return ("M", id(v))
            if v is None or isinstance(v, (bool, int, float, str)):
                return ("V", v)
            return ("R", repr(v))

        return (
            tuple(one(a) for a in args),
            tuple((k, one(v)) for k, v in sorted(kwargs.items())),
            self.module.training,
            torch.is_grad_enabled(),
        )

    def __call__(self, *args, **kwargs):
        key = self._key(args, kwargs)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
        out = self.orig_forward(*args, **kwargs)
        with self._lock:
            self.misses += 1
            self.cache[key] = out
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
        return out

    def install(self) -> "ModuleMemo":
        self.module.forward = self
        return self

    def uninstall(self) -> None:
        if self.module.__dict__.get("forward") is self:
            del self.module.forward
        self.cache.clear()


class IrodoriTts:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        num_steps: int | None = None,
        decode_mode: str = "sequential",
        precision: str = "auto",
        compile_model: bool = False,
        cache_conditions: bool = True,
        ref_latent_cache: bool = True,
        ref_cache_dir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.num_steps = num_steps
        self.decode_mode = decode_mode
        self.precision = precision
        self.compile_model = bool(compile_model)
        self.cache_conditions = bool(cache_conditions) and not self.compile_model
        self.ref_latent_cache = bool(ref_latent_cache)
        self.ref_cache_dir = Path(ref_cache_dir) if ref_cache_dir else default_ref_cache_dir()
        self._runtime = None
        self._device_resolved: str | None = None
        self.precision_resolved: str | None = None
        self._synth_count = 0
        self._memos: list[ModuleMemo] = []
        self._ref_mem: dict[str, str] = {}

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

        checkpoint, _catalog_precision = CATALOG.get(self.model_id, (self.model_id, "bf16"))
        device = self.device
        if device in (None, "", "auto"):
            device = default_runtime_device()
        capability = None
        if device == "cuda":
            import torch  # noqa: PLC0415

            try:
                capability = tuple(torch.cuda.get_device_capability())
            except Exception:  # pragma: no cover - 環境依存
                capability = None
        model_precision = resolve_precision(device, self.precision, capability)

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
                compile_model=self.compile_model,
                compile_dynamic=self.compile_model,
            )
        )
        self._device_resolved = device
        self.precision_resolved = model_precision
        if self.cache_conditions:
            self._install_condition_memos()
        cap = f" sm_{capability[0]}{capability[1]}" if capability else ""
        report(f"ロード完了: {checkpoint} @ {device}{cap}/{model_precision}")

    def _install_condition_memos(self) -> None:
        model = getattr(self._runtime, "model", None)
        if model is None:
            return
        for name in ("text_encoder", "caption_encoder", "speaker_encoder"):
            module = getattr(model, name, None)
            if module is not None and hasattr(module, "forward"):
                self._memos.append(ModuleMemo(module, name).install())

    def memo_stats(self) -> dict[str, tuple[int, int]]:
        return {m.name: (m.hits, m.misses) for m in self._memos}

    def _ref_latent_paths(self, ref_wavs: list[str]) -> list[str]:
        """参照 WAV を DACVAE latent(.pt)に変換してキャッシュし、そのパスを返す。"""
        import torch  # noqa: PLC0415
        from irodori_tts.inference_runtime import _load_audio  # noqa: PLC0415

        rt = self._runtime
        max_ref_seconds = float(getattr(rt, "default_max_ref_seconds", 0) or 0)
        trim = max_ref_seconds if (len(ref_wavs) == 1 and max_ref_seconds > 0) else None
        self.ref_cache_dir.mkdir(parents=True, exist_ok=True)
        out: list[str] = []
        for path in ref_wavs:
            key = ref_cache_key(path, codec_repo=CODEC_REPO, trim_seconds=trim)
            cached = self.ref_cache_dir / f"{key}.pt"
            if not cached.is_file():
                wav, sr = _load_audio(path)
                if trim is not None:
                    max_samples = max(1, int(trim * float(sr)))
                    wav = wav[:, :max_samples]
                with torch.inference_mode():
                    piece = rt.codec.encode_waveform(
                        wav.unsqueeze(0),
                        sample_rate=int(sr),
                        normalize_db=REF_NORMALIZE_DB,
                        ensure_max=REF_ENSURE_MAX,
                    ).cpu()
                if piece.shape[1] == 0:
                    raise ValueError(f"Reference waveform produced an empty latent: {path}")
                tmp = cached.with_suffix(".tmp")
                torch.save(piece[0].clone(), tmp)
                os.replace(tmp, cached)
                log.info("ref latent cached: %s -> %s %s", path, cached.name, tuple(piece.shape))
            out.append(str(cached))
        return out

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

        t0 = time.perf_counter()
        ref_latents = None
        ref_ms = 0.0
        if ref_wavs and self.ref_latent_cache:
            try:
                r0 = time.perf_counter()
                ref_latents = self._ref_latent_paths(list(ref_wavs))
                ref_ms = (time.perf_counter() - r0) * 1000.0
            except Exception:
                log.exception("ref latent cache failed; falling back to ref_wavs")
                ref_latents = None
        no_ref = not ref_wavs and not caption
        req = SamplingRequest(
            text=text,
            caption=caption,
            ref_wavs=None if ref_latents else (list(ref_wavs) if ref_wavs else None),
            ref_latents=ref_latents,
            ref_normalize_db=REF_NORMALIZE_DB,
            ref_ensure_max=REF_ENSURE_MAX,
            no_ref=no_ref,
            num_candidates=1,
            decode_mode=self.decode_mode,  # 低VRAM既定
            num_steps=self.num_steps,  # None で checkpoint 既定(MF:4 / RF:40)
            seed=seed,
        )
        try:
            result = self._runtime.synthesize(req)
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                "GPUメモリ不足(INT8モデル切替 / codec を CPU 化 / sequential デコードを試してください)"
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

        stages = {name: round(sec * 1000.0, 1) for name, sec in (result.stage_timings or [])}
        if ref_latents:
            stages["ref_latent_cache"] = round(ref_ms, 1)
        duration_ms = int(audio_np.shape[-1] / sample_rate * 1000) if audio_np.size else 0
        return SynthResult(
            wav_bytes=buf.getvalue(),
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            gen_ms=gen_ms,
            used_seed=result.used_seed,
            stages=stages,
        )

    def unload(self) -> None:
        for memo in self._memos:
            memo.uninstall()
        self._memos = []
        if self._runtime is not None:
            self._runtime.unload()
            self._runtime = None
