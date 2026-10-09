"""Nemotron 3.5 ASR(ONNX / onnxruntime CPU)による多言語 ASR(任意エンジン)。

asr.engine = "nemotron" で選択する。onnxruntime は標準依存。
重みは HuggingFace のコミュニティ export(fp16)を自動DL:
    codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx(Apache-2.0 コード / OpenMDW-1.1 重み)
ベースモデル: nvidia/nemotron-3.5-asr-streaming-0.6b(cache-aware FastConformer-RNNT, 600M)。

トレードオフ(kotoba / ReazonSpeech との比較、i5-12600KF 実測):
- 速い: 発話単位の確定が 0.3〜0.6 秒(chunk=320ms fp16 実測。kotoba CPU は 3 秒超)。
  chunk=1120 のグラフ(要 自前export)なら RTF 0.1 台まで下がる。
- 句読点をネイティブ出力する(ReazonSpeech に無い強み。TTS チャンク分割に効く)。
- 精度は whisper large-v3 級(FLEURS ja CER ~12)。
- モデルは fp16 で ~2.5GB(int8 は dynamic quantum で精度劣化するため非推奨)。
- ストリーミング エンジン本体は vendor/nemotron_onnx_streaming.py(Apache-2.0)。
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger("sttts.asr")

DEFAULT_REPO = "codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx"

# HF 公式パッケージに用意されているチャンク(80/160/…は自前exportが必要)
KNOWN_CHUNKS = (320,)


def _materialize_snapshot(snapshot: str | Path) -> str:
    """HF snapshot のシンボリックリンクを実ファイルに展開したディレクトリを返す。

    onnxruntime は外部データ(*.onnx.data)のパスがモデルディレクトリ外
    (symlink 先の hub/blobs)へ出ると拒否する。ハードリンク(不可ならコピー)で
    snapshot と同階層の ``_resolved/<revision>`` に実体を置く。冪等。
    """
    src = Path(snapshot)
    dst = src.parent.parent / "_resolved" / src.name
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        target = dst / rel
        real = f.resolve()
        if target.exists() and target.stat().st_size == real.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            os.link(real, target)
        except OSError:
            shutil.copy2(real, target)
    return str(dst)


class NemotronOnnxAsr:
    engine_name = "nemotron"

    def __init__(
        self,
        model_id: str = DEFAULT_REPO,
        model_dir: str | None = None,
        chunk_ms: int = 320,
        precision: str = "fp16",
        language: str = "ja",
        num_threads: int = 0,
    ) -> None:
        self.model_id = model_id or DEFAULT_REPO
        self.model_dir = model_dir
        self.chunk_ms = int(chunk_ms or 320)
        self.precision = precision or "fp16"
        self.language = _normalize_language(language)
        # 0 = onnxruntime 既定(コア数をエンジンに任せる)
        self.num_threads = int(num_threads or 4)
        self._engine = None
        self.model_id_resolved = self.model_id

    def load(self, progress=None) -> None:
        try:
            import onnxruntime  # noqa: F401, PLC0415
        except ImportError as e:
            raise RuntimeError(
                "Nemotron エンジンには onnxruntime が必要です: uv sync を実行してください"
            ) from e

        model_dir = self._resolve_dir(progress)
        if progress is not None:
            progress(f"Nemotron ONNX 構築中: chunk={self.chunk_ms}ms {self.precision}")
        from .vendor.nemotron_onnx_streaming import NemotronOnnxStreaming  # noqa: PLC0415

        self._engine = NemotronOnnxStreaming(
            model_dir,
            language=self.language,
            chunk_ms=self.chunk_ms,
            precision=self.precision,
            num_threads=self.num_threads,
        )
        log.info(
            "ASR ready: Nemotron 3.5 ASR ONNX (%s, chunk=%dms, %s)",
            model_dir,
            self.chunk_ms,
            self.precision,
        )
        if progress is not None:
            progress(f"ASR準備完了: Nemotron 3.5 ASR (chunk={self.chunk_ms}ms)")

    def _resolve_dir(self, progress=None) -> str:
        if self.model_dir:
            return self.model_dir
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        if progress is not None:
            progress(f"ASRモデル取得中: {self.model_id}")
        d = _materialize_snapshot(snapshot_download(self.model_id))
        self.model_id_resolved = str(d)
        # 指定チャンクのグラフが無ければ HF パッケージ既定の 320ms に戻す
        # (他のチャンクは export/export_onnx.py による自前 export が必要)
        if self.chunk_ms not in KNOWN_CHUNKS and not any(
            p.name.startswith(f"encoder_{self.chunk_ms}ms") for p in Path(d).iterdir()
        ):
            log.warning(
                "chunk=%dms のグラフが %s に無いため 320ms に戻します(他チャンクは自前export)",
                self.chunk_ms,
                d,
            )
            self.chunk_ms = 320
        return str(d)

    def transcribe_utterance(self, audio: np.ndarray) -> str:
        return self._transcribe(audio)

    def transcribe_partial(self, audio: np.ndarray) -> str:
        # 発話途中バッファの再デコードにも全文デコードを使う(RNNT の仮説自体が強く、
        # flush 前の partial は末尾の取りこぼしがあるため)。
        return self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._engine is None:
            raise RuntimeError("ASR not loaded")
        pcm = np.ascontiguousarray(audio, dtype=np.float32)
        self._engine.reset()
        self._engine.accept_waveform(pcm)
        return self._engine.finish().strip()


def _normalize_language(lang: str | None) -> str:
    """backend 共通の language キー("ja" 等)を Nemotron の locale 辞書に合わせる。"""
    if not lang or lang == "auto":
        return "auto"
    l = lang.strip()
    if "-" in l:
        return l
    # 短いコードは主要 locale へ拡張(Nemotron は bare code も受け付けるが
    # 辞書に無いコードで落ちるため既定のlocaleを明示する)
    bare = {"ja": "ja-JP", "en": "en-US", "zh": "zh-CN", "ko": "ko-KR"}
    return bare.get(l, l)
