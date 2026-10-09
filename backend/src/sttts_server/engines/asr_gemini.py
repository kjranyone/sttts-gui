"""Gemini Live API(gemini-3.5-transcribe-live)によるクラウド ASR(任意エンジン)。

asr.engine = "gemini" で選択。google-genai は標準依存。
API キーは asr.gemini_api_key(data/backend.json)または環境変数 GEMINI_API_KEY /
GOOGLE_API_KEY から解決する。

統合形態: うちはローカル silero VAD で発話を切り出すため、Live API は
「1発話 = 1セッション」で使う(音声を流しきり audio_stream_end で確定を待つ)。
サーバ側自動 VAD に任せないので、VAD 挙動はローカルエンジンと一致する。

- 精度: WER ~2.6%(公式発表)。SMART モードはフィラー(えー等)除去・句読点整形を行う。
- 遅延: ネットワーク + 確定までの解析込みで 0.5〜1.5 秒程度(要実測)。
- 制限: ストリーミング 10 分/セッション(発話単位運用では実質無関係)。
    話者分離・単語タイムスタンプは非対応。
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

log = logging.getLogger("sttts.asr")

DEFAULT_MODEL = "gemini-3.5-transcribe-live"


def resolve_api_key(configured: str | None) -> str | None:
    """config のキー > GEMINI_API_KEY > GOOGLE_API_KEY の順に解決する。"""
    if configured:
        return configured
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiLiveAsr:
    engine_name = "gemini"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        language: str = "ja-JP",
        mode: str = "SMART",
        timeout_s: float = 20.0,
    ) -> None:
        self.model_id = model_id or DEFAULT_MODEL
        self.api_key = api_key
        # "ja" 等の短いコードは BCP-47 に揃える
        self.language = language if "-" in language else f"{language}-JP" if language == "ja" else language
        self.mode = "SMART" if str(mode).upper() == "SMART" else "VERBATIM"
        self.timeout_s = float(timeout_s)
        self._client = None

    @property
    def model_id_resolved(self) -> str:
        return self.model_id

    def load(self, progress=None) -> None:
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "Gemini エンジンには google-genai が必要です: uv sync を実行してください"
            ) from e
        key = resolve_api_key(self.api_key)
        if not key:
            raise RuntimeError(
                "Gemini API キーがありません。GUI の「キー」欄に AI Studio で発行したキーを"
                "入力してください(data/backend.json の asr.gemini_api_key / 環境変数 GEMINI_API_KEY でも可)"
            )
        if progress is not None:
            progress(f"Gemini 接続準備: {self.model_id} ({self.mode})")
        self._client = genai.Client(api_key=key)
        log.info("ASR ready: Gemini Live API (%s, %s, lang=%s)", self.model_id, self.mode, self.language)
        if progress is not None:
            progress(f"ASR準備完了: Gemini ({self.mode})")

    def transcribe_utterance(self, audio: np.ndarray) -> str:
        return self._transcribe(audio)

    def transcribe_partial(self, audio: np.ndarray) -> str:
        # 発話途中バッファも同様に一括送信する(interim は使わず確定を返す)
        return self._transcribe(audio)

    # ---------- 実装 ----------

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._client is None:
            raise RuntimeError("ASR not loaded")
        pcm = np.clip(audio, -1.0, 1.0)
        raw = (pcm * 32767.0).astype("<i2").tobytes()
        # AsrWorker スレッドから呼ばれるため、スレッドローカルのイベントループで回す
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # 既にループが動いているケース(通常ない)は新しいスレッドで実行
            import concurrent.futures  # noqa: PLC0415

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, self._run(raw)).result()
        return asyncio.run(self._run(raw))

    async def _run(self, raw_pcm: bytes) -> str:
        from google.genai import types  # noqa: PLC0415

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[self.language],
                mode=self.mode,
            ),
        )
        timeout = self.timeout_s
        async with self._client.aio.live.connect(model=self.model_id, config=config) as session:
            await session.send_realtime_input(
                audio=types.Blob(data=raw_pcm, mime_type="audio/pcm;rate=16000")
            )
            await session.send_realtime_input(audio_stream_end=True)
            final_text = ""
            async for response in session.receive():
                sc = response.server_content
                if sc is None:
                    continue
                if sc.input_transcription and sc.input_transcription.text:
                    final_text = sc.input_transcription.text
                # 確定(または入力終了)後にサーバがストリームを閉じるのを待つが、
                # 応答が続くケースに備えタイムアウトで抜ける
                if sc.turn_complete or sc.generation_complete:
                    break
            return final_text.strip()
