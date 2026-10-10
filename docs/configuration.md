# 上級設定とチューニング

GUI に UI の無い設定は `data/backend.json`(任意。環境変数 `STTTS_CONFIG` で別パス)に書くと、
起動時に既定値へマージされます(GUI から送られる設定はその上に適用)。例:

```json
{
  "asr": { "engine": "nemotron", "vad_min_silence_ms": 280 },
  "pipeline": { "first_chunk_mora_max": 12, "speculative_tts": false },
  "tts": { "warmup": true, "sampling": { "duration_scale": 1.1 } }
}
```

## 設定キー

| キー | 既定 | 説明 |
|---|---|---|
| `asr.engine` | `kotoba` | `kotoba`(kotoba-whisper、GPU)/ `nemotron`(ONNX、CPU)/ `gemini`(クラウド、要 APIキー)。GUI の「認識」ドロップダウンでも切替可 |
| `asr.model` | `kotoba-tech/kotoba-whisper-v2.0` | kotoba 用の HF リポジトリ |
| `asr.final_beam_size` | `2` | kotoba の確定デコードのビーム幅(1 にすると少し速い) |
| `asr.partial_interval_ms` | `800` | 途中経過デコードの間隔。`0` で無効 |
| `asr.preload` | `true` | 起動時に ASR をロードして「マイク開始」を即座に使えるようにする |
| `asr.vad_min_silence_ms` | `280` | 無音がこの長さ続いたら発話終了。短いほど速いが文中の間で切れやすい |
| `asr.vad_threshold` | `0.5` | Silero VAD のしきい値 |
| `asr.nemotron_model_dir` | `null` | Nemotron の ONNX ディレクトリ(null で HF から自動DL。自前exportグラフはここへ) |
| `asr.nemotron_chunk_ms` | `320` | ストリーミングチャンク。HF パッケージは 320 のみ |
| `asr.nemotron_precision` | `fp16` | `int8` は精度劣化するため非推奨 |
| `asr.nemotron_threads` | `4` | onnxruntime のスレッド数 |
| `asr.gemini_api_key` | `null` | AI Studio の API キー。**通常は GUI で入力する**(「認識」で Gemini を選ぶと「キー」欄が出る。`data/config.json` に Windows DPAPI で暗号化保存される)。GUI 未入力なら backend.json のこの値 → 環境変数 `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `asr.gemini_mode` | `VERBATIM` | 話し方を残す逐語転写。`SMART`=フィラー除去・句読点整形 |
| `asr.gemini_timeout_s` | `20` | 1発話の確定待ちタイムアウト |
| `pipeline.auto_speak` | `true` | 認識した文を自動で読み上げる(GUI の「自動再生」) |
| `pipeline.first_chunk_mora_min` / `max` | `8` / `12` | 先頭チャンクを読点または約 8〜12 モーラの文節境界で切る(`max=0` で無効) |
| `pipeline.chunk_min_chars` | `16` | 2チャンク目以降の最小文字数 |
| `pipeline.chunk_max_chars` | `80` | これを超える塊は読点 / 文節境界で分割(句読点の無い ASR 出力対策)。話し続けている間の逐次読み上げでも、文末が無いままこの長さを超えたら読点で切る |
| `pipeline.speculative_tts` | `false` | 投機的 TTS(下記) |
| `pipeline.speculative_stable_partials` | `2` | 同じ先頭チャンクが何回連続したら先行合成するか |
| `pipeline.performance_enabled` | `true` | 元音声の速さと間を Irodori の発話単位指示へ写す。GUI の「テンポと間を再現」で切替 |
| `pipeline.performance_wait_ms` | `150` | ASR 確定後に表現分析を待つ上限。超過時は表現を付けず発話 |
| `tts.warmup` | `true` | モデル決定時にロード + 短文合成を先行して初回の待ちを無くす |
| `tts.model` | `v4.1-small-mf` | 合成モデル。`v4.1-small-mf`(MeanFlow、4 ステップ、会話向け)、`v4.1-small`(RF、40 ステップ + CFG、高品質・低速)、`v4.1-small-int8`(その int8 版)、`v4-large`(33 億パラメータ、GPU メモリ 16GB 程度)、`v4-large-int8`(その int8 版)。`sttts-say model use` が書く。GUI は詳細設定での選択を使う |
| `tts.num_steps` | `null` | サンプラのステップ数(null でモデルの既定: MeanFlow 4、RF 40) |
| `tts.sampling` | `{}` | Irodori の `SamplingRequest` 項目を上書き(GUI 未対応でも使える)。使える項目: `num_steps` `duration_scale` `seconds` `min_seconds` `max_seconds` `max_ref_seconds` `ref_normalize_db` `ref_ensure_max` `trim_tail` `tail_window_size` `tail_std_threshold` `tail_mean_threshold` `watermark`、RF のモデルだけで効く `cfg_scale_text` `cfg_scale_caption` `cfg_scale_speaker` `cfg_scale` `cfg_guidance_mode`(independent / joint / alternating)`cfg_min_t` `cfg_max_t` `truncation_factor` `rescale_k` `rescale_sigma` `speaker_kv_scale` `speaker_kv_min_t` `speaker_kv_max_layers` `speaker_uncond_mode`(mask / noise)`t_schedule_mode`(linear / sway)`sway_coeff`(MeanFlow は原典と同じく無視する)。`text` / `caption` / `ref_*` / `no_ref` / `seed` は発話ごとにアプリが決めるため指定不可、知らない項目もエラーで知らせます(黙って捨てません)。設定変更は次の発話から反映 |

## 低遅延のための工夫

- **ASR を VAD スレッドから分離**: デコードは専用ワーカー。確定(final)を最優先し、
  途中経過(partial)は最新1件に合体。確定が来た発話の partial は待機中・処理中とも破棄。
  音声キューは無制限で、ASR が遅れても音声は捨てません。
- **マイクを先に開く(mic-first)**: ASR のロードと並行してマイクを開き、ロード中の音声は
  保持してロード完了後に流します。レベルメーターはロード中も止まりません。
- **話し続けている間も読み上げる(逐次読み上げ)**: partial に文末が出て次の文が始まったら、
  2 回続けて一致した範囲だけを先に Irodori へ送ります。1 発話 = 1 リクエストのまま、確定時は
  未読の残りだけを足して閉じるので、二度読みはしません。Nemotron は partial に発話の先頭から
  全音声を渡す(続きから再開するので軽い)ため、長い独白でも途切れずに追従します。
- **先頭チャンクを短く**: Irodori は1チャンク全体を一括生成するので、初音までの時間は
  先頭チャンクの長さにほぼ比例します。先頭だけ読点か約 8〜12 モーラで切り、以降は大きめに。
  全角 `！？` も文末として扱います。
- **seed はリクエスト単位**: ランダム seed のときも1回の発話内の全チャンクで同じ seed を
  使うため、参照音声なしでもチャンク間で声質が変わりません。
- **投機的 TTS(既定 OFF)**: 同じ先頭チャンクが `speculative_stable_partials` 回連続した
  partial から先頭チャンクを先行合成します。結果はエンジン内に保持し、確定文の先頭チャンク・
  声・設定が**完全一致したときだけ**再生に回します。不一致なら破棄するので、誤った
  音声が鳴ることはありません(外れた場合は GPU 時間を1チャンク分無駄にします)。

## レイテンシ KPI

主 KPI は **発話終了→初音** = 話し終わり → 最初の音声チャンクが再生キューに入るまで。
エンジンが計測フィールドをメッセージに載せ(`asr_final.vad_wait_ms` / `asr_ms`、
`tts_audio.first_chunk_ms` / `e2e_ms` / `rtf` / `stages` 等)、GUI は受信→再生キュー投入の時間を足して
ヘッダに表示します。内訳はおおよそ:

```
発話終了→初音 ≈ VAD 待ち(≈ vad_min_silence_ms) + ASR 確定デコード + TTS 先頭チャンク合成
```

## Nemotron 1120ms export(発話確定をさらに速く)

HF 配布の fp16 パッケージは 320ms チャンクのみ。chunk=1120ms は発話単位デコードがより速いので、
速さを優先する場合は自前で export します(CPU だけで可、ベースモデル ~2.5GB を DL):

```bash
git clone --depth 1 https://github.com/codavidgarcia/nemotron-3.5-asr-streaming-onnx
cd nemotron-3.5-asr-streaming-onnx
uv run --no-project --with "torch>=2.6" --with "transformers>=5.13.0" \
  --with "librosa>=0.10" --with "onnx>=1.17" --with "onnxruntime>=1.20" \
  --with "onnxmltools>=1.12" --with scipy --with soundfile --with "sentencepiece>=0.2" \
  python export/export_onnx.py --output-dir ../onnx-out --chunk-ms 1120 --validate
uv run --no-project --with "torch>=2.6" --with onnx --with onnxruntime \
  --with "onnxmltools>=1.12" python export/quantize.py --model-dir ../onnx-out --fp16
```

`encoder_1120ms_fp16.onnx(.data)` / `encoder_1120ms_first_fp16.onnx(.data)` と `decoder.onnx` /
`joiner.onnx` / `tokens.txt` / `nemotron_onnx_config.json` を `data/nemotron-onnx/`(gitignore 済み)へ
平置きし、`data/backend.json` に:

```json
{"asr": {"engine": "nemotron", "nemotron_model_dir": "data/nemotron-onnx", "nemotron_chunk_ms": 1120}}
```

(この export 作業だけは PyTorch が要ります。アプリの実行には不要です。)
参照実装のライセンスは Apache-2.0(`crates/nemotron/LICENSE`)、重みは OpenMDW-1.1(NVIDIA 所有・商用可)。
