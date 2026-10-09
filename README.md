# sttts-gui

**speech-to-text-to-speech GUI** — マイク音声をリアルタイムに文字起こしし(ストリーミングASR)、
確定文を文チャンクへ分割して [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) で逐次合成・再生する
Windows デスクトップアプリです。UI は Rust 製 **GPUI**(gpui-kit / Zed 系)、モデル実行は
**uv で管理された Python バックエンド**が担い、両者は stdio NDJSON で接続されます。

```
┌─────────────────────────────┐          ┌──────────────────────────────────┐
│ GUI (Rust / gpui-kit)       │  stdin   │ backend (Python / uv)            │
│  ・文字起こし表示(partial/final)│ ◀────── │  mic(48k)→soxr→16k→silero VAD    │
│  ・テキスト発話・モデル選択・履歴│  NDJSON  │  → ASRワーカー(kotoba CUDA/CPU    │
│  ・rodio でチャンクWAVを逐次再生│  ──────▶ │     または ReazonSpeech)          │
│  ・発話終了→初音 の計測表示     │          │  → 文チャンク分割 → TTSキュー     │
└─────────────────────────────┘          │  → irodori-tts (XPU / CUDA / CPU) │
                                         └──────────────────────────────────┘
```

Irodori-TTS は文単位の非ストリーミング合成のため、**確定文→句点等でチャンク分割→チャンクごとに
逐次合成→完成したチャンクから即座に送出・再生**という疑似ストリーミング構成を採用しています
(Irodori-TTS-Server の SSE モードと同じ発想)。

## 構成

| パス | 内容 |
|---|---|
| `crates/protocol/` | GUI⇄backend の NDJSON メッセージ型(Rust/serde) |
| `crates/gui/` | GPUI クライアント(gpui-kit 0.7 + rodio) |
| `backend/` | Python バックエンド(uv プロジェクト) |
| `backend/src/sttts_server/` | stdio サーバ本体・チャンク分割・エンジン実装 |
| `backend/src/sttts_server/engines/` | `tts_irodori` / `asr`(ファクトリ)/ `asr_whisper` / `asr_reazon` / `vad_silero` / `mic` / `wav_source` / `mock` |
| `backend/scripts/bench_latency.py` | レイテンシ計測ハーネス(P50/P90) |
| `output/` | 生成 WAV(チャンクごとに自動保存) |
| `data/config.json` | GUI 設定(モデル・キャプション・自動発話等) |
| `data/backend.json` | 任意。backend の上級設定(下記「低レイテンシ設定」) |

## セットアップ(Windows 11 + Intel Arc B570)

要件:

- Windows 11 / Intel Arc ドライバ **32.0.101.7028 以上**(XPU 用。古い場合は更新)
- Rust stable(cargo 1.99 で検証)/ MSVC ビルドツール
- [uv](https://docs.astral.sh/uv/) と git

```bat
:: 1) Python 環境(torch 2.10 XPU 含む。初回は数GBダウンロード)
cd backend
uv sync --extra xpu

:: 2) XPU 疎通確認
uv run --no-sync python -c "import torch; print(torch.xpu.is_available())"  :: True になること

:: 3) GUI ビルド(リポジトリルートへ戻る)
cd ..
cargo build --release
```

注意: 本リポジトリの Python 環境はグローバルに入れず、必ず上記 venv(uv)で隔離してください。

## セットアップ(NVIDIA GPU / CUDA 12.8)

RTX 20xx〜50xx 向け。PyTorch extra は **xpu / cu128 / cpu のどれか1つ**だけ指定します
(`[tool.uv] conflicts` で同時指定は禁止。切り替えるときは指定し直して `uv sync`)。

要件: NVIDIA ドライバ R570 以上(CUDA 12.8 ランタイムは torch wheel に同梱)。

```bat
cd backend
uv sync --extra cu128
:: 任意: ReazonSpeech ASR も使う場合
:: uv sync --extra cu128 --extra reazonspeech

:: 疎通確認(どちらも True / 1 以上になること)
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
uv run --no-sync python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

`dev.ps1` からは `.\dev.ps1 -Mode real -Sync -Backend cu128`。Linux / macOS / GPU なしは
`uv sync --extra cpu`(GUI の backend 探索は `backend/.venv/bin/python` にも対応)。

- **TTS**: `tts.device=auto` で CUDA を使います。`tts.precision=auto` は compute capability
  8.0 未満(RTX 2080Ti = sm_75 など Turing)では bf16 のハード支援が無いため **fp32**、
  Ampere 以降は bf16。XPU は従来どおり bf16。
- **ASR(kotoba-whisper)**: `asr.device=auto` で CTranslate2 が CUDA を認識すれば
  **cuda / float16**、無ければ従来どおり cpu / int8。CUDA 初期化に失敗した場合
  (Windows で cuDNN / cuBLAS の DLL が見つからない等)は自動で cpu / int8 に
  フォールバックし、ログに理由を出します。Windows では torch 同梱の DLL
  (`torch/lib`)を CTranslate2 の検索パスへ追加しています。
- Intel Arc 環境では CTranslate2 が XPU 非対応のため ASR は CPU です
  (速度が必要なら ReazonSpeech エンジンを検討)。

## 実行

開発用起動スクリプト `dev.ps1`(Windows PowerShell)。**`-Mode` を省略すると起動モードを対話式で尋ねます(空欄で real)**:

```powershell
.\dev.ps1                        # モードを対話式で選択(1: real / 2: mock、空欄で real)
.\dev.ps1 -Mode real             # 実エンジンモードを直接指定(対話なし・自動化向け)
.\dev.ps1 -Mode real -Sync       # uv sync --extra xpu してから起動
.\dev.ps1 -Mode real -Sync -Backend cu128   # NVIDIA GPU 用に同期してから起動
.\dev.ps1 -DebugBuild            # debug プロファイルで起動
```

スクリプトは 前提確認(cargo / uv)→ 必要時のみ `uv sync --extra <Backend>`(既定 xpu)→
`cargo build`(毎回実行。変更がなければ差分ビルドで数秒)→ GUI 起動、を行います。
コード編集後に古いバイナリが起動することはありません。

直接起動する場合:

```bat
:: モックモード(モデルDLなし。UI/配線の確認用)
cargo run --release -p sttts-gui -- --mock

:: 実エンジンモード(初回発話時に Irodori モデル ≈3GB と ASR モデル ≈1GB を自動DL)
cargo run --release -p sttts-gui -- --real
```

使い方:

1. **入力/出力デバイス**を右パネル上部のセレクタで選択(選択は設定に保存され、
   入力デバイス変更時はマイクセッションが自動で張り直されます)
2. **入力レベルメーター**がマイク稼働中は dB 表示とバーで入力レベルを可視化します
   (ASR に音が入っているかの確認用。緑→黄→赤)
3. 「発話」パネルにテキストを入力し**発話**ボタン → チャンク合成されて順次再生されます
4. **マイク開始** → 話すと部分文字起こし(灰色)→ 確定文(白)が表示され、
   「ASR確定文を自動発話」が ON なら確定文がそのまま TTS されます
5. モデルは右パネルで切替(`tts.warmup` が有効なら選択直後にロード + 短文合成でウォームアップ)
6. 「キュー取消」は再生中・未再生の音声と未合成チャンクを破棄します。合成中の1チャンクは
   Irodori の仕様上中断できませんが、完了後に結果を捨てるので鳴りません
7. ヘッダ右側に **発話終了→初音**(話し終わってから最初の音声が再生キューに入るまで、
   直近値と直近20回の中央値)と、ASR確定 / TTS初チャンク / RTF を表示します

> **ヘッドホン推奨**: マイクは TTS 再生中も開いたままです。スピーカー再生だと合成音声を
> マイクが拾い、それが文字起こし → 自動発話されてループします(エコーキャンセル未実装)。
> デモ・収録はヘッドホンで行ってください。

### バックエンド単体のセルフチェック

```bat
cd backend
uv run --no-sync python -m sttts_server --self-check-tts "こんにちは" --model v4.1-small-mf
uv run --no-sync python -m sttts_server --self-check-asr
```

`--self-check-tts` は Irodori の段階別時間(`stages`)も出力します。
`--self-check-asr` は実際に使われた device / compute_type を出力します。

## 低レイテンシ設定

### 声のバンク(voice cloning)

`data/voices/` に参照音声の wav(10秒程度・話者の声)を置くと、GUI の「声」ドロップダウンから
選択できるようになります(📁 ボタンでフォルダを開けます)。選択すると Irodori-TTS は
その音声を話者参照として合成し、話し方を模倣します。未選択(既定の声)は
キャプション/自動音質での合成になります。選択は `data/config.json` に保存されます。

注意: 参照音声は本人の同意のある声のみ使ってください(モデルカードの利用制限参照)。

GUI に UI の無い設定は `data/backend.json`(任意。`STTTS_CONFIG` 環境変数または
`--config` で別パス)に書くと、backend 起動時に既定値へマージされます
(GUI から送られる設定はその上に適用)。例:

```json
{
  "asr": { "engine": "kotoba", "device": "auto", "vad_min_silence_ms": 280 },
  "pipeline": { "first_chunk_mora_max": 12, "speculative_tts": false },
  "tts": { "precision": "auto", "warmup": true }
}
```

| キー | 既定 | 説明 |
|---|---|---|
| `asr.engine` | `kotoba` | `kotoba`(faster-whisper)/ `reazonspeech`(sherpa-onnx、要 `--extra reazonspeech`)/ `nemotron`(onnxruntime、要 `--extra nemotron`)/ `gemini`(クラウド、要 `--extra gemini` + APIキー)。GUI の「認識」ドロップダウンでも切替可 |
| `asr.device` / `asr.compute_type` | `auto` / `auto` | kotoba 用。auto = CUDA なら cuda/float16、無ければ cpu/int8 |
| `asr.cpu_threads` | `0` | CTranslate2 の CPU スレッド数(0 = 既定) |
| `asr.final_beam_size` | `2` | 確定デコードのビーム幅(1 にすると少し速い) |
| `asr.partial_interval_ms` | `800` | 途中経過デコードの間隔。`0` で無効(CPU kotoba では確定の待ちを減らせる) |
| `asr.preload` | `true` | 起動時に ASR をロードして「マイク開始」を即座に使えるようにする |
| `asr.vad_min_silence_ms` | `280` | 無音がこの長さ続いたら発話終了(従来 400)。短いほど速いが文中の間で切れやすい |
| `asr.vad_threshold` | `0.5` | silero VAD のしきい値 |
| `asr.reazon_model_dir` | `null` | ReazonSpeech のモデルディレクトリ(null で HF から自動DL) |
| `asr.reazon_precision` | `fp32` | `int8` は短い発話で崩れやすいので非推奨 |
| `asr.reazon_threads` | `4` | ReazonSpeech の CPU スレッド数 |
| `asr.nemotron_model_dir` | `null` | Nemotron の ONNX ディレクトリ(null で HF から自動DL。自前exportグラフはここへ) |
| `asr.nemotron_chunk_ms` | `320` | ストリーミングチャンク。HF パッケージは 320 のみ(1120 は発話確定がさらに速い。下記「Nemotron 1120ms export」参照) |
| `asr.nemotron_precision` | `fp16` | `int8` は dynamic quantum で精度劣化するため非推奨 |
| `asr.nemotron_threads` | `4` | onnxruntime の intra_op スレッド数 |
| `asr.gemini_api_key` | `null` | AI Studio の API キー(null なら環境変数 `GEMINI_API_KEY` / `GOOGLE_API_KEY`) |
| `asr.gemini_mode` | `SMART` | `SMART`=フィラー除去・句読点整形 / `VERBATIM`=逐語 |
| `asr.gemini_timeout_s` | `20` | 1発話の確定待ちタイムアウト |
| `pipeline.first_chunk_mora_min` / `max` | `8` / `12` | 先頭チャンクを読点または約 8〜12 モーラの文節境界で切る(`max=0` で無効) |
| `pipeline.chunk_min_chars` | `16` | 2チャンク目以降の最小文字数 |
| `pipeline.chunk_max_chars` | `80` | これを超える塊は読点 / 文節境界で分割(句読点の無い ASR 出力対策) |
| `pipeline.speculative_tts` | `false` | 投機的 TTS(下記) |
| `pipeline.speculative_stable_partials` | `2` | 同じ先頭チャンクが何回連続したら先行合成するか |
| `tts.precision` | `auto` | `auto` / `fp32` / `bf16`(auto: CUDA cc<8.0 → fp32、cc≥8.0・XPU → bf16、CPU → fp32) |
| `tts.warmup` | `true` | モデル決定時にロード + 短文合成を先行して初回の待ちを無くす |
| `tts.cache_conditions` | `true` | text / caption / 話者エンコードのメモ化(下記) |
| `tts.ref_latent_cache` | `true` | 参照 WAV の DACVAE latent をキャッシュ(`~/.cache/sttts-gui/ref_latents`、Windows は `%LOCALAPPDATA%\sttts-gui\cache`) |
| `tts.compile` | `false` | `torch.compile`。初回が遅く、Windows では triton が必要。有効時はメモ化を無効化 |

### 何が速くなったか

- **ASR を VAD スレッドから分離**: デコードは専用ワーカー。確定(final)を最優先し、
  途中経過(partial)は最新1件に合体。確定が来た発話の partial は待機中・処理中とも破棄。
  音声キューは無制限で、ASR が遅れても音声は捨てません(従来は満杯時に破棄)。
  なお Irodori と同様 CTranslate2 も途中中断できないため、partial のデコード中に発話が
  終わると確定はその完了を待ちます(GPU なら数百 ms 以下、CPU kotoba では数秒)。
- **先頭チャンクを短く**: Irodori は1チャンク全体を一括生成するので、初音までの時間は
  先頭チャンクの長さにほぼ比例します。先頭だけ読点か約 8〜12 モーラで切り、以降は大きめに。
  末尾の短い余りを前のチャンクへ連結する処理は廃止(最終チャンクが長くなるため)。
  全角 `！？` も文末として扱います。
- **seed はリクエスト単位**: ランダム seed のときも1回の発話内の全チャンクで同じ seed を
  使うため、参照音声なしでもチャンク間で声質が変わりません。
- **投機的 TTS(既定 OFF)**: 同じ先頭チャンクが `speculative_stable_partials` 回連続した
  partial から先頭チャンクを先行合成します。結果は backend 内に保持し、確定文の先頭チャンク・
  声・モデル設定が**完全一致したときだけ**再生に回します。不一致なら破棄するので、誤った
  音声が鳴ることはありません(外れた場合は GPU 時間を1チャンク分無駄にします)。
- **TTS の固定コスト削減(Irodori 本体は無改変)**: 参照 WAV の再エンコードをキャッシュ、
  1合成内で2回走る条件エンコード(尺予測とサンプラ)をメモ化で1回に、Turing では fp32。
  CPU 実機(下記)で、メモ化・latent キャッシュの有無で**出力波形がビット一致**することを確認済み。
  `encode_conditions` 自体の二重呼び出し解消は Irodori 本体の改変が必要なため対象外(今後の課題)。

## レイテンシ KPI と計測

主 KPI は **発話終了→初音** = 話し終わり → 最初の音声チャンクが再生キューに入るまで。
backend が計測フィールドを NDJSON に載せ(`asr_final.vad_wait_ms` / `asr_ms`、
`tts_audio.first_chunk_ms` / `e2e_ms` / `rtf` / `stages` 等、`protocol.py` の
`TIMING_FIELDS` 参照)、GUI は受信→再生キュー投入の時間を足してヘッダに表示します。
内訳はおおよそ:

```
発話終了→初音 ≈ VAD 待ち(≈ vad_min_silence_ms) + ASR 確定デコード + TTS 先頭チャンク合成 (+ 転送・キュー待ち)
```

計測ハーネス(固定 WAV を実時間ペースでマイク代わりに流す。VAD・ASR ワーカー・
チャンク分割・TTS ワーカーは本番と同じ経路):

```bat
cd backend
:: モデル不要(silero VAD は本物、ASR/TTS はモック、合成音声信号を使用)
uv run --no-sync python scripts/bench_latency.py --mode mock
:: 実 ASR + モック TTS(ASR 単体の寄与を測る)
uv run --no-sync python scripts/bench_latency.py --mode real --mock-tts --asr-engine kotoba --wav 録音.wav
:: 実 ASR + 実 Irodori(GPU 機での本番計測)
uv run --no-sync python scripts/bench_latency.py --mode real --asr-engine kotoba --wav 録音.wav --repeat 3 --json result.json
```

P50 / P90 を表示します(`--speculative`、`--vad-min-silence-ms`、`--partial-interval-ms`、
`--first-mora-max` で設定を変えて比較可能)。

#### 参考実測(CPU のみ・GPU 未計測)

計測環境: Linux / 8 vCPU Xeon(共有マシンのため数値は揺れます)。入力は日本語講演音声 207 秒
(VAD で 20 発話)、TTS はモック(固定 50ms)にして ASR と配線の寄与だけを測ったもの。
**GPU(CUDA / XPU)での数値はまだ測っていません**。

| 構成(すべて CPU) | VAD 待ち P50 | ASR 確定 P50 / P90 | 発話終了→先頭送出 P50 / P90 |
|---|---|---|---|
| kotoba int8、partial 800ms | 312 ms | 3706 / 4426 ms | 5995 / 7536 ms |
| kotoba int8、partial 無効 | 312 ms | 3644 / 3852 ms | 4013 / 4317 ms |
| ReazonSpeech fp32(4 threads) | 312 ms | 240 / 401 ms | 604 / 838 ms |
| ReazonSpeech + 投機的 TTS | 312 ms | 238 / 391 ms | 590 / 784 ms(先頭チャンク 20 中 11 で投機ヒット) |

- CPU の kotoba は partial のデコード中に確定が待たされるため、partial 無効の方が約 2 秒速い
  (GPU では1デコードが短いので差は小さくなる見込み。要実機確認)。
- モックモード(合成音声信号 + 実 silero VAD + モック ASR/TTS)では VAD 待ち P50 306 ms、
  発話終了→先頭送出 P50 358 ms。

Irodori v4.1-Small-MF(CPU fp32)での TTS オーバーヘッド削減の確認(同一 seed):

- 条件エンコードのメモ化 ON/OFF、参照 WAV 直接 / latent キャッシュ経由で **出力波形がビット一致**
- 参照音声(10 秒)ありの `prepare_reference`: 3271 ms → キャッシュヒット時 0.8 ms
- 同じ文の2回目の合成では `predict_duration` がほぼ 0 になる(条件エンコードがキャッシュに乗るため)

## モデルカタログ

| エイリアス | 実体 | 備考 |
|---|---|---|
| `v4.1-small-mf` | Aratako/Irodori-TTS-v4.1-Small-MF | MeanFlow 4steps。**既定**(B570 で RTF≈0.9 を実測) |
| `v4.1-small` | Aratako/Irodori-TTS-v4.1-Small | RF 40steps |
| `v4-large` | Aratako/Irodori-TTS-v4-Large | 高品質。bf16 で VRAM ≈6.6GB+、生成は遅め |
| `v4.1-small-int8` / `v4-large-int8` | 量子化版(subfolder int8-weight-only) | OOM 時のフォールバック |

ASR(`asr.engine`):

| エンジン | 実体 | 備考 |
|---|---|---|
| `kotoba`(既定) | `kotoba-tech/kotoba-whisper-v2.0-faster`(CTranslate2) | CUDA があれば float16、無ければ CPU int8。句読点は出ない |
| `reazonspeech` | `reazon-research/reazonspeech-k2-v2`(sherpa-onnx、Apache-2.0) | CPU でも非常に速い(下表)。**句読点なし**・**固有名詞/英字略語に弱い**(例:「NLP」→「エネルギー」)・**int8 は短い発話で崩れる**ので fp32 推奨。`uv sync --extra <torch extra> --extra reazonspeech` |
| `nemotron` | `nemotron-3.5-asr-streaming-0.6b` の ONNX export(cache-aware FastConformer-RNNT / onnxruntime、コード Apache-2.0 / 重み OpenMDW-1.1) | **句読点をネイティブ出力**・whisper large-v3 級の精度・発話確定 **平均 0.31 秒 / 最大 0.51 秒**(i5-12600KF、chunk=1120ms fp16 実測。chunk=320ms は平均 0.56 秒)。モデル ~2.5GB(fp16)。`uv sync --extra <torch extra> --extra nemotron`。ストリーディングエンジンは `engines/vendor/nemotron_onnx_streaming.py` として同梱。**既知の弱点: 母音のみの連続(「あいうえお」等)を正しく認識しない**(直渡しでも「i」等に潰れる。kotoba は「アイウエオ」と認識。2026-10-09 検証)。通常の発話(子音を含む)では影響なし |
| `gemini` | Google AI Studio「Gemini 3.5 Transcribe Live」(`gemini-3.5-transcribe-live` / Live API WebSocket) | **クラウド**。WER ~2.6%、SMART モードでフィラー(えー等)除去・句読点整形。発話単位で API へ送信(ローカル VAD はそのまま)。1発話 0.5〜1.5 秒程度(要実測)。話者分離・単語タイムスタンプ非対応。`uv sync --extra <torch extra> --extra gemini` + API キー |

VAD は silero-vad(ONNX)。VAD 発話終了時にバッファ全体を再デコードして確定文を作り、
発話中は partial_interval_ms(既定800ms)ごとに部分表示を更新します。
kotoba / reazonspeech は句読点を出さないため、チャンク分割は文節境界の近似と長さで切ります
(nemotron のみ句読点をネイティブ出力するため、読点で綺麗に切れます)。

### Nemotron 1120ms export(発話確定をさらに速く)

HF 配布の fp16 パッケージは 320ms チャンクのみ。発話単位デコードには
chunk=1120ms(RTF 約 0.14、i5-12600KF 実測: 発話確定 平均 0.31s / 最大 0.51s)が最速なので、
速さを優先する場合は下記で自前 export する(CPU だけで可、ベースモデル ~2.5GB を DL):

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

`encoder_1120ms_fp16.onnx(.data)` / `encoder_1120ms_first_fp16.onnx(.data)`(各ディレクトリ内)
と `decoder.onnx` / `joiner.onnx` / `tokens.txt` / `nemotron_onnx_config.json` を
`data/nemotron-onnx/`(gitignore 済み)へ平置きし、`data/backend.json` に:

```json
{"asr": {"engine": "nemotron", "nemotron_model_dir": "data/nemotron-onnx", "nemotron_chunk_ms": 1120}}
```

参照実装のライセンスは Apache-2.0(vendor 同梱の LICENSE ファイル参照)、
重みは OpenMDW-1.1(NVIDIA 所有・商用可)。

## テスト

```bat
cargo test            :: プロトコル単体(計測フィールド含む)+ rodio Sink のキャンセル + モックバックエンドとの往復統合テスト
cd backend && uv run --no-sync pytest   :: チャンク分割・ASR ワーカー・投機的 TTS・キャンセル・計測フィールド・設定等
```

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `torch.xpu.is_available()` が False | Intel ドライバを 32.0.101.7028+ へ更新 |
| TTS でメモリ不足(OOM) | モデルを INT8 版へ / `decode_mode` sequential 維持 / codec の CPU 追い出し(将来設定化) |
| ASR が CUDA にならない | ログの「ASR ... フォールバック」を確認。`uv run --no-sync python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"` が 0 なら cu128 extra / ドライバを確認。`asr.device` を `cuda` にすると失敗理由がログに出ます |
| 文中の短い間で発話が切れる | `asr.vad_min_silence_ms` を 350〜400 に戻す |
| スピーカーで自分の合成音声を拾ってループする | ヘッドホンを使う(マイクは再生中も開いています) |
| 最初の発話まで数分かかる | 正常です。irodori の依存(torch/transformers 等)の import とモデル構築に冷起動で数分かかります。`tts.warmup`(既定 ON)により起動直後からバックグラウンドでロードが始まります |
| マイクを開けない / レートエラー | バックエンドは既定レート(通常48kHz)で開き soxr で16kへ変換します。他アプリの排他占有を解除 |
| ASR の部分表示・確定が遅い | GPU なら cu128 extra で kotoba を CUDA に。CPU なら `asr.engine: "reazonspeech"`、または `asr.partial_interval_ms: 0` で partial を止めて確定を優先 |
| 発話が止まって進まない | バックエンドログ(GUI 下段)を確認。モデル初回 DL 中は待つ必要があります |
| タスクマネージャに python.exe が残る | GUI を強制終了した場合は子のバックエンドが孤児化します。正常終了は「終了」ボタンから。孤児は手動で終了してください |

## ライセンス・クレジット

- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)(コード: MIT)
  - モデルのライセンスはチェックポイントごとに異なります。配布・利用の際は各モデルカードを確認してください
    - [v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) / [v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) / [v4.1-Small-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-Quantized): MIT
    - [v4-Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large) / [v4-Large-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized): **Gemma Terms of Use**
    - コーデック [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim): MIT
  - 上記 Irodori-TTS の各モデルカードには、ライセンスに加えて倫理的な利用制限があります(本人の同意なく声優・著名人など実在人物の声を複製・なりすましに使わない、誤情報やディープフェイク目的に使わない 等)
- [kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)(Apache-2.0)/ 本アプリが使う CTranslate2 変換版 [kotoba-whisper-v2.0-faster](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster)(MIT)
- [silero-vad](https://github.com/snakers4/silero-vad)(MIT)
- gpui-kit(Apache-2.0)/ Zed GPUI(Apache-2.0)
- 本リポジトリのコード: ライセンスは未確定(リポジトリオーナーが確定してください)
